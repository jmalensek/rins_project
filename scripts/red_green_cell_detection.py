#!/usr/bin/env python3

'''
node, pasivno teče v ozadju.

gleda po tleh, ko vidi:
- tla - to ignorira
- rumeno črto - to ignorira
- rdečo črto - če jo je že zaznal, jo ignorira, če jo še ni, shrani koordinate
- zeleno črto - če jo je že zaznal, jo ignorira, če jo še ni, shrani koordinate

od task_managerja dobi call, da se je task začel -> če že ima koordinate za ta pravi cell - oz. kadar jih dobi:

gre do cella, na desni konec, se premakne, da gleda proti drugemu koncu

extenda kamero - arm_mover - desno zgoraj

- pošlje signal tile_detection, da lahko začne z delom

- se počasi premakne proti drugemu koncu, ko ne vidi več zelene/rdeče linije, konča - pošlje signal tile detection, da je konec
'''

import math
import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistStamped, PointStamped
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import tf2_ros
from tf2_ros import TransformException
from tf2_geometry_msgs import do_transform_point


class CellState(Enum):
    IDLE = auto()
    MOVING_TO_CELL = auto()
    ALIGNING = auto()
    FOLLOWING_TO_FAR_END = auto()
    TURNING_180 = auto()
    PREPARING_TILE_DETECTION = auto()
    FOLLOWING_BACK = auto()
    DONE = auto()


class RedGreenCellDetection(Node):
    def __init__(self):
        super().__init__('red_green_cell_detection')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_topic', '/oakd/rgb/preview/image_raw'),
                ('task_start_topic', '/task_manager/task_started'),
                ('tile_detection_start_topic', '/tile_detection/start'),
                ('tile_detection_stop_topic', '/tile_detection/stop'),
                # koliko zaporednih framov brez linije = "izgubljena"
                ('line_lost_frames_threshold', 10),
                # minimalna površina konture, da jo upoštevamo (px²)
                ('min_contour_area', 500),
                # hitrosti za posamezne faze
                ('approach_linear_speed', 0.08),
                ('align_angular_speed', 0.3),
                ('follow_linear_speed', 0.06),
                ('follow_angular_gain', 0.005),   # P-gain: rad/s na px offseta
                ('follow_max_angular_speed', 0.4),
                ('turn_180_angular_speed', 0.4),
                # kot (v stopinjah) znotraj katerega štejemo linijo za "poravnano" (navpično)
                ('alignment_tolerance_deg', 6.0),
            ],
        )

        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.task_start_topic = self.get_parameter('task_start_topic').get_parameter_value().string_value
        self.tile_detection_start_topic = self.get_parameter('tile_detection_start_topic').get_parameter_value().string_value
        self.tile_detection_stop_topic = self.get_parameter('tile_detection_stop_topic').get_parameter_value().string_value
        self.line_lost_threshold = self.get_parameter('line_lost_frames_threshold').get_parameter_value().integer_value
        self.min_contour_area = self.get_parameter('min_contour_area').get_parameter_value().integer_value

        self.approach_linear_speed = self.get_parameter('approach_linear_speed').get_parameter_value().double_value
        self.align_angular_speed = self.get_parameter('align_angular_speed').get_parameter_value().double_value
        self.follow_linear_speed = self.get_parameter('follow_linear_speed').get_parameter_value().double_value
        self.follow_angular_gain = self.get_parameter('follow_angular_gain').get_parameter_value().double_value
        self.follow_max_angular_speed = self.get_parameter('follow_max_angular_speed').get_parameter_value().double_value
        self.turn_180_angular_speed = self.get_parameter('turn_180_angular_speed').get_parameter_value().double_value
        self.alignment_tolerance_deg = self.get_parameter('alignment_tolerance_deg').get_parameter_value().double_value

        # po pridobitvi pointcloud-a se to spremeni v map-frame (x, y, z) tocko,
        # ne vec pixel (cx, cy) - glej pixel_to_map_coords / pointcloud_callback
        self.red_cell_coord: tuple[float, float, float] | None = None
        self.green_cell_coord: tuple[float, float, float] | None = None
        self.task_active = False
        self.search_active = False

        # state machine
        self.state = CellState.IDLE
        self._turn_start_time = None
        self._turn_target_duration = None

        # števec framov, v katerih aktivna linija ni bila vidna
        self._frames_without_line = 0

        # zadnji prejeti pointcloud (organiziran, height x width kot kamera slika)
        # in zacasno shranjeni pixel centroidi, ki cakajo na uskladitev s
        # pointcloud-om/TF (ce pointcloud/TF se ni na voljo ob detekciji)
        self._latest_pointcloud: PointCloud2 | None = None
        self._pending_red_pixel: tuple[int, int] | None = None
        self._pending_green_pixel: tuple[int, int] | None = None

        self.bridge = CvBridge()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.camera_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.camera_callback,
            qos_profile_sensor_data,
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/oakd/rgb/preview/depth/points',
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )

        # sub na topic od task managerja - ta bo tudi povedal katero barvo mora gledati
        self.task_start_sub = self.create_subscription(
            Bool,
            self.task_start_topic,
            self.on_task_started,
            10,
        )

        self.color_of_the_cell = None

        self.color_cell_sub = self.create_subscription(
            String,
            '/task_manager/color_of_the_cell',
            self.color_of_the_cell_callback,
            10,
        )

        self.tile_start_pub = self.create_publisher(Bool, self.tile_detection_start_topic, 10)
        self.tile_stop_pub = self.create_publisher(Bool, self.tile_detection_stop_topic, 10)
        self.arm_mover_pub = self.create_publisher(String, '/arm_command', 10)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)

        self.get_logger().info('red_green_cell_detection node initialized')

    def color_of_the_cell_callback(self, msg: String):
        self.color_of_the_cell = msg.data
        self.get_logger().info(
            f"Color of the cell received: {self.color_of_the_cell}"
        )

    def _active_color(self) -> str | None:
        """Vrne barvo, ki jo trenutno štejemo za 'aktivno' linijo za ta task."""
        if self.color_of_the_cell in ('red', 'green'):
            return self.color_of_the_cell
        if self.red_cell_coord is not None:
            return 'red'
        if self.green_cell_coord is not None:
            return 'green'
        return None

    def _has_coordinates(self) -> bool:
        if self.color_of_the_cell == "red":
            return self.red_cell_coord is not None
        if self.color_of_the_cell == "green":
            return self.green_cell_coord is not None
        return self.red_cell_coord is not None or self.green_cell_coord is not None

    def _active_line_visible(self, image: np.ndarray) -> bool:
        """Vrne True, če je vsaj ena od zaznanih linij še vidna v kadru.

        Uporablja detect_color_for_navigation (brez aspect_ratio filtra),
        ker se to kliče med manevrom, ko robot gleda vzdolž črte.
        """
        if self.red_cell_coord is not None:
            result = self.detect_color_for_navigation(image, 'red')
            if result is not None:
                return True
        if self.green_cell_coord is not None:
            result = self.detect_color_for_navigation(image, 'green')
            if result is not None:
                return True
        return False

    def classify_lab(self, L, A, B):
        # OpenCV vraca A,B kanala v razponu 0-255 z nevtralno tocko pri 128
        # (ne 0!). Treba jih je centrirati, drugace je chroma/hue popacen.
        A = A - 128.0
        B = B - 128.0
        chroma = np.sqrt(A ** 2 + B ** 2)
        if chroma < 15:
            return "other"
        hue = np.degrees(np.arctan2(B, A)) % 360
        if hue < 42 or hue >= 330:
            return "red"
        elif 75 <= hue < 100:
            return "green"
        else:
            return "other"

    def detect_line_of_color(self, image: np.ndarray, color: str) -> tuple[int, int] | None:
        """
        Detect a red or green line on the floor.
        Returns the center (cx, cy) of the largest valid contour.

        Namen te funkcije je IZKLJUČNO enkratna pasivna zaznava začetnih
        koordinat celice (klicana iz process_frame). Ni mišljena za sledenje
        črti ali poravnavo med manevrom - zato filtrira "linijsko" obliko
        (aspect_ratio) na način, ki ustreza prvotnemu opažanju črte, ne pa
        gledanju vzdolž nje. Za navigacijo/sledenje uporabi
        detect_color_for_navigation.
        """

        h, w = image.shape[:2]

        # Only look at the bottom part of the image
        floor_band_height = 80
        roi = image[h-floor_band_height:h, :]

        # Convert ROI to LAB
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        # OpenCV vraca A,B kanala v razponu 0-255 z nevtralno tocko pri 128
        # (ne pri 0). Treba jih je centrirati, drugace sta chroma in hue
        # popacena za vse piksle.
        A = lab[:, :, 1] - 128.0
        B = lab[:, :, 2] - 128.0

        chroma = np.sqrt(A**2 + B**2)
        hue = np.degrees(np.arctan2(B, A)) % 360

        if color == 'red':
            mask = (chroma >= 15) & ((hue < 42) | (hue >= 330))
        elif color == 'green':
            mask = (chroma >= 15) & (hue >= 75) & (hue < 120)
        else:
            return None

        mask = mask.astype(np.uint8) * 255

        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(largest)
        if area < self.min_contour_area:
            return None

        # Reject objects that are not line-like
        x, y, w_box, h_box = cv2.boundingRect(largest)

        aspect_ratio = w_box / h_box

        if aspect_ratio < 2:
            self.get_logger().info(
                f"Rejected {color} object because it is too tall "
                f"(w={w_box}, h={h_box})"
            )
            return None

        M = cv2.moments(largest)

        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy_roi = int(M['m01'] / M['m00'])

        # Convert back to image coordinates
        cy = cy_roi + (h - floor_band_height)

        self.get_logger().info(
            f"Detected {color} line at ({cx}, {cy}) "
            f"(hue={hue[cy_roi, cx]:.1f}, chroma={chroma[cy_roi, cx]:.1f}"
            f"(area={area:.0f}, aspect_ratio={aspect_ratio:.2f})"
        )

        return (cx, cy)

    def detect_color_for_navigation(self, image: np.ndarray, color: str):
        """
        Zaznava barvnega madeža (blob) za potrebe poravnave in sledenja črti
        (faze ALIGNING, FOLLOWING_TO_FAR_END, FOLLOWING_BACK ter preverjanje
        vidnosti med approachom).

        Za razliko od detect_line_of_color NE filtrira po aspect_ratio, ker
        med sledenjem/poravnavo robot gleda VZDOLŽ črte - kontura je takrat
        pričakovano ozka in visoka (oz. trapezoidna zaradi perspektive), kar
        bi aspect_ratio filter v detect_line_of_color napačno zavrnil.

        Returns a dict with:
            'center': (cx, cy) v polnih slikovnih koordinatah
            'angle_deg': odklon glavne osi konture od navpičnice v sliki
                         (0.0 = kontura je navpična, robot gleda vzdolž črte)
            'offset_x': cx - sredina_slike_x (predznačen horizontalni odklon)
        ali None, če barva ni zaznana.
        """
        h, w = image.shape[:2]

        floor_band_height = 80
        roi = image[h - floor_band_height:h, :]

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        A = lab[:, :, 1] - 128.0
        B = lab[:, :, 2] - 128.0

        chroma = np.sqrt(A ** 2 + B ** 2)
        hue = np.degrees(np.arctan2(B, A)) % 360

        if color == 'red':
            mask = (chroma >= 15) & ((hue < 42) | (hue >= 330))
        elif color == 'green':
            mask = (chroma >= 15) & (hue >= 75) & (hue < 120)
        else:
            return None

        mask = mask.astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < self.min_contour_area:
            return None

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy_roi = int(M['m01'] / M['m00'])
        cy = cy_roi + (h - floor_band_height)

        rect = cv2.minAreaRect(largest)
        (rect_w, rect_h) = rect[1]
        raw_angle = rect[2]

        # Normaliziramo tako, da 0 deg pomeni "glavna os je navpicna v sliki".
        if rect_w < rect_h:
            angle_from_vertical = raw_angle
        else:
            angle_from_vertical = raw_angle + 90.0

        if angle_from_vertical > 90:
            angle_from_vertical -= 180
        elif angle_from_vertical < -90:
            angle_from_vertical += 180

        offset_x = cx - (w / 2.0)

        self.get_logger().debug(
            f"[nav] {color} blob at ({cx},{cy}) area={area:.0f} "
            f"angle_from_vertical={angle_from_vertical:.1f} offset_x={offset_x:.1f}"
        )

        return {
            'center': (cx, cy),
            'angle_deg': angle_from_vertical,
            'offset_x': offset_x,
        }

    def pointcloud_callback(self, data: PointCloud2):
        """
        Pasivno shrani zadnji prejeti pointcloud. NE dela nobene detekcije ali
        TF transformacije tukaj - to je samo subscription handler, ki teče na
        vsak prejet pointcloud sporočilo. Dejanska pretvorba pixel -> map se
        zgodi v pixel_to_map_coords, klicana iz _try_resolve_pending_coords,
        ko imamo (cx, cy) iz detect_line_of_color.

        Shranjevanje samo sporočila (brez read_points_numpy na vsak frame) je
        poceni - reshape/branje naredimo šele, ko dejansko potrebujemo en
        piksel.
        """
        self._latest_pointcloud = data

    def pixel_to_map_coords(self, cx: int, cy: int, cloud: PointCloud2) -> tuple[float, float, float] | None:
        """
        Pretvori pixel koordinato (cx, cy) iz kamere v map-frame (x, y, z)
        točko, z uporabo organiziranega pointclouda (height x width, ujema se
        z resolucijo kamere) in TF transformacije base_link -> map.

        Vrne None, če:
        - pixel je zunaj meja pointclouda,
        - točka na tem pixlu je NaN/neveljavna (npr. brez veljavne globine),
        - TF transformacija (map <- base_link) ni na voljo za ta timestamp.
        """
        height = cloud.height
        width = cloud.width

        if not (0 <= cy < height and 0 <= cx < width):
            self.get_logger().warning(
                f'Pixel ({cx}, {cy}) is out of pointcloud bounds ({width}x{height})'
            )
            return None

        # Lookup TF enkrat - base_link -> map, na timestamp pointclouda
        # (usklajeno z originalno funkcijo, ki je uporabljala data.header.stamp)
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', cloud.header.stamp
            )
        except TransformException as exc:
            self.get_logger().warning(f'TF map<-base_link not available yet: {exc}')
            return None

        # Organiziran pointcloud -> 3-kanalni numpy array (x, y, z) v base_link frame
        points = pc2.read_points_numpy(cloud, field_names=('x', 'y', 'z'))
        points = points.reshape((height, width, 3))

        d = points[cy, cx, :]

        if not np.all(np.isfinite(d)):
            self.get_logger().warning(
                f'Point at pixel ({cx}, {cy}) is invalid (NaN/inf) - no valid depth there.'
            )
            return None

        point_stamped = PointStamped()
        point_stamped.header.frame_id = 'base_link'
        point_stamped.header.stamp = cloud.header.stamp
        point_stamped.point.x = float(d[0])
        point_stamped.point.y = float(d[1])
        point_stamped.point.z = float(d[2])

        point_map = do_transform_point(point_stamped, transform)

        x = float(point_map.point.x)
        y = float(point_map.point.y)
        z = float(point_map.point.z)

        self.get_logger().debug(
            f'Pixel ({cx}, {cy}) -> map coords ({x:.3f}, {y:.3f}, {z:.3f})'
        )

        return (x, y, z)

    def camera_callback(self, msg: Image):
        """Handle incoming camera frames."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f'CV Bridge error: {exc}')
            return

        # pasivno zaznavanje - vedno, ne glede na task, dokler nismo aktivno
        # v sredini manevra (da ne shranjujemo koordinat med samim izvajanjem)
        if self.state in (CellState.IDLE,):
            self.process_frame(cv_image)

        # state machine korak - en korak na frame, nikoli blokirajoče
        self.step_state_machine(cv_image)

    def process_frame(self, image: np.ndarray):
        """Zazna rdeče/zelene linije in shrani map-frame koordinate, če še niso shranjene."""
        if self.red_cell_coord is None and self._pending_red_pixel is None:
            coord = self.detect_line_of_color(image, 'red')
            if coord is not None:
                self._pending_red_pixel = coord
                self.get_logger().info(f'Red line pixel detected: {coord}, resolving to map coords...')

        if self.green_cell_coord is None and self._pending_green_pixel is None:
            coord = self.detect_line_of_color(image, 'green')
            if coord is not None:
                self._pending_green_pixel = coord
                self.get_logger().info(f'Green line pixel detected: {coord}, resolving to map coords...')

        # poskusi razresiti morebitne cakajoce piksle v map-frame koordinate
        # (potrebuje pointcloud, ki morda se ni prispel ob detekciji)
        self._try_resolve_pending_coords()

    def _try_resolve_pending_coords(self):
        """Poskusi pretvoriti zacasno shranjene pixel centroide (iz
        detect_line_of_color) v map-frame koordinate s pomocjo zadnjega
        prejetega pointclouda. Ce pointcloud ali TF se nista na voljo, pusti
        piksel v stanju 'pending' in poskusi spet ob naslednjem klicu."""
        if self._latest_pointcloud is None:
            return

        if self._pending_red_pixel is not None:
            cx, cy = self._pending_red_pixel
            map_coord = self.pixel_to_map_coords(cx, cy, self._latest_pointcloud)
            if map_coord is not None:
                self.red_cell_coord = map_coord
                self._pending_red_pixel = None
                self.get_logger().info(f'Red cell map coordinate saved: {map_coord}')
                if self.task_active and self.state == CellState.IDLE:
                    self.begin_cell_approach()

        if self._pending_green_pixel is not None:
            cx, cy = self._pending_green_pixel
            map_coord = self.pixel_to_map_coords(cx, cy, self._latest_pointcloud)
            if map_coord is not None:
                self.green_cell_coord = map_coord
                self._pending_green_pixel = None
                self.get_logger().info(f'Green cell map coordinate saved: {map_coord}')
                if self.task_active and self.state == CellState.IDLE:
                    self.begin_cell_approach()

    def on_task_started(self, msg: Bool):
        """ROS callback, ko task manager signalizira začetek taska."""
        if not msg.data:
            return

        self.task_active = True
        self.get_logger().info('Task started signal received')

        if self._has_coordinates():
            self.begin_cell_approach()
        else:
            self.get_logger().info('Waiting for line coordinates before starting approach...')

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def begin_cell_approach(self):
        """Sproži celoten manever: premik do celice, poravnava, sledenje, itd."""
        if self.state != CellState.IDLE:
            self.get_logger().warning('Approach already in progress, ignoring duplicate trigger.')
            return
        self._frames_without_line = 0
        self.state = CellState.MOVING_TO_CELL
        self.get_logger().info('State -> MOVING_TO_CELL')

    def step_state_machine(self, image: np.ndarray):
        if self.state == CellState.IDLE or self.state == CellState.DONE:
            return

        color = self._active_color()
        if color is None:
            self.get_logger().warning('No active color set, cannot proceed with state machine.')
            return

        if self.state == CellState.MOVING_TO_CELL:
            self._do_moving_to_cell(image, color)
        elif self.state == CellState.ALIGNING:
            self._do_aligning(image, color)
        elif self.state == CellState.FOLLOWING_TO_FAR_END:
            self._do_following(image, color, next_state=CellState.TURNING_180)
        elif self.state == CellState.TURNING_180:
            self._do_turning_180()
        elif self.state == CellState.PREPARING_TILE_DETECTION:
            self._do_preparing_tile_detection()
        elif self.state == CellState.FOLLOWING_BACK:
            self._do_following(image, color, next_state=CellState.DONE)

    def _publish_cmd(self, linear_x: float, angular_z: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        self.cmd_vel_pub.publish(msg)

    def _stop(self):
        self._publish_cmd(0.0, 0.0)

    def _do_moving_to_cell(self, image: np.ndarray, color: str):
        """
        Premik naprej, dokler ne zaznamo aktivne barvne linije neposredno pred/pod
        robotom (v spodnjem ROI-ju). To nadomešča potrebo po metričnih
        koordinatah - dokler nimamo pointcloud_callback z dejansko 3D pozicijo
        celice, se približujemo vizualno.
        """
        result = self.detect_color_for_navigation(image, color)

        if result is None:
            self._publish_cmd(self.approach_linear_speed, 0.0)
            return

        # Linija je vidna v spodnjem ROI-ju -> smo "na" celici.
        self._stop()
        self.get_logger().info('Reached cell, line visible underneath. State -> ALIGNING')
        self.state = CellState.ALIGNING

    def _do_aligning(self, image: np.ndarray, color: str):
        """
        Zavrti se v desno (negativna angular.z, robotska konvencija desno = -z
        po common REP-103; prilagodi znak, če je tvoj robot obraten), dokler
        kontura linije ni dovolj navpična v sliki (kar pomeni, da je robot
        obrnjen vzdolž linije, proti njenemu 'desnemu' koncu).
        """
        result = self.detect_color_for_navigation(image, color)

        if result is None:
            # linijo smo med obratom izgubili iz vidnega polja - rahlo se vrti
            # naprej v isto smer, lahko spet pride v kader
            self._publish_cmd(0.0, -self.align_angular_speed)
            self._frames_without_line += 1
            if self._frames_without_line > self.line_lost_threshold * 3:
                self.get_logger().warning(
                    'Lost line for too long while aligning, stopping as a precaution.'
                )
                self._stop()
                self.state = CellState.IDLE
            return

        self._frames_without_line = 0
        angle = result['angle_deg']

        if abs(angle) <= self.alignment_tolerance_deg:
            self._stop()
            self.get_logger().info(
                f'Aligned with line (angle={angle:.1f} deg). State -> FOLLOWING_TO_FAR_END'
            )
            self._frames_without_line = 0
            self.state = CellState.FOLLOWING_TO_FAR_END
            return

        # Obračaj se v desno, dokler kot ni znotraj tolerance.
        # (Predznak self.align_angular_speed je negativen za "desno" po REP-103;
        # prilagodi, če je tvoja konvencija drugačna.)
        self._publish_cmd(0.0, -self.align_angular_speed)

    def _do_following(self, image: np.ndarray, color: str, next_state: CellState):
        """
        Najpreprostejši delujoč line-follower: P-regulator na horizontalni
        offset centra linije od sredine slike. Ko linije ne vidimo
        `line_lost_threshold` zaporednih framov, štejemo, da smo prišli do
        konca in preidemo v naslednjo fazo.
        """
        result = self.detect_color_for_navigation(image, color)

        if result is None:
            self._frames_without_line += 1
            self._stop()
            if self._frames_without_line >= self.line_lost_threshold:
                self.get_logger().info(
                    f'Line lost for {self._frames_without_line} frames - end of line reached. '
                    f'State -> {next_state.name}'
                )
                self._frames_without_line = 0
                self.state = next_state
            return

        self._frames_without_line = 0
        offset_x = result['offset_x']

        # P-regulator: angular.z proporcionalen offsetu, s saturacijo.
        angular_z = -self.follow_angular_gain * offset_x
        angular_z = max(-self.follow_max_angular_speed,
                         min(self.follow_max_angular_speed, angular_z))

        self._publish_cmd(self.follow_linear_speed, angular_z)

    def _do_turning_180(self):
        """
        Časovno voden obrat za ~180 stopinj. Enostavna rešitev brez odometrije:
        vrti se s fiksno kotno hitrostjo za izračunan čas. Če imaš na voljo
        odometrijo/IMU, je bolje obrat voditi po dejanskem kotu - to je
        najpreprostejša verzija, ki dela "dovolj dobro" za zdaj.
        """
        if self._turn_start_time is None:
            self._turn_start_time = self.get_clock().now()
            self._turn_target_duration = math.pi / self.turn_180_angular_speed
            self.get_logger().info(
                f'Starting 180 deg turn, estimated duration {self._turn_target_duration:.2f}s'
            )

        elapsed = (self.get_clock().now() - self._turn_start_time).nanoseconds / 1e9

        if elapsed >= self._turn_target_duration:
            self._stop()
            self._turn_start_time = None
            self._turn_target_duration = None
            self.get_logger().info('180 deg turn complete. State -> PREPARING_TILE_DETECTION')
            self.state = CellState.PREPARING_TILE_DETECTION
            return

        self._publish_cmd(0.0, self.turn_180_angular_speed)

    def _do_preparing_tile_detection(self):
        """Dvigne kamero in pošlje start signal, nato preide na pot nazaj."""
        self.raise_top_camera()
        self.start_tile_detection()
        self._frames_without_line = 0
        self.get_logger().info('Tile detection prepared. State -> FOLLOWING_BACK')
        self.state = CellState.FOLLOWING_BACK

    # ------------------------------------------------------------------
    # Obstoječe pomožne metode
    # ------------------------------------------------------------------

    def raise_top_camera(self):
        """Dvigne zgornjo kamero v položaj za tile detection."""
        command = 'look_at_belt_right'
        self.arm_mover_pub.publish(String(data=command))
        self.get_logger().info('Sent arm mover command: look_at_belt_right')

    def start_tile_detection(self):
        """Objavi start signal za tile detection."""
        self.search_active = True
        self.tile_start_pub.publish(Bool(data=True))
        self.get_logger().info('Tile detection start signal sent')

    def stop_tile_detection(self):
        """Objavi stop signal za tile detection in ustavi node."""
        self.search_active = False
        self.task_active = False
        self._stop()
        self.tile_stop_pub.publish(Bool(data=True))
        self.get_logger().info('Tile detection stop signal sent')
        self.state = CellState.DONE

        # ZA DODAT, DA RESETIRA POZICIJO KAMERE

        self.get_logger().info('Shutting down node...')
        self.destroy_node()
        rclpy.shutdown()

    def reset_state(self):
        """Ponastavi interno stanje med taski."""
        self.red_cell_coord = None
        self.green_cell_coord = None
        self._pending_red_pixel = None
        self._pending_green_pixel = None
        self.task_active = False
        self.search_active = False
        self._frames_without_line = 0
        self.state = CellState.IDLE
        self._turn_start_time = None
        self._turn_target_duration = None
        self.get_logger().info('Internal state reset')


def main():
    rclpy.init()
    node = RedGreenCellDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()