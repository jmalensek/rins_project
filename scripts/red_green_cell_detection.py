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



NOVO:
ko pride robotek do koordinat, se samo preveri katera barva je, pa se pošilja nav goal za 0.5 v x/y smer (v minus). 
Vsakič znova se potem preveri, ali je ta barva še v vidnem polju (roi, leva polovica). Potem ko ni več vidna, se robotek 
zarotira za 180 stopinj, in potem nadaljuje z nav goal za 0.5 v x/y smer (v plus). vsakič preveri ali je barva še vidna v roi 
na desni strani. ko ni več, je konec
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

from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, TwistStamped

from action_msgs.msg import GoalStatus

from tf_transformations import euler_from_quaternion
from nav_msgs.msg import Odometry


class CellState(Enum):
    IDLE = auto()
    WAITING_FOR_COLOR = auto()
    MOVING_TO_CELL = auto()
    ROTATING_90_RIGHT = auto()
    FIRST_PASS_STEPPING = auto()
    TURNING_180 = auto()
    RETURN_PASS_STEPPING = auto()
    PREPARING_TILE_DETECTION = auto()
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
                ('last_goal', )
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

        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        self.nav_to_pose_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose'
        )

        self.navigation_goal_active = False

        # po pridobitvi pointcloud-a se to spremeni v map-frame (x, y, z) tocko,
        # ne vec pixel (cx, cy) - glej pixel_to_map_coords / pointcloud_callback
        self.red_cell_coord: tuple[float, float, float] | None = None
        self.green_cell_coord: tuple[float, float, float] | None = None
        self.task_active = False
        self.search_active = False

        # state machine
        self.state = CellState.IDLE
        
        # step-and-check state tracking
        self._current_goal_x: float | None = None
        self._current_goal_y: float | None = None
        self._step_size = 0.5  # meters
        self._frames_without_line = 0
        self._line_lost_threshold = 10

        # zadnji prejeti pointcloud (organiziran, height x width kot kamera slika)
        # in zacasno shranjeni pixel centroidi, ki cakajo na uskladitev s
        # pointcloud-om/TF (ce pointcloud/TF se ni na voljo ob detekciji)
        self._latest_pointcloud: PointCloud2 | None = None
        self._pending_red_pixel: tuple[int, int] | None = None
        self._pending_green_pixel: tuple[int, int] | None = None

        self.current_yaw = None

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

        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.tile_start_pub = self.create_publisher(Bool, self.tile_detection_start_topic, 10)
        self.tile_stop_pub = self.create_publisher(Bool, self.tile_detection_stop_topic, 10)
        self.arm_mover_pub = self.create_publisher(String, '/arm_command', 10)
        self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)

        self.get_logger().info('red_green_cell_detection node initialized')

    def get_current_yaw(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'odom',          # or 'map'
                'base_link',
                rclpy.time.Time()
            )

            q = trans.transform.rotation

            import math
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            return yaw

        except Exception as e:
            return 0.0
    def _get_yaw_from_odom(self, odom_msg):
        q = odom_msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)
        return yaw
    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation

        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)

        self.current_yaw = self.get_current_yaw()

    def color_of_the_cell_callback(self, msg: String):
        self.color_of_the_cell = msg.data
        self.get_logger().info(
            f"Color of the cell received: {self.color_of_the_cell}"
        )
        if self.task_active and self.state == CellState.WAITING_FOR_COLOR:
            if self._has_coordinates():
                self.get_logger().info('Color received and coordinates present; starting approach.')
                self.begin_cell_approach()
            else:
                self.get_logger().info('Color received; still waiting for line coordinates before starting.')

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

        #cv2.imshow(f'line', roi)
        #cv2.waitKey(1)

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
            mask = (chroma >= 15) & (hue >= 105) & (hue < 140)
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

        # Convert contour bounding box from ROI coordinates to full-image coordinates
        y_full = y + (h - floor_band_height)


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


        # Build color mask for the entire image
        lab_full = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        A_full = lab_full[:, :, 1] - 128.0
        B_full = lab_full[:, :, 2] - 128.0

        chroma_full = np.sqrt(A_full**2 + B_full**2)
        hue_full = np.degrees(np.arctan2(B_full, A_full)) % 360

        if color == 'red':
            full_mask = (chroma_full >= 15) & ((hue_full < 42) | (hue_full >= 330))
        else:  # green
            full_mask = (chroma_full >= 15) & (hue_full >= 105) & (hue_full < 140)

        full_mask = full_mask.astype(np.uint8) * 255

        upper_region = full_mask[:h-floor_band_height, :]

        contours_upper, _ = cv2.findContours(
            upper_region,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours_upper:
            if cv2.contourArea(cnt) > self.min_contour_area:
                self.get_logger().info(
                    f"Rejected {color} line because significant {color} region "
                    f"exists above ROI"
                )
                return None

        self.get_logger().info(
            f"Detected {color} line at ({cx}, {cy}) "
            f"(hue={hue[cy_roi, cx]:.1f}, chroma={chroma[cy_roi, cx]:.1f}"
            f"(area={area:.0f}, aspect_ratio={aspect_ratio:.2f})"
        )

        return (cx, cy)

    def detect_color_for_navigation(self, image: np.ndarray, color: str):
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

        mask = mask.astype(np.uint8)

        # če premalo pikslov → nič
        if np.count_nonzero(mask) <= 3:
            self.get_logger().info(f"Detected {color} pixels, but too few to consider it a line.")
            return None

        # centroid iz mask (isti princip kot moments)
        M = cv2.moments(mask)
        if M["m00"] == 0:
            self.get_logger().info(f"Detected {color} pixels, but no valid centroid found.")
            return None

        cx_roi = int(M["m10"] / M["m00"])
        cy_roi = int(M["m01"] / M["m00"])

        # pretvorba v globalne koordinate slike
        cx = cx_roi
        cy = cy_roi + (h - floor_band_height)

        self.get_logger().info(f"Detected {color} line for navigation at ({cx}, {cy})")

        return {
            "center": (cx, cy)
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
    
    def get_offset_goal(self, target_x, target_y, offset=0.6):

        try:
            tf = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time()
            )

            robot_x = tf.transform.translation.x
            robot_y = tf.transform.translation.y

        except TransformException as exc:
            self.get_logger().warning(
                f'Failed getting robot pose: {exc}'
            )
            return target_x, target_y

        dx = target_x - robot_x
        dy = target_y - robot_y

        dist = math.hypot(dx, dy)

        if dist < 0.05:
            return target_x, target_y

        goal_x = target_x - offset * dx / dist
        goal_y = target_y - offset * dy / dist

        return goal_x, goal_y
    
    def get_target_coordinate(self):
        if self.color_of_the_cell == 'red':
            return self.red_cell_coord

        if self.color_of_the_cell == 'green':
            return self.green_cell_coord

        return None
    
    def send_navigation_goal(self):

        target = self.get_target_coordinate()

        if target is None:
            self.get_logger().warning(
                'No target coordinate available'
            )
            return

        target_x, target_y, _ = target

        if not self.nav_to_pose_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                'NavigateToPose action server not available'
            )
            return

        
        goal_msg = NavigateToPose.Goal()

        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_x, goal_y = self.get_offset_goal(
            target_x,
            target_y,
            offset=0.7
        )

        goal_msg.pose.pose.position.x = target_x
        goal_msg.pose.pose.position.y = target_y
        goal_msg.pose.pose.position.z = 0.0

        #
        # Orientation:
        #

        yaw = math.atan2(
            target_y - goal_y,
            target_x - goal_x
        )
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(
            f'Sending Nav2 goal ({target_x:.2f}, {target_y:.2f})'
        )

        self.navigation_goal_active = True

        send_goal_future = self.nav_to_pose_client.send_goal_async(
            goal_msg
        )

        send_goal_future.add_done_callback(
            self.nav_goal_response_callback
        )

    def nav_goal_response_callback(self, future):

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error(
                'Navigation goal rejected'
            )

            self.navigation_goal_active = False
            self.state = CellState.IDLE
            return

        self.get_logger().info(
            'Navigation goal accepted'
        )

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            self.nav_result_callback
        )

    def nav_result_callback(self, future):

        self.navigation_goal_active = False

        result = future.result()
        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation step succeeded')
            # Transition from MOVING_TO_CELL -> ROTATING_90_RIGHT
            if self.state == CellState.MOVING_TO_CELL:
                self.get_logger().info('Reached initial cell area. State -> ROTATING_90_RIGHT')
                self.state = CellState.ROTATING_90_RIGHT
                # Initialize goal coordinates from target cell
                if self.red_cell_coord and self.color_of_the_cell == 'red':
                    self._current_goal_x = self.red_cell_coord[0]
                    self._current_goal_y = self.red_cell_coord[1]
                elif self.green_cell_coord and self.color_of_the_cell == 'green':
                    self._current_goal_x = self.green_cell_coord[0]
                    self._current_goal_y = self.green_cell_coord[1]
        else:
            self.get_logger().error(f'Navigation failed with status {status}')
            self.state = CellState.IDLE
    def camera_callback(self, msg: Image):
        """Handle incoming camera frames."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            h, w = cv_image.shape[:2]

            # Only look at the bottom part of the image
            floor_band_height = 80
            roi = cv_image[h-floor_band_height:h, :]
            cv2.imshow(f'line', roi)
            cv2.waitKey(1)

        except CvBridgeError as exc:
            self.get_logger().error(f'CV Bridge error: {exc}')
            return

        # pasivno zaznavanje - vedno, ne glede na task, dokler nismo aktivno
        # v sredini manevra (da ne shranjujemo koordinat med samim izvajanjem)
        if self.state in (CellState.IDLE, CellState.WAITING_FOR_COLOR):
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
                #if self.task_active and self.state == CellState.IDLE:
                    #self.begin_cell_approach()

        if self._pending_green_pixel is not None:
            cx, cy = self._pending_green_pixel
            map_coord = self.pixel_to_map_coords(cx, cy, self._latest_pointcloud)
            if map_coord is not None:
                self.green_cell_coord = map_coord
                self._pending_green_pixel = None
                self.get_logger().info(f'Green cell map coordinate saved: {map_coord}')
                #if self.task_active and self.state == CellState.IDLE:
                    #self.begin_cell_approach()

    def on_task_started(self, msg: Bool):
        """ROS callback, ko task manager signalizira začetek taska."""
        if not msg.data:
            return

        self.task_active = True
        self.get_logger().info('Task started signal received')

        if self.color_of_the_cell is None:
            self.state = CellState.WAITING_FOR_COLOR
            self.get_logger().info('Waiting for color signal before starting approach...')
            return

        if not self._has_coordinates():
            self.state = CellState.WAITING_FOR_COLOR
            self.get_logger().info('Color known, waiting for line coordinates before starting approach...')
            return

        self.begin_cell_approach()


    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def begin_cell_approach(self):
        """Sproži celoten manever: premik do celice, nato step-and-check."""
        if self.state != CellState.IDLE:
            self.get_logger().warning('Approach already in progress, ignoring duplicate trigger.')
            return
        self._frames_without_line = 0
        self.state = CellState.MOVING_TO_CELL
        self.send_navigation_goal()
        self.get_logger().info('State -> MOVING_TO_CELL')
        self._current_goal_x = None
        self._current_goal_y = None

    def step_state_machine(self, image: np.ndarray):
        if self.state == CellState.IDLE or self.state == CellState.DONE:
            return

        color = self._active_color()
        if color is None:
            self.get_logger().warning('No active color set, cannot proceed with state machine.')
            return

        if self.state == CellState.MOVING_TO_CELL:
            # Čakamo, da Nav2 doseže ciljno lokacijo, nato preide v rotacijo
            return
        elif self.state == CellState.ROTATING_90_RIGHT:
            self._do_rotate_90_right(image, color)
        elif self.state == CellState.FIRST_PASS_STEPPING:
            self._do_first_pass_step(image, color)
        elif self.state == CellState.TURNING_180:
            self._do_turn_180(image, color)
        elif self.state == CellState.RETURN_PASS_STEPPING:
            self._do_return_pass_step(image, color)
        elif self.state == CellState.PREPARING_TILE_DETECTION:
            self._do_preparing_tile_detection()

    def _color_direction(self, color: str) -> tuple[str, str]:
        """Vrne smer premika za dano barvo. Red = y, Green = x. Vrne ('x'/'y', 'pos'/'neg')."""
        if color == 'red':
            return ('x', 'neg')  # Horizontalna črta -> y gibanje
        else:  # green
            return ('y', 'neg')  # Vertikalna črta -> x gibanje

    def _line_visible_in_roi_half(self, image: np.ndarray, color: str, side: str) -> bool:
        """Preveri, ali je črta vidna v izbrani polovici ROI. side='left' ali 'right'."""
        result = self.detect_color_for_navigation(image, color)
        if result is None:
            return False
        
        cx = result['center'][0]
        w = image.shape[1]
        
        if side == 'left':
            return cx < w / 2.0
        else:  # 'right'
            return cx >= w / 2.0

    def _send_step_goal(self, x: float, y: float, yaw: float = 0.0):
        """Pošlje Nav2 cilj na podanih svetovalnih koordinatah z dano yaw orientacijo."""
        if not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('NavigateToPose action server not available')
            return
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        
        # Set orientation from yaw
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        self.get_logger().info(f'Sending Nav2 step goal: ({x:.2f}, {y:.2f}), yaw={math.degrees(yaw):.1f}°')
        self.navigation_goal_active = True
        
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.nav_goal_response_callback)

    def _do_rotate_90_right(self, image: np.ndarray, color: str):
        """In-place 90° right rotation using cmd_vel (stateful)."""
        if not hasattr(self, "_rotate_90_active"):
            self.get_logger().info("Starting 90° right turn via cmd_vel")
            target = self.current_yaw - math.pi / 2.0  # 90° to the right (negative)
            self._rotate_90_target = math.atan2(
                math.sin(target),
                math.cos(target)
            )
            self._rotate_90_active = True

        # ---- CONTINUOUS CONTROL ----
        err = self._rotate_90_target - self.current_yaw
        err = math.atan2(math.sin(err), math.cos(err))

        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = "base_link"

        if abs(err) > 0.05:
            twist.twist.angular.z = max(-0.8, min(0.8, 2.0 * err))
            self.cmd_vel_pub.publish(twist)
            return

        # ---- STOP ----
        twist.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

        self._rotate_90_active = False
        self._frames_without_line = 0
        self.get_logger().info('90° right rotation complete. State -> FIRST_PASS_STEPPING')
        self.state = CellState.FIRST_PASS_STEPPING

    def _do_first_pass_step(self, image: np.ndarray, color: str):
        """Prvi prehod: korak v negativni smeri, preverjanje leve polovice ROI."""
        if not self.navigation_goal_active:
            # Prejšnji korak je zaključen, pošlji naslednji
            if self._line_visible_in_roi_half(image, color, 'left'):
                # črta je še vidna na levi -> pošlji naslednji korak
                axis, _ = self._color_direction(color)
                if axis == 'y':
                    next_y = (self._current_goal_y or (self.green_cell_coord[1] if self.green_cell_coord else 0)) - self._step_size
                    next_x = self._current_goal_x or (self.green_cell_coord[0] if self.green_cell_coord else 0)
                else:  # 'x'
                    next_x = (self._current_goal_x or (self.red_cell_coord[0] if self.red_cell_coord else 0)) - self._step_size
                    next_y = self._current_goal_y or (self.red_cell_coord[1] if self.red_cell_coord else 0)
                
                self._current_goal_x = next_x
                self._current_goal_y = next_y
                self._send_step_goal(next_x, next_y)
            else:
                # črta ni vidna na levi -> konec prvega prehoda, preidi na zavoj
                self.get_logger().info('Line lost on left side. State -> TURNING_180')
                self.state = CellState.TURNING_180
                self._frames_without_line = 0

    def _do_turn_180(self, image: np.ndarray, color: str):
        """In-place 180° rotation using cmd_vel (stateful)."""

        if not hasattr(self, "_turn_180_active"):
            self.get_logger().info("Starting 180° turn via cmd_vel")

            target = self.current_yaw + math.pi
            self._turn_180_target = math.atan2(
                math.sin(target),
                math.cos(target)
            )

            self._turn_180_active = True
            self.state = CellState.RETURN_PASS_STEPPING

        # ---- CONTINUOUS CONTROL ----
        err = self._turn_180_target - self.current_yaw
        err = math.atan2(math.sin(err), math.cos(err))

        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = "base_link"

        if abs(err) > 0.05:
            twist.twist.angular.z = max(-0.8, min(0.8, 2.0 * err))
            self.cmd_vel_pub.publish(twist)
            return

        # ---- STOP ----
        twist.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

        self._turn_180_active = False
        self._frames_without_line = 0

    def _do_return_pass_step(self, image: np.ndarray, color: str):
        """Povratni prehod: korak v pozitivni smeri, preverjanje desne polovice ROI."""
        if not self.navigation_goal_active:
            # Prejšnji korak je zaključen, pošlji naslednji
            if self._line_visible_in_roi_half(image, color, 'right'):
                # črta je še vidna na desni -> pošlji naslednji korak
                axis, _ = self._color_direction(color)
                if axis == 'y':
                    next_y = self._current_goal_y + self._step_size
                    next_x = self._current_goal_x
                else:  # 'x'
                    next_x = self._current_goal_x + self._step_size
                    next_y = self._current_goal_y
                
                self._current_goal_x = next_x
                self._current_goal_y = next_y
                self._send_step_goal(next_x, next_y)
            else:
                # črta ni vidna na desni -> konec povratnega prehoda, pripravi tile detection
                self.get_logger().info('Line lost on right side. State -> PREPARING_TILE_DETECTION')
                self.state = CellState.PREPARING_TILE_DETECTION
                self._frames_without_line = 0

    def _do_preparing_tile_detection(self):
        """Dvigne kamero in pošlje start signal, nato se zaključi."""
        if not self.navigation_goal_active:
            self.raise_top_camera()
            self.start_tile_detection()
            self.get_logger().info('Tile detection started. State -> DONE')
            self.state = CellState.DONE

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
        self.tile_stop_pub.publish(Bool(data=True))
        self.get_logger().info('Tile detection stop signal sent')
        self.state = CellState.DONE

        command = 'look_for_qr'
        self.arm_mover_pub.publish(String(data=command))
        self.get_logger().info('Sent arm mover command: look_for_qr')

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
        self._rotate_90_active = False
        self._turn_180_active = False
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
