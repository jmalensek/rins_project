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

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np




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
            ],
        )

        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.task_start_topic = self.get_parameter('task_start_topic').get_parameter_value().string_value
        self.tile_detection_start_topic = self.get_parameter('tile_detection_start_topic').get_parameter_value().string_value
        self.tile_detection_stop_topic = self.get_parameter('tile_detection_stop_topic').get_parameter_value().string_value
        self.line_lost_threshold = self.get_parameter('line_lost_frames_threshold').get_parameter_value().integer_value
        self.min_contour_area = self.get_parameter('min_contour_area').get_parameter_value().integer_value

        self.red_cell_coord: tuple[int, int] | None = None
        self.green_cell_coord: tuple[int, int] | None = None
        self.task_active = False
        self.search_active = False

        # števec framov, v katerih aktivna linija ni bila vidna
        self._frames_without_line = 0

        self.bridge = CvBridge()

        self.camera_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.camera_callback,
            qos_profile_sensor_data,
        )

        self.task_start_sub = self.create_subscription(
            Bool,
            self.task_start_topic,
            self.on_task_started,
            10,
        )

        # tukaj bo treba še sub na topic od task managerja

        self.tile_start_pub = self.create_publisher(Bool, self.tile_detection_start_topic, 10)
        self.tile_stop_pub = self.create_publisher(Bool, self.tile_detection_stop_topic, 10)

        self.get_logger().info('red_green_cell_detection node initialized')


    def _has_coordinates(self) -> bool:
        return self.red_cell_coord is not None or self.green_cell_coord is not None

    def _active_line_visible(self, image: np.ndarray) -> bool:
        """Vrne True, če je vsaj ena od zaznanih linij še vidna v kadru."""
        if self.red_cell_coord is not None:
            coord = self.detect_line_of_color(image, 'red')
            if coord is not None:
                return True
        if self.green_cell_coord is not None:
            coord = self.detect_line_of_color(image, 'green')
            if coord is not None:
                return True
        return False


    def classify_lab(self, L, A, B):
        chroma = np.sqrt(A ** 2 + B ** 2)
        if chroma < 15:
            return "other"
        hue = np.degrees(np.arctan2(B, A)) % 360
        if hue < 42 or hue >= 330:
            return "red"
        elif 75 <= hue < 145:
            return "green"
        else:
            return "other"

    def detect_line_of_color(self, image: np.ndarray, color: str) -> tuple[int, int] | None:
        """
        Poišče linijo podane barve ('red' ali 'green') v sliki.
        Vrne center (cx, cy) največje konture ali None.
        """
        # Pretvori v LAB in razdeli na kanale
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # OpenCV LAB: A in B sta shifted [0..255] → centriraj na [-128..127]
        A -= 128.0
        B -= 128.0

        # Klasificiraj vsak pixel vektorizirano
        chroma = np.sqrt(A ** 2 + B ** 2)
        hue = np.degrees(np.arctan2(B, A)) % 360

        if color == 'red':
            mask = (chroma >= 15) & ((hue < 42) | (hue >= 330))
        elif color == 'green':
            mask = (chroma >= 15) & (hue >= 75) & (hue < 145)
        else:
            return None

        mask = mask.astype(np.uint8) * 255

        # Morfološko čiščenje šuma
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_contour_area:
            return None

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)


    def camera_callback(self, msg: Image):
        """Handle incoming camera frames."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f'CV Bridge error: {exc}')
            return

        # pasivno zaznavanje – vedno, ne glede na task
        self.process_frame(cv_image)

        # med aktivnim searchom preverimo, ali smo linijo izgubili
        if self.search_active:
            if self.should_finish_search(cv_image):
                self.stop_tile_detection()

    def process_frame(self, image: np.ndarray):
        """Zazna rdeče/zelene linije in shrani koordinate, če še niso shranjene."""
        if self.red_cell_coord is None:
            coord = self.detect_line_of_color(image, 'red')
            if coord is not None:
                self.red_cell_coord = coord
                self.get_logger().info(f'Red cell coordinate saved: {coord}')
                # če task čaka na koordinate, ga zdaj sproži
                if self.task_active and not self.search_active:
                    self.start_tile_detection()

        if self.green_cell_coord is None:
            coord = self.detect_line_of_color(image, 'green')
            if coord is not None:
                self.green_cell_coord = coord
                self.get_logger().info(f'Green cell coordinate saved: {coord}')
                if self.task_active and not self.search_active:
                    self.start_tile_detection()

    def on_task_started(self, msg: Bool):
        """ROS callback, ko task manager signalizira začetek taska."""
        if not msg.data:
            return

        self.task_active = True
        self.get_logger().info('Task started signal received')

        # pošljemo start samo, če že imamo koordinate
        if self._has_coordinates():
            self.start_tile_detection()
        else:
            self.get_logger().info('Waiting for line coordinates before starting tile detection...')


    def start_tile_detection(self):
        """Objavi start signal za tile detection."""
        self.search_active = True
        self._frames_without_line = 0
        self.tile_start_pub.publish(Bool(data=True))
        self.get_logger().info('Tile detection start signal sent')

    def stop_tile_detection(self):
        """Objavi stop signal za tile detection in ustavi node."""
        self.search_active = False
        self.task_active = False
        self.tile_stop_pub.publish(Bool(data=True))
        self.get_logger().info('Tile detection stop signal sent')

        # node se ustavi sam
        self.get_logger().info('Shutting down node...')
        self.destroy_node()
        rclpy.shutdown()


    def should_finish_search(self, image: np.ndarray) -> bool:
        """
        Vrne True, ko aktivna linija ni vidna za vsaj `line_lost_frames_threshold`
        zaporednih framov → robot je prišel čez konec linije.
        """
        if self._active_line_visible(image):
            self._frames_without_line = 0
            return False

        self._frames_without_line += 1
        self.get_logger().debug(
            f'Line not visible for {self._frames_without_line}/{self.line_lost_threshold} frames'
        )
        return self._frames_without_line >= self.line_lost_threshold


    def save_cell_coordinates(self):
        """Shrani / objavi koordinate, če je potrebno."""
        # TODO: implementiraj shranjevanje, če bo potrebno
        pass

    def reset_state(self):
        """Ponastavi interno stanje med taski."""
        self.red_cell_coord = None
        self.green_cell_coord = None
        self.task_active = False
        self.search_active = False
        self._frames_without_line = 0
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