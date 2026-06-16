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

from matplotlib import image

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
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


    def _has_coordinates(self) -> bool:
        if self.color_of_the_cell == "red":
            return self.red_cell_coord is not None
        if self.color_of_the_cell == "green":
            return self.green_cell_coord is not None
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
        elif 75 <= hue < 100:
            return "green"
        else:
            return "other"

    def detect_line_of_color(self, image: np.ndarray, color: str) -> tuple[int, int] | None:
        """
        Detect a red or green line on the floor.
        Returns the center (cx, cy) of the largest valid contour.
        """

        h, w = image.shape[:2]

        # Only look at the bottom part of the image
        floor_band_height = 80
        roi = image[h-floor_band_height:h, :]

        # Convert ROI to LAB
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

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
            self.prepare_for_tile_detection()
        else:
            self.get_logger().info('Waiting for line coordinates before starting tile detection...')


    def prepare_for_tile_detection(self):
        """Pripravi robota za začetek tile detectiona."""
        # Premakni se na koordinate in poišči desni konec linije.
        self.go_to_cell_coordinates()
        self.find_and_position_at_right_end()
        self.raise_top_camera()
        self.start_tile_detection()
        self.move_along_line_towards_left()

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

        # ZA DODAT, DA RESETIRA POZICIJO KAMERE

        # node se ustavi sam
        self.get_logger().info('Shutting down node...')
        self.destroy_node()
        rclpy.shutdown()

    def go_to_cell_coordinates(self):
        """Premakne robota na shranjene koordinate cella z osnovnimi vozilično-motornimi metodami."""
        if self.color_of_the_cell == 'red' and self.red_cell_coord is not None:
            target = self.red_cell_coord
        elif self.color_of_the_cell == 'green' and self.green_cell_coord is not None:
            target = self.green_cell_coord
        elif self.red_cell_coord is not None:
            target = self.red_cell_coord
        elif self.green_cell_coord is not None:
            target = self.green_cell_coord
        else:
            self.get_logger().warning('No valid cell coordinate available for movement.')
            return

        self.get_logger().info(f'Moving toward cell coordinate {target} using base motion.')
        self.turn(math.pi / 4, angular_speed=0.3)
        self.move_straight(distance=0.3, speed=0.1)

    def find_and_position_at_right_end(self):
        """Poišče desni konec linije in se obrnjen pomakne tja."""
        self.get_logger().info('Positioning robot at right end of the line using turn and straight movement.')
        self.turn(-math.pi / 4, angular_speed=0.3)
        self.move_straight(distance=0.2, speed=0.1)

    def raise_top_camera(self):
        """Dvigne zgornjo kamero v položaj za tile detection."""
        command = 'raise_top_camera'
        self.arm_mover_pub.publish(String(data=command))
        self.get_logger().info('Sent arm mover command: raise_top_camera')

    def move_along_line_towards_left(self):
        """Začne počasen premik vzdolž linije proti levemu koncu."""
        self.get_logger().info('Starting slow movement along line toward left.')
        while self.search_active and rclpy.ok():
            self.move_straight(distance=0.05, speed=0.05)
            if self._frames_without_line >= self.line_lost_threshold:
                break

    def move_straight(self, distance: float, speed: float = 0.2) -> None:
        if distance <= 0:
            self.get_logger().error('Distance must be positive.')
            return

        speed = abs(speed)
        duration = distance / speed

        twist_msg = TwistStamped()
        twist_msg.twist.linear.x = speed

        stop_msg = TwistStamped()

        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds < duration * 1e9:
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = 'base_link'
            self.cmd_vel_pub.publish(twist_msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.1)

        for _ in range(3):
            stop_msg.header.stamp = self.get_clock().now().to_msg()
            stop_msg.header.frame_id = 'base_link'
            self.cmd_vel_pub.publish(stop_msg)
            time.sleep(0.05)

    def turn(self, angle: float, angular_speed: float = 0.5) -> None:
        if angle is None:
            self.get_logger().warn('Turn angle is None; skipping turn command')
            return

        angular_speed = abs(angular_speed)
        duration = abs(angle) / angular_speed

        twist_msg = TwistStamped()
        twist_msg.twist.angular.z = angular_speed if angle > 0 else -angular_speed

        stop_msg = TwistStamped()

        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds < duration * 1e9:
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = 'base_link'
            self.cmd_vel_pub.publish(twist_msg)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.1)

        for _ in range(3):
            stop_msg.header.stamp = self.get_clock().now().to_msg()
            stop_msg.header.frame_id = 'base_link'
            self.cmd_vel_pub.publish(stop_msg)
            time.sleep(0.05)

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
