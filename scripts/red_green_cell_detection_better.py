#!/usr/bin/env python3

import math
import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from robot_commander import RobotCommander 

from robot_explorer import RobotExplorer

class RedGreenCellDetection(Node):
    def __init__(self):
        super().__init__('red_green_cell_detection')

        self.callback_group = ReentrantCallbackGroup()

        self.points = {
            'cell 1': {
                'start': (-1.57, -4.8),
                'end': (0.59, -4.8)
                },
            'cell 2': {
                'start': (-4.76, 0.343),
                'end': (-4.76, -2.749)
                }
            }
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_topic', '/oakd/rgb/preview/image_raw'),
                ('task_start_topic', '/task_manager/task_started'),
                ('tile_detection_start_topic', '/tile_detection/start'),
                ('tile_detection_stop_topic', '/tile_detection/stop'),
            ],
        )

        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.task_start_topic = self.get_parameter('task_start_topic').get_parameter_value().string_value
        self.tile_detection_start_topic = self.get_parameter('tile_detection_start_topic').get_parameter_value().string_value
        self.tile_detection_stop_topic = self.get_parameter('tile_detection_stop_topic').get_parameter_value().string_value

        self.explo = RobotExplorer()

        self.task_active = False
        self.color_of_the_cell = None
        self.latest_frame = None
        self.bridge = CvBridge()

        self.camera_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.camera_callback,
            qos_profile_sensor_data,
            callback_group=self.callback_group
        )

        self.task_start_sub = self.create_subscription(
            Bool,
            self.task_start_topic,
            self.on_task_started,
            10,
            callback_group=self.callback_group
        )

        self.color_cell_sub = self.create_subscription(
            String,
            '/task_manager/color_of_the_cell',
            self.color_of_the_cell_callback,
            10,
            callback_group=self.callback_group
        )

        self.tile_start_pub = self.create_publisher(Bool, self.tile_detection_start_topic, 10)
        self.tile_stop_pub = self.create_publisher(Bool, self.tile_detection_stop_topic, 10)
        self.arm_mover_pub = self.create_publisher(String, '/arm_command', 10)

        self.get_logger().info('RedGreenCellDetection node initialized and ready to receive tasks.')

    def camera_callback(self, msg: Image):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f'CV Bridge error: {exc}')

    def color_of_the_cell_callback(self, msg: String):
        self.color_of_the_cell = msg.data.lower().strip()
        self.get_logger().info(f"Color of the cell: {self.color_of_the_cell}")

    def on_task_started(self, msg: Bool):
        if not msg.data:
            return
        self.task_active = True
        self.get_logger().info('Start signal received.')
        self.start()

    def check_floor_color(self, expected_color: str) -> bool:
        if self.latest_frame is None:
            self.get_logger().warning("No camera frame received yet.")
            return False

        h, w = self.latest_frame.shape[:2]
        roi = self.latest_frame[h-100:h, :]

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        A = lab[:, :, 1] - 128.0
        B = lab[:, :, 2] - 128.0
        chroma = np.sqrt(A**2 + B**2)
        hue = np.degrees(np.arctan2(B, A)) % 360

        if expected_color == 'red':
            mask = (chroma >= 15) & ((hue < 42) | (hue >= 330))
        elif expected_color == 'green':
            mask = (chroma >= 15) & (hue >= 105) & (hue < 140)
        else:
            return False

        pixel_count = np.count_nonzero(mask)
        self.get_logger().info(f"Detected pixels of color '{expected_color}': {pixel_count}")
        
        return pixel_count > 500
    
    def raise_top_camera(self):
        command = 'look_at_belt_right'
        self.arm_mover_pub.publish(String(data=command))
        self.get_logger().info('Poslan ukaz za roko: look_at_belt_right')
        time.sleep(5)

    def lower_top_camera(self):
        command = 'look_for_qr'
        self.arm_mover_pub.publish(String(data=command))
        self.get_logger().info('Poslan ukaz za roko: look_for_qr')
        time.sleep(5)

    def start_tile_detection(self):
        self.tile_start_pub.publish(Bool(data=True))
        self.get_logger().info('Tile detection START signal sent.')

    def stop_tile_detection(self):
        self.tile_stop_pub.publish(Bool(data=True))
        self.get_logger().info('Tile detection STOP signal sent.')
        try:
            self.destroy_node()
        except Exception as exc:
            self.get_logger().error(f'Error destroying node: {exc}')
        rclpy.shutdown()

    def start(self):
        if self.color_of_the_cell not in ('red', 'green'):
            self.get_logger().error(f"Invalid cell color: {self.color_of_the_cell}. Expected 'red' or 'green'.")
            return
        
        self.get_logger().info(f"Starting tile detection for {self.color_of_the_cell} cells.")
        self.start_tile_detection()
        
        for cell_key in self.points:
            (start_x, start_y) = self.points[cell_key]['start']
            (end_x, end_y) = self.points[cell_key]['end']

            if not self.explo.go_to_pose(start_x, start_y, yaw=0.0):
                self.get_logger().error(f"Failed to resume goal to ({start_x:.2f}, {start_y:.2f})")
                break

            while rclpy.ok():
                rclpy.spin_once(self.explo, timeout_sec=0.0)

                if self.explo.result_future.done():
                    self.get_logger().info(f"Successfully reached ({start_x:.2f}, {start_y:.2f})")
                    break

            ayaw = self.explo.compute_absolute_yaw((start_x, start_y), (end_x, end_y))
            distance = self.explo.compute_distance((start_x, start_y), (end_x, end_y))

            # Turn and move towards the end point
            self.explo.turn_odom(ayaw)

            # Check floor color before moving
            if self.check_floor_color(self.color_of_the_cell):
                self.get_logger().info(f"Floor color '{self.color_of_the_cell}' confirmed. Raising top camera and moving straight towards the end point.")
                self.raise_top_camera()

                time.sleep(1)  # Brief pause before moving

                self.explo.move_straight_odom(distance)

                self.get_logger().info(f"Reached end point of {cell_key}. Stopping tile detection and lowering camera.")
                self.lower_top_camera()

                break
        
        self.get_logger().info("Finished processing all cells. Stopping tile detection.")
        self.stop_tile_detection()

def main():
    rclpy.init()
    node = RedGreenCellDetection()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.add_node(node.explo)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            try:
                node.destroy_node()
            except Exception:
                pass
            rclpy.shutdown()

if __name__ == '__main__':
    main()
