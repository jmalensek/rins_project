#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')

        # Parameters
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("obstacle_distance_threshold", 0.5)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.threshold = self.get_parameter("obstacle_distance_threshold").value

        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_callback,
            scan_qos
        )

        self.pub_detected = self.create_publisher(Bool, "/obstacle/detected", 10)
        self.pub_ahead = self.create_publisher(Bool, "/obstacle/ahead", 10)

        self.get_logger().info(f"Obstacle detector started on {self.scan_topic}")

    def _scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = ranges[np.isfinite(ranges)]

        if len(ranges) == 0:
            return

        # Take center window (front of robot), 1/4 of the scan range I guess
        mid = len(msg.ranges) // 4
        window_size = 20  # adjust (10–50 typical)

        start = max(0, mid - window_size)
        end = min(len(msg.ranges), mid + window_size)

        front_ranges = np.array(msg.ranges[start:end], dtype=np.float32)
        front_ranges = front_ranges[np.isfinite(front_ranges)]

        if len(front_ranges) == 0:
            return

        min_dist = np.min(front_ranges)

        obstacle_ahead = min_dist < self.threshold
        obstacle_detected = np.any(ranges < self.threshold)

        # Publish
        msg_ahead = Bool()
        msg_ahead.data = bool(obstacle_ahead)
        self.pub_ahead.publish(msg_ahead)

        msg_detected = Bool()
        msg_detected.data = bool(obstacle_detected)
        self.pub_detected.publish(msg_detected)

        # Optional logging (avoid spam)
        if obstacle_ahead:
            self.get_logger().warn(f"Obstacle ahead; distance={min_dist:.2f} m")


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
