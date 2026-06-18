#!/usr/bin/env python3

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32


class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')

        # Parameters
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("obstacle_detected_distance_threshold", 0.25)
        self.declare_parameter("obstacle_ahead_distance_threshold", 0.45)
        self.declare_parameter("window_size", 50)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.detected_threshold = self.get_parameter("obstacle_detected_distance_threshold").value
        self.ahead_threshold = self.get_parameter("obstacle_ahead_distance_threshold").value
        self.window_size = self.get_parameter("window_size").value

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
        self.pub_nearest_direction = self.create_publisher(Float32, "/obstacle/nearest_direction_rad", 10)

        self.get_logger().info(f"Obstacle detector started. Subscribing to {self.scan_topic}")

    def _scan_callback(self, msg: LaserScan):
        # 1. Keep the raw ranges array intact so indices map perfectly to angles
        raw_ranges = np.array(msg.ranges, dtype=np.float32)

        # 2. Create a masked version where inf/nan are replaced with infinity 
        # This keeps the array the exact same size, but ignores bad readings for np.argmin
        clean_ranges = np.where(np.isfinite(raw_ranges), raw_ranges, np.inf)

        # Quick safety check: if everything is inf/nan, skip
        if np.all(clean_ranges == np.inf):
            return

        # 3. Handle the front window safely using the original indices
        mid = len(msg.ranges) // 4
        start = max(0, mid - self.window_size)
        end = min(len(msg.ranges), mid + self.window_size)
        
        front_ranges = clean_ranges[start:end]
        min_dist_front = np.min(front_ranges)

        # 4. Get the minimum distance and its exact ORIGINAL index
        min_all_index = np.argmin(clean_ranges)
        min_dist_all = clean_ranges[min_all_index]

        # 5. Compute the angle of the nearest obstacle relative to the front (mid)
        # Difference in indices * angle step size
        nearest_angle = (min_all_index - mid) * msg.angle_increment

        # --- The rest of your publishing code ---
        obstacle_ahead = min_dist_front < self.ahead_threshold
        obstacle_detected = min_dist_all < self.detected_threshold

        msg_ahead = Bool()
        msg_ahead.data = bool(obstacle_ahead)
        self.pub_ahead.publish(msg_ahead)

        msg_detected = Bool()
        msg_detected.data = bool(obstacle_detected)
        self.pub_detected.publish(msg_detected)

        # Publish the direction of the nearest obstacle
        msg_direction = Float32()
        msg_direction.data = float(nearest_angle)
        self.pub_nearest_direction.publish(msg_direction)

        # Optional logging (avoid spam)
        if obstacle_ahead:
            #self.get_logger().warn(f"Obstacle ahead; distance={min_dist_front:.2f} m")
            pass

        if obstacle_detected:
            #self.get_logger().warn(f"Obstacle detected; min distance={min_dist_all:.2f} m at angle {math.degrees(nearest_angle):.1f} deg")
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
