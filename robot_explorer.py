#! /usr/bin/env python3

import rclpy
import time
import math
import tf2_ros

from tf_transformations import euler_from_quaternion

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan

from nav_msgs.msg import OccupancyGrid

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

class RobotExplorer(Node):

    def __init__(self, node_name='robot_explorer', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        
        self.pose_frame_id = 'map'
        
        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None
        self.scan_data = None
        self.map_data = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)

        # Subscribers
        self.create_subscription(LaserScan, 'scan_filtered', self._scanCallback, qos_profile_sensor_data)
        self.create_subscription(OccupancyGrid, 'map', self._mapCallback, qos_profile_sensor_data)

        # Initialisation successful
        self.get_logger().info(f"Robot explorer has been initialized!")

    # BASIC MOTORIC METHODS
    def move_straight(self, distance: float, speed: float = 0.2) -> None:

        """
        Moves the robot in a straight line for a specified distance at a specified speed.
        Args:
            distance(float): meters to move forward
            speed(float): linear speed in m/s (default: 0.2)

        Returns:
            None
        """

        if distance <= 0:
            self.get_logger().error("Distance must be positive.")
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

    def turn(self, angle: float, angular_speed: float = 0.5):

        """
        Turns the robot by a specified angle at a specified angular speed.
        Args:
            angle(float): angle to turn in radians (positive for counterclockwise, negative for clockwise)
            angular_speed(float): angular speed in rad/s (default: 0.5)
        
        Returns:
            None
        """

        if angle is None:
            self.get_logger().warn("Turn angle is None; skipping turn command")
            return

        if angular_speed <= 0:
            self.get_logger().error("Angular speed must be positive")
            return
            
        angular_speed = abs(angular_speed)
        duration = abs(angle) / angular_speed

        twist_msg = TwistStamped()
        if angle > 0:
            twist_msg.twist.angular.z = angular_speed
        else:
            twist_msg.twist.angular.z = -angular_speed
            
        stop_msg = TwistStamped()

        start_time = self.get_clock().now()
        while(self.get_clock().now() - start_time).nanoseconds < duration * 1e9:
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

    # NAVIGATION METHODS
    def find_frontier_direction(self,
                                num_rays: int = 72,
                                max_range: float = 10.0,
                                step: float = 0.05,
                                occ_thresh: int = 50,
                                unknown_is_blocked: bool = True) -> tuple[float, float]:
        
        if self.map_data is None:
            self.get_logger().error("No map data available.")
            return None, None

        if num_rays <= 0:
            self.get_logger().error("num_rays must be > 0")
            return None, None

        if step <= 0.0:
            self.get_logger().error("step must be > 0")
            return None, None
        
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        except tf2_ros.TransformException as exc:
            self.get_logger().warn(f"TF lookup failed (map->base_link): {exc}")
            return None, None

        rx = t.transform.translation.x
        ry = t.transform.translation.y
        q = t.transform.rotation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        best_rel = 0.0
        best_dist = -1.0

        for i in range(num_rays):
            rel = -math.pi + (2.0 * math.pi) * (i / num_rays)
            world_angle = yaw + rel

            d = 0.0
            while d < max_range:
                px = rx + d * math.cos(world_angle)
                py = ry + d * math.sin(world_angle)
                mp = self._world_to_map(px, py)
                if mp is None:
                    break
                val = self._cell(mp[0], mp[1])
                if val >= occ_thresh or (unknown_is_blocked and val < 0):
                    break
                d += step

            if d > best_dist:
                best_dist = d
                best_rel = rel
        
        return best_rel, best_dist
        
    
    # NAVIGATION HELPER METHODS
    def wait_for_scan_data(self, timeout_sec: float = 5.0) -> bool:
        
        """
        Waits until laser scan data is received or a timeout occurs.
        Args:
            timeout_sec(float): maximum time to wait for scan data in seconds (default: 5.0)
        Returns:
            bool: True if scan data was received, False if timeout occurred
        """
        start_time = self.get_clock().now()

        while rclpy.ok():
            if self.scan_data is not None:
                return True

            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed >= timeout_sec:
                self.get_logger().warn(f"No scan data received within {timeout_sec:.1f}s")
                return False

            rclpy.spin_once(self, timeout_sec=0.1)

    def wait_for_map_data(self, timeout_sec: float = 5.0) -> bool:
        start_time = self.get_clock().now()

        while rclpy.ok():
            if self.map_data is not None:
                return True

            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed >= timeout_sec:
                self.get_logger().warn(f"No map data received within {timeout_sec:.1f}s")
                return False

            rclpy.spin_once(self, timeout_sec=0.1)

    def _cell(self, mx: int, my: int) -> int:
        w = self.map_data.info.width
        return self.map_data.data[my * w + mx]
    
    def _world_to_map(self, x: float, y: float) -> tuple[int, int]:
        info = self.map_data.info
        ox = info.origin.position.x
        oy = info.origin.position.y
        res = info.resolution
        mx = int((x - ox) / res)
        my = int((y - oy) / res)
        if mx < 0 or my < 0 or mx >= info.width or my >= info.height:
            return None
        return mx, my

    # CALLBACKS
    def _scanCallback(self, msg: LaserScan) -> None:
        self.scan_data = msg

    def _mapCallback(self, msg: OccupancyGrid) -> None:
        self.map_data = msg

def main(args = None):

    rclpy.init(args = args)
    re = RobotExplorer()

    re.wait_for_scan_data()
    has_map = re.wait_for_map_data()

    if not has_map:
        print("No map data available; cannot compute frontier direction.")
        re.destroy_node()
        rclpy.shutdown()
        return

    while(True):
        angle, distance = re.find_frontier_direction()
        if angle is None or distance is None:
            print("No frontier direction found.")
        else:
            print(f"Best frontier direction: angle={math.degrees(angle):.1f} deg, distance={distance:.2f} m")
            re.turn(angle)
            re.move_straight(min(distance, 1.0))

    #re.turn(angle)
    #re.move_straight(distance - 0.5)
    #re.turn(re.scan_data.angle_max)

    # Shutdown
    re.destroy_node()
    rclpy.shutdown()


if __name__=="__main__":
    main()