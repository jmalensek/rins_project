#! /usr/bin/env python3

import rclpy
import time
import math
import tf2_ros

from collections import deque

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

import numpy as np
import cv2
from sensor_msgs.msg import Image

map_qos = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

class RobotExplorer(Node):

    def __init__(self, node_name='robot_explorer', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        
        self.pose_frame_id = 'map'
        
        # FLAGS AND STATE VARIABLES
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None
        self.scan_data = None
        self.map_data = None
        self.finished_count = 0
        self.amcl_pose_msg = None
        self._amcl_window = deque()
        self.localisation_streak = 0
        self.yellow_detected = False

        self._last_bgr = None

        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # PUBLISHERS
        # Publisher for velocity commands to control the robot's movement
        self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)

        # Publisher to signal finish or interruption to other nodes
        self.finished_pub = self.create_publisher(Bool, "/finished", 10)

        # SUBSCRIBERS
        # Laser scan data for obstacle detection and navigation
        self.create_subscription(LaserScan, 'scan_filtered', self._scan_callback, qos_profile_sensor_data)

        # Occupancy grid map data for navigation and waypoint generation
        self.create_subscription(OccupancyGrid, 'map', self._map_callback, map_qos)

        # Finished signal for interruption mechanisms
        self.create_subscription(Bool, '/finished', self._finished_callback, 10)

        # AMCL pose estimates for localisation and navigation feedback
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amcl_pose_callback, 10)

        # Camera images for line following and visual processing
        self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self._image_callback, image_qos)

        # Subscription for yellow line detection
        self.create_subscription(Bool, '/yellow_line/detected', self._yellow_line_callback, 10)

        self.nav_to_pose_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # Initialisation successful
        self.get_logger().info(f"Robot explorer has been initialized!")

    # BASIC MOTORIC METHODS

    # Moves the robot straight for a given distance at a given speed
    def move_straight(self, distance: float, speed: float = 0.2) -> None:
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

    # Turns the robot in place by a given angle (in radians) at a given angular speed
    def turn(self, angle: float, angular_speed: float = 0.5) -> None:
        if angle is None:
            self.get_logger().warn("Turn angle is None; skipping turn command")
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

    # Rotates the robot in place by dividing a full circle into equal turns, with a wait time in between
    def rotate(self, turns: int = 4, angular_speed: float = 0.5, wait_time: float = 1.0) -> None:
            angle = (2 * math.pi) / turns

            for _ in range(turns):
                self.turn(angle, angular_speed)
                time.sleep(wait_time)

     # Moves the robot straight ahead at a given speed until it either detects an obstacle or a yellow line
    def move_in_area1(self, speed: float = 0.2, obst_range: float = 0.5, range_sample_n: int = 10,timeout_sec: float = 30.0) -> None:
        speed = abs(speed)

        twist_msg = TwistStamped()
        twist_msg.twist.linear.x = speed

        n_half = range_sample_n//2

        stop_msg = TwistStamped()

        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds < timeout_sec * 1e9:

            # If a yellow line is detected, stop
            if self.yellow_detected:
                self.get_logger().info("Yellow line detected; stopping movement.")
                break
            
            # If an obstacle is detected straight ahead (check narrow wedge of middle ranges) within a given range, stop
            if self.scan_data is not None:
                # Check the middle ranges for obstacles
                len_ranges = len(self.scan_data.ranges)
                middle_ranges = [
                    self.scan_data.ranges[i] for i in range(len_ranges//2 - n_half, len_ranges//2 + n_half) 
                    if not math.isinf(self.scan_data.ranges[i])
                    ]
                
                if min(middle_ranges) < obst_range:
                    self.get_logger().info("Obstacle detected; stopping movement.")
                    break

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

    # Moves the robot to a specific pose on the map using Nav2's navigate_to_pose action
    def go_to_pose(self, x: float, y: float, yaw: float = 0.0) -> bool:

        # Check if mapa data is available before sending the goal
        if self.map_data is None:
            self.get_logger().error("Map data is not available")
            return False

        # Check if the goal is within the map bounds and in free space before sending the goal
        map_cell = self._world_to_map(x, y)
        if map_cell is None:
            self.get_logger().warn(f"Goal ({x:.2f}, {y:.2f}) is outside the known map")
            return False

        mx, my = map_cell
        if self._cell(mx, my) != 0:
            self.get_logger().warn(
                f"Goal ({x:.2f}, {y:.2f}) is not in free space (map cell value: {self._cell(mx, my)})"
                )
            return False

        # Wait for Nav2 action server
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info("Waiting for navigate_to_pose action server...")

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(x)
        goal_pose.pose.position.y = float(y)
        goal_pose.pose.orientation = self._yaw_to_quaternion(yaw)

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if self.goal_handle is None or not self.goal_handle.accepted:
            self.get_logger().warn(f"Goal rejected at ({x:.2f}, {y:.2f})")
            return False

        self.result_future = self.goal_handle.get_result_async()
        self.get_logger().info(f"Goal accepted, navigating to ({x:.2f}, {y:.2f})")
        return True

    # NAVIGATION METHODS

    def explore_area1(self, speed: float = 0.2, angular_speed: float = 1.0, timeout_sec: float = 120.0) -> None:
        self.get_logger().info("Starting exploration of area 1...")

        start_time = self.get_clock().now()

        # Move straight ahead until an obstacle or yellow line is detected, then turn left/right and repeat until timeout
        while rclpy.ok() and (self.get_clock().now() - start_time).nanoseconds < timeout_sec * 1e9:
            self.get_logger().info("Moving ahead...")
            self.move_in_area1(speed=speed, timeout_sec=timeout_sec)

            self.get_logger().info("Turning to explore new direction...")
            self.turn(math.pi/2, angular_speed=angular_speed)

    # Sequentially visits a set of waypoints on a provided map
    def cover_waypoints(self, waypoints: list[tuple[float, float]], turns: int = 4, angular_speed: float = 0.5, wait_time: float = 1.0, localise: bool = True) -> None:
        for index, (x, y) in enumerate(waypoints):

            if self.finished_count > 0:
                self.get_logger().info("Interrupt signal received, waiting...")

                # Publish a /finished message to signal interruption to other nodes
                self.finished_pub.publish(Bool(data=True))
                self.get_logger().info("Published /finished=True to signal interruption to other nodes.")

                # Waits for the next /finished message to arrive
                while self.finished_count < 2:
                    rclpy.spin_once(self, timeout_sec=0.1)

                self.get_logger().info("Resuming waypoint coverage...")
                self.finished_count = 0

            # Compute the yaw angle to face at the next waypoint
            yaw = 0.0
            if index < len(waypoints) - 1:
                yaw = self.compute_absolute_yaw((x, y), waypoints[index + 1])
            elif index > 0:
                yaw = self.compute_absolute_yaw(waypoints[index - 1], (x, y))

            self.get_logger(). info(f"Navigating to waypoint ({x:.2f}, {y:.2f}) with yaw {yaw:.2f} rad")

            # Rotate to help with localisation
            if localise:
                self.rotate(turns, angular_speed, wait_time)

            # Compute the yaw for the next

            if not self.go_to_pose(x, y, yaw):
                self.get_logger().error(f"Failed to send goal to ({x:.2f}, {y:.2f})")
                continue
            
            if not self.wait_task_done(timeout_sec=30.0):
                self.get_logger().error(f"Failed to reach ({x:.2f}, {y:.2f}) within timeout")
            else:
                self.get_logger().info(f"Successfully reached ({x:.2f}, {y:.2f})")

    # Simple routine to localise the robot on the map
    def localise_self(self, turns: int = 4, angular_speed: float = 0.5, wait_time: float = 1.0) -> None:
        self.get_logger().info("Starting self-localization procedure...")

        while not self.is_localised():
            self.get_logger().info("Robot localising...")

            self.rotate(turns, angular_speed, wait_time)

    # Follows a blue line on the ground
    # The line possibly branches out, so the robot should follow the leftmost branch
    # and return when the line ends
    # IMPORTANT: Set the arm position to look_for_qr
    def follow_blue_line(
        self,
        linear_speed: float = 0.2,
        max_angular_speed: float = 0.5,
        k_p: float = 0.004,
        min_area: int = 500,
        lost_timeout_sec: float = 2.0,
        loop_dt: float = 0.1,
        ) -> None:
        self.get_logger().info("Starting to follow the blue line...")

        uturn_attempts = 0
        max_uturn_attempts = 4

        # Define HSV color range for blue line detection
        lower_blue = np.array([95, 80, 40], dtype=np.uint8)
        upper_blue = np.array([140, 255, 255], dtype=np.uint8)

        kernel = np.ones((5, 5), np.uint8)

        # Function to publish velocity commands
        def publish_cmd(v: float, w: float) -> None:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.twist.linear.x = float(v)
            msg.twist.angular.z = float(w)
            self.cmd_vel_pub.publish(msg)

        # Wait for the first camera image to be received
        start_time = self.get_clock().now()
        while rclpy.ok() and self._last_bgr is None:
            if (self.get_clock().now() - start_time).nanoseconds * 1e-9 > 2.0:
                self.get_logger().warn("No camera image received; cannot follow line.")
                return
            rclpy.spin_once(self, timeout_sec=0.1)

        lost_time = 0.0

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)

            img = self._last_bgr
            if img is None:
                time.sleep(loop_dt)
                continue
            
            # Only process the lower part of the image to focus on the area near the robot.
            h, w, _ = img.shape
            y0 = int(h * 0.77) # 0.77 kind of works fine with the look_for_qr camera pose
            roi = img[y0:h, :]

            # Convert to HSV color space and create a mask for the blue line
            img_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for c in contours if cv2.contourArea(c) >= min_area]

            # If no blue line is detected, increment the lost time
            if not contours:
                lost_time += loop_dt
                publish_cmd(0.0, 0.0)

                # If the line has been lost for too long, assume a dead end and attempt a U-turn
                if lost_time >= lost_timeout_sec:

                    # Allow a limited amount of U-turn attempts
                    if uturn_attempts < max_uturn_attempts:
                        self.get_logger().info("Line lost; turning around to continue following.")
                        for _ in range(3):
                            publish_cmd(0.0, 0.0)
                            time.sleep(0.05)

                        self.turn(math.pi/2, angular_speed=max(0.2, max_angular_speed))
                        uturn_attempts += 1
                        lost_time = 0.0
                        time.sleep(loop_dt)
                        continue
                    
                    # If maximum U-turn attempts reached, assume the line is fully lost and exit
                    for _ in range(3):
                        publish_cmd(0.0, 0.0)
                        time.sleep(0.05)
                    self.get_logger().info("Line lost; giving up after repeated turnarounds.")
                    return

                time.sleep(loop_dt)
                continue
            
            # If the line is detected, reset the lost time and U-turn attempts
            lost_time = 0.0
            uturn_attempts = 0
            best = None
            best_cx = None

            # Find the contour with the smallest x-coordinate of the centroid (leftmost)
            for c in contours:
                M = cv2.moments(c)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10'] / M['m00'])
                if best is None or cx < best_cx:
                    best = c
                    best_cx = cx

            # If no valid contour is found, treat it as lost
            if best is None:
                lost_time += loop_dt
                publish_cmd(0.0, 0.0)
                time.sleep(loop_dt)
                continue
            
            # Compute the error in pixels between the centroid of the detected line 
            # and the center of the image
            target_x = float(best_cx)
            center_x = float(w) / 2.0
            error_px = target_x - center_x

            # Compute the angular velocity
            angular = -k_p * error_px
            angular = max(-max_angular_speed, min(max_angular_speed, angular))

            # Compute the linear velocity
            speed_scale = max(0.3, 1.0 - abs(angular) / max_angular_speed)
            v = linear_speed * speed_scale

            # Move the robot
            publish_cmd(v, angular)
            time.sleep(loop_dt)

    # NAVIGATION HELPER METHODS

    # Makes a grid of points using two opposing corner bounds
    def generate_grid(self, corner1: tuple[float, float], corner2: tuple[float, float], step: float = 1.0) -> list[tuple[float,float]]:
        x1, y1 = corner1
        x2, y2 = corner2

        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)

        grid_points = []
        y = min_y
        row_index = 0
        while y <= max_y:
            row_points = []
            x = min_x
            while x <= max_x:
                row_points.append((x, y))
                x += step

            if row_index % 2 == 1:
                row_points.reverse()

            grid_points.extend(row_points)
            y += step
            row_index += 1

        # Reverse the order of the points
        grid_points.reverse()

        return grid_points

    # Computes the yaw between two absolute poses
    def compute_absolute_yaw(self, from_pose: tuple[float, float], to_pose: tuple[float, float]) -> float:
        from_x, from_y = from_pose
        to_x, to_y = to_pose
        return math.atan2(to_y - from_y, to_x - from_x)
    
    # Compute the relative yaw to a target pose relative to the current robot pose
    # The coordinates of the target pose are relative, i.e. current pose is (0,0)
    def compute_relative_yaw(self, target_pose: tuple[float, float]) -> float:
        target_x, target_y = target_pose
        return math.atan2(target_y, target_x)
    
    # Computes the distance between two poses
    def compute_distance(self, pose1: tuple[float, float], pose2: tuple[float, float]) -> float:
        x1, y1 = pose1
        x2, y2 = pose2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    # Computes the distance to a relative target pose
    # The coordinates of the target pose are relative, i.e. current pose is (0,0)
    def compute_relative_distance(self, target_pose: tuple[float, float]) -> float:
        target_x, target_y = target_pose
        return math.sqrt(target_x ** 2 + target_y ** 2)

    # Checks whether amcl pose is stable an the robot is well-localised
    def is_localised(self, pos_threshold: float = 0.2, yaw_threshold: float = 0.2, min_streak: int = 3) -> bool:
        if self.amcl_pose_msg is None:
            self.get_logger().warn("AMCL pose not received yet")
            return False
        
        cov = self.amcl_pose_msg.pose.covariance
        var_x = cov[0]
        var_y = cov[7]
        var_yaw = cov[35]

        std_x = math.sqrt(max(var_x, 0.0))
        std_y = math.sqrt(max(var_y, 0.0))
        std_yaw = math.sqrt(max(var_yaw, 0.0))

        if std_x < pos_threshold and std_y < pos_threshold and std_yaw < yaw_threshold:
            self.localisation_streak += 1
        else:
            self.localisation_streak = 0

        return self.localisation_streak >= min_streak

    # Divides the map into a grid of points and returns those that are in free space
    def get_waypoints_from_map(self, step: float = 1.0) -> list[tuple[float, float]]:
        if self.map_data is None:
            self.get_logger().error("Map data not available")
            return []

        waypoints = []
        info = self.map_data.info
        stride = max(1, int(step / info.resolution))

        for my in range(0, info.height, stride):
            x_iter = range(0, info.width, stride)
            if (my // stride) % 2 == 1:
                x_iter = reversed(x_iter)
            
            for mx in x_iter:
                if self._cell(mx, my) == 0:
                    wx, wy = self._map_to_world(mx, my)
                    waypoints.append((wx, wy))

        self.get_logger().info(f"Generated {len(waypoints)} waypoints from map with step {step}m")
        return waypoints

    # Helper method to convert yaw angle to quaternion for goal pose
    def _yaw_to_quaternion(self, yaw: float) -> Quaternion:
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw * 0.5)
        q.w = math.cos(yaw * 0.5)
        return q

    def _quaternion_to_yaw(self, q: Quaternion) -> float:
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    # Waits for scan data to be received, with a timeout
    def wait_for_scan_data(self, timeout_sec: float = 5.0) -> bool:
        start_time = self.get_clock().now()

        while rclpy.ok():
            if self.scan_data is not None:
                return True

            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed >= timeout_sec:
                self.get_logger().warn(f"No scan data received within {timeout_sec:.1f}s")
                return False

            rclpy.spin_once(self, timeout_sec=0.1)

    # Waits for map data to be received, with a timeout
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

    # Helper method to get the occupancy value of a cell in the map
    def _cell(self, mx: int, my: int) -> int:
        w = self.map_data.info.width
        return self.map_data.data[my * w + mx]
    
    # Helper method to convert world coordinates to map cell indices
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

    # UTILITY METHODS

    # Waits for the current navigation task to complete, with a timeout, and returns whether it succeeded
    def wait_task_done(self, timeout_sec: float = 120.0) -> bool:
        start = self.get_clock().now()
        while rclpy.ok():
            rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.2)
            if self.result_future.done():
                result = self.result_future.result()
                if result is None:
                    return False
                status = result.status
                return status == GoalStatus.STATUS_SUCCEEDED

            elapsed = (self.get_clock().now() - start).nanoseconds / 1e9
            if elapsed > timeout_sec:
                self.get_logger().warn("Navigation timeout, canceling goal.")
                if self.goal_handle is not None:
                    cancel_future = self.goal_handle.cancel_goal_async()
                    rclpy.spin_until_future_complete(self, cancel_future)
                return False
        return False
    
    # Cancels the current navigation goal
    def cancel_task(self) -> bool:
        if self.goal_handle is not None:
            cancel_future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future)
            return cancel_future.result().return_code == GoalStatus.STATUS_SUCCEEDED
        return False

    # Helper method to convert map cell indices to world coordinates (center of the cell)
    def _map_to_world(self, mx: int, my: int) -> tuple[float, float]:
        info = self.map_data.info
        wx = info.origin.position.x + (mx + 0.5) * info.resolution
        wy = info.origin.position.y + (my + 0.5) * info.resolution
        return wx, wy

    # CALLBACKS
    def _scan_callback(self, msg: LaserScan) -> None:
        try:
            self.scan_data = msg
        except Exception:
            self.get_logger().warn('Received malformed scan message')

    def _map_callback(self, msg: OccupancyGrid) -> None:
        try:
            self.map_data = msg
        except Exception:
            self.get_logger().warn('Received malformed map message')

    def _finished_callback(self, msg: Bool) -> None:
        if msg.data:
            self.finished_count += 1

    def _amcl_pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        try:
            self.amcl_pose_msg = msg
            self._amcl_window.append(msg)
            if len(self._amcl_window) > 20:
                self._amcl_window.popleft()
        except Exception:
            self.get_logger().warn('Received malformed amcl_pose message')

    def _yellow_line_callback(self, msg: Bool) -> None:
        # store latest detection state
        try:
            self.yellow_detected = bool(msg.data)
        except Exception:
            self.get_logger().warn('Received malformed /yellow_line/detected message')

    def _image_callback(self, msg: Image) -> None:
        try:
            encoding = (msg.encoding or "").lower()

            if encoding in ("bgr8", "rgb8"):
                channels = 3
            elif encoding in ("mono8", "8uc1"):
                channels = 1
            else:
                self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}")
                return

            row_stride = int(msg.step)
            if row_stride <= 0:
                self.get_logger().warn("Image step/stride is invalid")
                return

            expected_row_bytes = int(msg.width) * channels
            buf = np.frombuffer(msg.data, dtype=np.uint8)

            if buf.size < int(msg.height) * row_stride:
                self.get_logger().warn("Image buffer is smaller than expected")
                return

            if row_stride == expected_row_bytes:
                if channels == 3:
                    img = buf[: int(msg.height) * expected_row_bytes].reshape((msg.height, msg.width, 3))
                else:
                    img = buf[: int(msg.height) * expected_row_bytes].reshape((msg.height, msg.width))
            else:
                rows = buf[: int(msg.height) * row_stride].reshape((msg.height, row_stride))
                rows = rows[:, :expected_row_bytes]
                if channels == 3:
                    img = rows.reshape((msg.height, msg.width, 3))
                else:
                    img = rows.reshape((msg.height, msg.width))

            if encoding == "rgb8":
                img = img[:, :, ::-1]
            elif channels == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            self._last_bgr = img
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image without cv_bridge: {e}")

def main(args = None):

    rclpy.init(args=args)
    re = RobotExplorer()

    time.sleep(3.0)

    if not re.wait_for_map_data(timeout_sec=10.0):
        re.get_logger().error('Map was not received, aborting.')
        re.destroy_node()
        rclpy.shutdown()
        return

    #re.localise_self()

    # hardcoded waypoints for the specific map
    area1_lower_bound = (0.7719925413173669, -4.4194223905944074)
    area1_upper_bound = (-4.644132348851085, 0.27827839427878537)
    blue_line_start = (2.797641390882936, 0.19569027139125528)

    blue_line_start_quaternion_yaw = re._quaternion_to_yaw(Quaternion(x=-0.0, y=0.0, z=0.707, w=0.707))

    waypoints_task1_sim = [
        (1.8396370262122645, -0.5751383533952286),
        (2.1833755802533554, -1.9678722468643208),
        (1.417301595011825, -2.666432722817975),
        (0.12696714192880537, -3.6631788241089063),
        (0.017924597277648373, -0.9779664499620027),
        (-0.011229038107887649,-2.3238756805248095),
        (-1.0622132922149174, -2.411171268639826),
        (-1.2901576432001445, -1.089920196088445),
        (-1.7972970814236089, -1.0256533410565287),
        (-1.9365451987254536, 0.4943115005568502),
        (-1.1507406707967371, 1.1061146346431132),
        (0.3933188510497119, 1.2343684129661787),
        (1.261126302529096, 1.6370923481849204),
        (0.6003921547337148, 2.6967011040456237),
        (-2.4022780168470765, 2.4666737072366414),
        (-2.6826873711016357, 0.7170108058401835)
        ]

    waypoints_task2_sim_area1 = [
        (0.8166266381361322, -1.384629774657411), 
        (0.9365609692799964, -2.559109391122671),
        (0.23718536869227425, -3.5608393315767075),
        (0.24711903053401427, -4.339664760167153),
        (-1.5414897143031596, -4.632740467430283),
        (-1.301480869050944, -3.472649222315032),
        (-1.1418930447646625, -1.0572847621756578),
        (-2.2146533506622936, -0.93424623199067),
        (-2.405889798581263, 8.960575301129037e-05),
        (-3.4066038384624733, -0.5416958313224103),
        (-4.288971251770049, -0.5087648881189769),
        (-4.342294411914256, -2.187262912653518),
        (-2.5173866494164465, -2.7962294350925982)
        ]

    waypoints_irl = [(-0.9063306841864244, -0.13056368384585265), 
                    (-1.6952106812727537, 0.2388238522478004),
                    (-2.06263535000406, -0.4041383919945738),
                    (-2.984074087218118, -0.6223435753081675),
                    (-2.9796109425001718, -1.0694166887545653),
                    (-3.26013179610062, -1.1159374308024788)]

    #waypoints_task2_sim_area1 = re.generate_grid(area1_upper_bound, area1_lower_bound, step=1.5)
    #print(f"Generated {len(waypoints_task2_sim_area1)} waypoints for area 1")

    #re.turn(2 * math.pi)
    #re.move_in_area1(speed=0.2, timeout_sec=30.0)
    #re.turn(math.pi/2)
    #re.move_in_area1(speed=0.2, timeout_sec=30.0)

    # TASK 2
    # For line detection run detect_yellow_line.py and arm_mover_actions.py
    # For best view of both the yellow and blue line 
    #re.cover_waypoints(waypoints_task2_sim_area1, localise=False)
    #re.go_to_pose(*blue_line_start, yaw=blue_line_start_quaternion_yaw)
    #re.follow_blue_line()

    # TASK 1R
    re.cover_waypoints(waypoints_irl, wait_time=0.5, localise=True)
    
    re.get_logger().info("Exploration complete, shutting down.")
    re.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
