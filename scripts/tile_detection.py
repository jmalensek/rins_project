#!/usr/bin/env python3

'''
Node, ki bo, kadar se ustavi na zelenem/rdečem cellu - ustavil se bo na enem koncu in potem šel proti drugemu
naredil bo tile detection (vprašanje, koliko ploščic je?)
...
'''

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Bool
import sensor_msgs_py.point_cloud2 as pc2

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
from datetime import datetime
import subprocess

from anomaly_detector import AnomalyDetector


from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rins_interfaces.msg import TilesResults



class detect_tiles(Node):
    def __init__(self):
        super().__init__('detect_tiles')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('device', ''),
                ('model_path', '/home/kappa/Documents/Task2/src/rins_project/best_model.pth'), 
                ('tile_size', 512),
                ('confidence_threshold', 0.5),
                ('tile_detection_start_topic', '/tile_detection/start'),
                ('tile_detection_stop_topic', '/tile_detection/stop'),
        ])

        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.tile_size = self.get_parameter('tile_size').get_parameter_value().integer_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        self.tile_detection_start_topic = self.get_parameter('tile_detection_start_topic').get_parameter_value().string_value
        self.tile_detection_stop_topic = self.get_parameter('tile_detection_stop_topic').get_parameter_value().string_value

        # Replace pointcloud tracking with TF2 components
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Spatial tracking (X, Y in meters relative to 'map' or 'odom' frame)
        self.processed_tile_positions = []  # List of (x, y) tuples in meters
        self.distance_threshold_meters = 0.25
        self.detection_active = False  

        # You can remove the pointcloud subscriber entirely if you don't need it for anything else!
        self.latest_pointcloud = None

        self.bridge = CvBridge()
        self.detector = self.load_model()

        self.status_list = []
        self.report_tile_pub = self.create_publisher(TilesResults, "/tiles_results", 10)

        # Subscriptions
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/oakd/rgb/preview/depth/points',
            self.pointcloud_callback,
            qos_profile_sensor_data,
        )

        self.image_sub = self.create_subscription(
            Image, 
            "/top_camera/rgb/preview/image_raw", 
            self.image_callback, 
            qos_profile_sensor_data
        )
        
        self.tile_start_sub = self.create_subscription(Bool, self.tile_detection_start_topic, self.on_tile_detection_start, 10)
        self.tile_stop_sub = self.create_subscription(Bool, self.tile_detection_stop_topic, self.on_tile_detection_stop, 10)

        self.tiles = {
                'total': 0,
                'anomalous': 0,
                'normal': 0
        }

        self.get_logger().info("Tile detection node initialized")

    def get_robot_pose_in_map(self):
        """
        Looks up the current 2D position (X, Y) of the robot in the global map frame.
        """
        try:
            # Target frame: 'map', Source frame: 'base_link' (robot center)
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('map', 'base_link', now)
            
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            return (x, y)
            
        except TransformException as ex:
            self.get_logger().warning(f"Could not transform 'base_link' to 'map': {ex}")
            return None

    def pointcloud_callback(self, msg: PointCloud2):
        """Stores the incoming point cloud message for spatial distance calculations."""
        self.latest_pointcloud = msg


    def say(self, text):
        """Use system spd-say for text-to-speech"""
        try:
            subprocess.run(["spd-say", "-r", "-60", "-p", "-55", "-t", "male3", text], check=False)
        except FileNotFoundError:
            print(f"spd-say not found. Would say: {text}")

    def load_model(self):
        return AnomalyDetector(
            model_path=self.model_path,
            threshold=self.confidence_threshold,
            device=self.device if self.device else None
        )

    def on_tile_detection_start(self, msg: Bool):
        if msg.data:
            self.detection_active = True
            self.get_logger().info("Tile detection activated")

    def on_tile_detection_stop(self, msg: Bool):
        if msg.data:
            self.detection_active = False
            self.get_logger().info("Tile detection deactivated")
            self.print_detected_tiles()
            self.publish_tile_results()
            self.destroy_node()
            rclpy.shutdown()

    def print_detected_tiles(self):
        print(f"Total tiles: {self.tiles['total']}")
        print(f"Anomalous tiles: {self.tiles['anomalous']}")
        print(f"Normal tiles: {self.tiles['normal']}")

    def publish_tile_results(self):
        msg = TilesResults()
        msg.total = self.tiles['total']
        msg.status = self.status_list
        self.report_tile_pub.publish(msg)
        self.get_logger().info(f"Published TilesResults: total={msg.total}, status={msg.status}")
            
    def detect_tiles(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tiles = []

        min_area = 5000
        max_area = 50000
        
        height, width = image.shape[:2]
        border_margin = 1  

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) != 4:
                    continue

                # Filter out tiles cut off by image borders
                points = approx.reshape(4, 2)
                touches_border = False
                for pt in points:
                    if (pt[0] <= border_margin or pt[0] >= width - border_margin or
                        pt[1] <= border_margin or pt[1] >= height - border_margin):
                        touches_border = True
                        break
                
                if touches_border:
                    continue

                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)

                if not (0.7 < aspect_ratio < 1.3):
                    continue

                tiles.append(approx)

        return tiles

    def get_perspective_transform(self, contour, tile_size=512):
        pts = contour.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  
        rect[2] = pts[np.argmax(s)]  

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  
        rect[3] = pts[np.argmax(diff)]  

        dst = np.array([
            [0, 0],
            [tile_size - 1, 0],
            [tile_size - 1, tile_size - 1],
            [0, tile_size - 1]
        ], dtype=np.float32)

        return cv2.getPerspectiveTransform(rect, dst)
    
    def warp_tile(self, image, H, tile_size=512):
        return cv2.warpPerspective(image, H, (tile_size, tile_size))
    
    def predict_anomaly(self, tile_image, save_path=None):
        is_anomaly = self.detector.detect(tile_image, save_path=save_path)
        return is_anomaly, 1.0 if is_anomaly else 0.0

    def get_tile_center(self, contour):
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        return None

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        
        cv2.imshow("Top Camera View", cv_image)
        cv2.waitKey(1)
        
        if not self.detection_active:
            return
        
        tiles_contours = self.detect_tiles(cv_image)

        debug_image = cv_image.copy()
        cv2.drawContours(debug_image, tiles_contours, -1, (0, 255, 0), 3)
        cv2.imshow("Detected Tiles", debug_image)
        cv2.waitKey(1)

        if not tiles_contours:
            return

        self.get_logger().info(f"Detected {len(tiles_contours)} candidate tiles.")

        # Get where the robot is globally standing right now
        robot_pose = self.get_robot_pose_in_map()
        if robot_pose is None:
            self.get_logger().warning("Skipping tile processing: Robot global pose unavailable.")
            return

        robot_x, robot_y = robot_pose

        # Process the detected contours
        for idx, contour in enumerate(tiles_contours):
            
            # 1. Distance constraint check based purely on global robot location
            is_duplicate = False
            for old_x, old_y in self.processed_tile_positions:
                # 2D Euclidean distance check in meters
                distance = np.sqrt((robot_x - old_x)**2 + (robot_y - old_y)**2)
                if distance < self.distance_threshold_meters:
                    is_duplicate = True
                    break
                    
            if is_duplicate:
                # Robot hasn't moved 0.25 meters away from the last logged tile position yet
                continue
                
            # If it's a completely fresh tile location sequence step:
            self.processed_tile_positions.append((robot_x, robot_y))
            self.get_logger().info(f"New tile registered at robot map position: X={robot_x:.2f}, Y={robot_y:.2f}")

            # 2. Proceed safely with homography and your AI model evaluation
            try:
                H = self.get_perspective_transform(contour, self.tile_size)
                warped_tile = self.warp_tile(cv_image, H, self.tile_size)
                
                cv2.imshow(f"Warped Tile {idx}", warped_tile)
                cv2.waitKey(1)

                os.makedirs("reports", exist_ok=True)
                report_path = f"reports/tile_cell_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                is_anomaly, confidence = self.predict_anomaly(warped_tile, save_path=report_path)

                self.tiles['total'] += 1
                self.status_list.append('NOK' if is_anomaly else 'OK')
                if is_anomaly:
                    self.tiles['anomalous'] += 1
                    self.say("Anomaly detected")
                else:
                    self.tiles['normal'] += 1
                    
            except Exception as e:
                self.get_logger().error(f"Error processing tile {idx}: {e}")
                continue

def main():
    print('Tile detection node starting.')
    rclpy.init(args=None)
    node = detect_tiles()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down tile detector.")
    finally:
        node.print_detected_tiles()
        node.destroy_node()
        try:    
            rclpy.shutdown()
        except Exception as e:
            print(f"Error during shutdown: {e}")

if __name__ == '__main__':
    main()
