#!/usr/bin/env python3

'''
Node, ki bo, kadar se ustavi na zelenem/rdečem cellu - ustavil se bo na enem koncu in potem šel proti drugemu
naredil bo tile detection (vprašanje, koliko ploščic je?)
za vsako ploščico bo naredil homografijo - da bo kvadrat
v datasetu imamo kvadratne slike velikosti 512x512 
se pravi ta node bo samo loadal model za anomaly detection, potem pa bo za vsako ploščico naredil homografijo in sliko poslal modelu
mora tudi shraniti koliko ploščic je - se pravi število ploščic ki imajo anomalije, in število, ki nima anomalij


imamo tudi kamero: top_camera

število ploščic je toliko da so od enega konca linije do drugega

da hkrati laufata cell detection in tile detection node

da si explerer zapomni koordinate rdečih /zelenih cellov, pa potem se vklopi 
cell detection, kjer poišče en konec linije, pa se počasi premika do drugega konca

najprej gre na desni konec, kamero pa ma gor na desno stran

ko dobi task - preko topica dobi kateri cell mora obiskat


potem je tukaj še vprašanje detekcije pravilnega cella - pasiven node, ki si shrani koordinate za rdeč in zelen cell

na task managerju se sproži cell detectiong, pa dobi ta task, in se vžge

'''


import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import subprocess

from anomaly_detector import AnomalyDetector


class detect_tiles(Node):
    def __init__(self):
        super().__init__('detect_tiles')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('device', ''),
                ('model_path', 'best_model.pth'), # TREBA ACTUALLY PREVERIT KJE JE LOCIRAN
                ('tile_size', 512),
                ('confidence_threshold', 0.5),
        ])

        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.tile_size = self.get_parameter('tile_size').get_parameter_value().integer_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        self.last_processed_tile_center = None
        self.tile_center_threshold = 50  # pixels

        self.bridge = CvBridge()
        self.scan = None
        self.detector = self.load_model()

        self.image_sub = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.image_callback, qos_profile_sensor_data)

        self.tiles = {
                'total': 0,
                'anomalous': 0,
                'normal': 0
        }

        self.get_logger().info("Tile detection node initialized")

    def say(self, text):
        """Use system spd-say for text-to-speech"""
        try:
            subprocess.run(["spd-say", "-r", "-60", "-p", "-55", "-t", "male3", text], check=False)
        except FileNotFoundError:
            print(f"spd-say not found. Would say: {text}")

    def load_model(self):
        # Use the existing AnomalyDetector class to load the model and handle preprocessing.
        detector = AnomalyDetector(
            model_path=self.model_path,
            threshold=self.confidence_threshold,
            device=self.device if self.device else None
        )
        return detector

    def print_detected_tiles(self):
        print(f"Total tiles: {self.tiles['total']}")
        print(f"Anomalous tiles: {self.tiles['anomalous']}")
        print(f"Normal tiles: {self.tiles['normal']}")
            
    def detect_tiles(self, image):
        # vrača lista s konturami ploščic

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # thresholding - treba prilagoditi vrednosti po potrebi
        _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tiles = []

        # za prilagodit
        min_area = 5000
        max_area = 50000

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:
                    tiles.append(approx)

        return tiles

    def get_perspective_transform(self, contour, tile_size=512):
        dst_points = np.array([[0, 0], [tile_size, 0], [tile_size, tile_size], [0, tile_size]], dtype=np.float32)

        contour = contour.reshape(4, 2).astype(np.float32)

        top = contour[:2][np.argsort(contour[:2, 0])]
        bottom = contour[2:][np.argsort(contour[2:, 0])]
        src_points = np.stack([top, bottom[::-1]]).astype(np.float32)

        H = cv2.getPerspectiveTransform(src_points, dst_points)
        return H
    
    def warp_tile(self, image, H, tile_size=512):
        warped = cv2.warpPerspective(image, H, (tile_size, tile_size))
        return warped
    
    def predict_anomaly(self, tile_image, save_path=None):
        # Use the detector's public prediction method, optionally saving a report.
        is_anomaly = self.detector.detect(tile_image, save_path=save_path)
        confidence = 1.0 if is_anomaly else 0.0
        return is_anomaly, confidence

    def get_tile_center(self, contour):
        """Calculate the center point of a tile contour"""
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        return None


    # callback ko prejme sliko iz top kamere
    def image_callback(self, msg):

        self.get_logger().info("Received image from top camera")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        
        cv2.imshow("Top Camera View", cv_image)
        cv2.waitKey(1)
        
        tiles_contours = self.detect_tiles(cv_image)

        if not tiles_contours:
            self.get_logger().info("No tiles detected in the image.")
            return
        self.get_logger().info(f"Detected {len(tiles_contours)} tiles in the image.")

        # Take the most prominent/largest tile (or center tile)
        largest_tile = max(tiles_contours, key=cv2.contourArea)
        center = self.get_tile_center(largest_tile)
        
        if center is None:
            return
        
        # Skip if it's the same tile as last frame
        if self.last_processed_tile_center is not None:
            distance = np.sqrt((center[0] - self.last_processed_tile_center[0])**2 + 
                            (center[1] - self.last_processed_tile_center[1])**2)
            if distance < self.tile_center_threshold:
                return  # Same tile, skip
        
        # New tile detected
        self.last_processed_tile_center = center
        # za vsako ploščico:
        for idx, contour in enumerate(tiles_contours):

            try:
                H = self.get_perspective_transform(contour, self.tile_size)
                warped_tile = self.warp_tile(cv_image, H, self.tile_size)
                
                cv2.imshow(f"Warped Tile {idx}", warped_tile)
                cv2.waitKey(1)

                report_path = f"reports/tile_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.png"
                is_anomaly, confidence = self.predict_anomaly(warped_tile, save_path=report_path)

                self.tiles['total'] += 1
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
