#!/usr/bin/env python3

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32

class YellowLineDetector(Node):
    def __init__(self):
        super().__init__("yellow_line_detector")

        # Parameters (tune these live with ros2 param set)
        self.declare_parameter("image_topic", "/top_camera/rgb/preview/image_raw")
        self.declare_parameter("roi_y_start_ratio", 0.5)   # amount of image to process 0.7-default
        self.declare_parameter("min_contour_area", 300.0)

        # HSV bounds for yellow (OpenCV H: 0-179)
        self.declare_parameter("h_low", 18)
        self.declare_parameter("s_low", 80)
        self.declare_parameter("v_low", 80)
        self.declare_parameter("h_high", 40)
        self.declare_parameter("s_high", 255)
        self.declare_parameter("v_high", 255)

        # "Ahead" window in ROI (center-bottom region)
        self.declare_parameter("ahead_window_x_min_ratio", 0.45)
        self.declare_parameter("ahead_window_x_max_ratio", 0.55)
        self.declare_parameter("ahead_window_y_min_ratio", 0.1)
        self.declare_parameter("ahead_pixel_ratio_threshold", 0.04)

        image_topic = self.get_parameter("image_topic").value

        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            )

        self.sub = self.create_subscription(
            Image, 
            image_topic, 
            self._image_callback, 
            image_qos
            )
        
        self.pub_detected = self.create_publisher(Bool, "/yellow_line/detected", 10)
        self.pub_ahead = self.create_publisher(Bool, "/yellow_line/ahead", 10)
        self.pub_offset = self.create_publisher(Float32, "/yellow_line/offset_px", 10)

        self.get_logger().info(f"Yellow line detector started. Subscribing to: {image_topic}")

    def _rosimg_to_bgr(self, msg: Image):
        encoding = (msg.encoding or "").lower()
        if encoding not in ("bgr8", "rgb8", "mono8", "8uc3", "8uc1"):
            self.get_logger().error(f"Unsupported image encoding: {msg.encoding}")
            return None
        
        channels = 3 if encoding in ("bgr8", "rgb8") else 1
        row_stride = int(msg.step)
        if row_stride <= 0:
            self.get_logger().error(f"Invalid row stride: {row_stride}")
            return None
        
        expected_row_bytes = int(msg.width) * channels
        buf = np.frombuffer(msg.data, dtype=np.uint8)

        if buf.size < int(msg.height) * row_stride:
            self.get_logger().error(f"Image data size {buf.size} is smaller than expected {msg.height * row_stride}")
            return None
        
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

        return img
    
    def _image_callback(self, msg: Image):
        img = self._rosimg_to_bgr(msg)
        if img is None:
            return

        h, w, _ = img.shape

        roi_y_start = float(self.get_parameter("roi_y_start_ratio").value)
        y0 = max(0, min(h - 1, int(h * roi_y_start)))
        roi = img[y0:h, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower = np.array(
            [
                int(self.get_parameter("h_low").value),
                int(self.get_parameter("s_low").value),
                int(self.get_parameter("v_low").value),
            ],
            dtype=np.uint8,
        )
        upper = np.array(
            [
                int(self.get_parameter("h_high").value),
                int(self.get_parameter("s_high").value),
                int(self.get_parameter("v_high").value),
            ],
            dtype=np.uint8,
        )

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = float(self.get_parameter("min_contour_area").value)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        detected = len(contours) > 0

        # Offset: centroid of largest contour relative to full image center
        offset_px = 0.0
        if detected:
            largest = max(contours, key=cv2.contourArea)
            m = cv2.moments(largest)
            if m["m00"] > 1e-6:
                cx_roi = float(m["m10"] / m["m00"])
                offset_px = cx_roi - (w / 2.0)

        # "Ahead" test: yellow occupancy in center-bottom window of ROI
        x_min_r = float(self.get_parameter("ahead_window_x_min_ratio").value)
        x_max_r = float(self.get_parameter("ahead_window_x_max_ratio").value)
        y_min_r = float(self.get_parameter("ahead_window_y_min_ratio").value)
        ratio_th = float(self.get_parameter("ahead_pixel_ratio_threshold").value)

        rx0 = int(max(0, min(w - 1, w * x_min_r)))
        rx1 = int(max(rx0 + 1, min(w, w * x_max_r)))
        rh = roi.shape[0]
        ry0 = int(max(0, min(rh - 1, rh * y_min_r)))
        ry1 = rh

        center_mask = mask[ry0:ry1, rx0:rx1]
        yellow_ratio = float(np.count_nonzero(center_mask)) / float(center_mask.size) if center_mask.size > 0 else 0.0
        ahead = yellow_ratio >= ratio_th

        self.pub_detected.publish(Bool(data=detected))
        self.pub_ahead.publish(Bool(data=ahead))
        self.pub_offset.publish(Float32(data=float(offset_px)))

        if ahead:
            #self.get_logger().warn(f"Yellow line ahead; Offset={offset_px:.1f} px, ratio={yellow_ratio:.3f}")
            pass


def main(args=None):
    rclpy.init(args=args)
    node = YellowLineDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
