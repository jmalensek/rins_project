#!/usr/bin/env python3


# HAVE TO DETECT TWO RINGS AND THEIR COLOR
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import subprocess

import subprocess

from tf2_ros import TransformListener, Buffer, TransformException
from tf2_geometry_msgs import do_transform_point


class detect_rings(Node):

    def __init__(self):
        super().__init__('detect_rings')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('device', ''),
        ])

        marker_topic = "/rings_marker"

        self.detection_color = (0,0,255)  # question about that
        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.scan = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

        self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.RELIABLE)

        self.rings = []
        # clustering
        self.ring_clusters = []
        self.cluster_threshold = 0.20  # 20 centimers

        self.min_detections = 5  # min hits before publishing a marker

        self.ema_alpha = 0.1  # EMA smoothing factor for centroid

        self.create_timer(1.0, self.publish_clusters)
       

        self.get_logger().info(f"Node has been initialized! Will publish ring markers to {marker_topic}.")

    def say(self, text):
        """Use system spd-say for text-to-speech"""
        try:
            subprocess.run(["spd-say", "-r", "-60", "-p", "-55", "-t", "male3", text], check=False)
        except FileNotFoundError:
            print(f"spd-say not found. Would say: {text}")

    def rgb_callback(self, data):
        self.rings = []

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            #self.get_logger().info(f"Running Hough Circle detection...")

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (9, 9), 2)

            # detectinh ONLY on upper half :)
            upper_h = gray.shape[0] // 2
            gray_upper = gray[:upper_h, :]

            circles = cv2.HoughCircles(gray_upper, cv2.HOUGH_GRADIENT, dp=1.0, minDist=50,
                                     param1=90, param2=40, minRadius=3, maxRadius=200)  # params could be changed
# maybe change param2 si it would depend also on radius? smaller radius -> smaller threshold for detection?
            if circles is not None:
                circles = np.uint16(np.around(circles))  # has to be uint16 -> for drawing

                accepted = []

                for c in circles[0, :]:
                    cx, cy, r = c

                    duplicate = False
                    for ax, ay, ar in accepted:
                        if np.hypot(cx - ax, cy - ay) < max(8, 0.5 * min(r, ar)) and abs(int(r) - int(ar)) < max(5, 0.3 * min(r, ar)):
                            duplicate = True
                            break
                    if duplicate:
                        continue
                    accepted.append((cx, cy, r))

                    #self.get_logger().info("Ring detected!")

                    # color detection
                    color = self.detect_ring_color(cv_image, cx, cy, r)



                    # Draw circle
                    cv2.circle(cv_image, (cx, cy), r, self.detection_color, 2)

                    # draw center
                    cv2.circle(cv_image, (cx, cy), 3, (0, 0, 255), -1)

                    if color:
                        #self.get_logger().info(f"Ring colour: {color['name']} | LAB: {[round(v,1) for v in color['lab']]}")
                        # Label on image
                        cv2.putText(cv_image, color['name'], (cx - r, cy - r - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    self.rings.append((cx, cy, r, color))

            cv2.imshow("rings", cv_image)
            key = cv2.waitKey(1)
            if key == 27:
                print("exiting")
                exit()

        except CvBridgeError as e:
            print(e)




    def pointcloud_callback(self, data):
        # get point cloud attributes
        height = data.height
        width = data.width

        # get 3-channel representation of the point cloud in numpy format once
        a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
        a = a.reshape((height, width, 3))

        source_frame = data.header.frame_id if data.header.frame_id else "base_link"

        # iterate over ring coordinates
        for i, (x, y, r, color) in enumerate(self.rings):
            # maybe needed a bounds check
            if not (0 <= x < width and 0 <= y < height):
                continue

            # Sample 3D points around the ring edge and use their mean position.
            # Using center pixel is unstable for hollow rings (often background depth).
            radius_px = max(4, int(0.9 * r))
            num_samples = 16
            angles = np.linspace(0.0, 2.0 * np.pi, num_samples, endpoint=False)

            edge_points = []
            for angle in angles:
                sx = int(round(x + radius_px * np.cos(angle)))
                sy = int(round(y + radius_px * np.sin(angle)))
                if 0 <= sx < width and 0 <= sy < height:
                    point = a[sy, sx, :]
                    if not np.isnan(point).any():
                        edge_points.append(point)

            if len(edge_points) < 6:
                continue

            edge_points = np.array(edge_points)
            ring_point = np.median(edge_points, axis=0)

            # Reject unstable geometry where sampled ring points disagree too much.
            if np.std(edge_points[:, 2]) > 0.08:
                continue

            # Transform point from point cloud frame to map
            point_stamped = PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = data.header.stamp
            point_stamped.point.x = float(ring_point[0])
            point_stamped.point.y = float(ring_point[1])
            point_stamped.point.z = float(ring_point[2])

            try:
                # Transform to map frame
                transform = self.tf_buffer.lookup_transform("map", source_frame, data.header.stamp)
                point_map = do_transform_point(point_stamped, transform)

                map_x = point_map.point.x
                map_y = point_map.point.y
                map_z = point_map.point.z

                self.add_to_clusters(map_x, map_y, map_z, color)
            except TransformException:
                self.get_logger().warn("map frame not available yet, skipping ring")

    def add_to_clusters(self, x, y, z, color):
        new_point = np.array([x, y, z])
        now_sec = self.get_clock().now().nanoseconds / 1e9

        for cluster in self.ring_clusters:
            centroid = np.array(cluster["centroid"])
            # Match in XY to reduce depth-noise splitting of the same physical ring
            distance = np.linalg.norm(new_point[:2] - centroid[:2])

            if distance < self.cluster_threshold:
                cluster["count"] += 1
                cluster["last_seen"] = now_sec

                if cluster["count"] == 5 and not cluster.get("spoken", False):
                    name = cluster.get("color", {}).get("name", "unknown")
                    
                    self.say(f"{name} ring")
                    cluster["spoken"] = True

                old = np.array(cluster["centroid"])
                # each new detection nudges the centroid 10% toward it.
                # as cluster grows, the centroid slowly corrects toward the true position
                cluster["centroid"] = ((1 - self.ema_alpha) * old + self.ema_alpha * new_point).tolist()
                
                # update the color similarly to centroid
                if color and cluster.get("color"):
                    old_lab = np.array(cluster["color"]["lab"])
                    new_lab = np.array(color["lab"])
                    smoothed_lab = (1 - self.ema_alpha) * old_lab + self.ema_alpha * new_lab
                    L, A, B = smoothed_lab
                    cluster["color"] = {"lab": smoothed_lab.tolist(), "name": self.classify_lab(L, A, B)}

                return

        # no existing cluster close enough -> new one
        self.ring_clusters.append({
            "centroid": new_point.tolist(),
            "count": 1,
            "spoken": False,
            "color": color
        })

    def publish_clusters(self):
        self.get_logger().info(f"Checking {len(self.ring_clusters)} clusters for publishing.")
        for i, cluster in enumerate(self.ring_clusters):
            if cluster["count"] < self.min_detections:
                continue

            self.get_logger().info(f"Publishing marker for cluster {i} at {cluster['centroid']} with count {cluster['count']}")

            marker = Marker()

            marker.header.frame_id = "map"

            marker.type = 2
            marker.id = i  # unique id per ring

            cx, cy, cz = cluster["centroid"]

            marker.pose.position.x = cx
            marker.pose.position.y = cy
            marker.pose.position.z = cz

            marker.pose.orientation.w = 1.0

            marker.header.stamp = self.get_clock().now().to_msg()

            # Set the scale of the marker
            scale = 0.25  # 15 cm
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            # Set the color
            color = cluster.get("color")
            if color:
                r_val, g_val, b_val = self.lab_to_marker_rgb(color["lab"])
            else:
                r_val, g_val, b_val = 0.0, 1.0, 1.0 

            marker.color.r = r_val
            marker.color.g = g_val
            marker.color.b = b_val
            marker.color.a = 1.0

            self.marker_pub.publish(marker)
    def detect_ring_color(self, cv_image, cx, cy, r):
        h_img, w_img = cv_image.shape[:2]

        x0 = max(0, int(cx - r))
        x1 = min(w_img, int(cx + r) + 1)
        y0 = max(0, int(cy - r))
        y1 = min(h_img, int(cy + r) + 1)

        roi = cv_image[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        img_float = roi.astype(np.float32) / 255.0
        lab_roi = cv2.cvtColor(img_float, cv2.COLOR_BGR2Lab)
        pixels = lab_roi.reshape(-1, 3)

        if pixels.shape[0] < 20:
            return None

        labels = [self.classify_lab(L, A, B) for L, A, B in pixels]
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1


        # so we have a bounding box and we count the colored pixels by color
        # we suspect,. that over 50% of pixel are background
        # we take the second most dominant color, thats not too similar to the background
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        dominant_name, dominant_count = sorted_counts[0]
        dominant_ratio = dominant_count / len(labels)

        class_medians = {}
        for class_name, _ in sorted_counts:
            class_pixels = np.array([pix for pix, label in zip(pixels, labels) if label == class_name])
            if class_pixels.size == 0:
                continue
            class_medians[class_name] = np.median(class_pixels, axis=0)

        selected_name = dominant_name

        # If dominant class is likely background, pick the strongest alternative
        # that is not too close to background color in LAB space.
        if dominant_ratio > 0.5:
            dominant_lab = class_medians.get(dominant_name)
            min_lab_distance = 20.0
            min_secondary_ratio = 0.05

            for class_name, class_count in sorted_counts[1:]:
                candidate_ratio = class_count / len(labels)
                if candidate_ratio < min_secondary_ratio:
                    continue

                candidate_lab = class_medians.get(class_name)
                if dominant_lab is None or candidate_lab is None:
                    continue

                lab_distance = np.linalg.norm(candidate_lab - dominant_lab)
                if lab_distance >= min_lab_distance:
                    selected_name = class_name
                    break

        selected_lab = class_medians.get(selected_name)
        if selected_lab is None:
            return None

        return {
            "lab": selected_lab.tolist(),
            "name": selected_name
        }


    def classify_lab(self, L, A, B):
        # Chroma = distance from the grey axis in the AB plane
        # Low chroma = achromatic (black / grey / white)
        chroma = np.sqrt(A ** 2 + B ** 2)

        if chroma < 15:
            # Achromatic — classify purely by lightness
            if L < 25:      return "black"
            elif L < 70:    return "grey"
            else:           return "white"

        # Chromatic — hue angle in LAB (atan2 of B over A)
        hue = np.degrees(np.arctan2(B, A)) % 360

        if hue < 15 or hue >= 345:     return "red"
        elif hue < 45:                  return "orange"
        elif hue < 75:                  return "yellow"
        elif hue < 150:                 return "green"
        elif hue < 195:                 return "cyan"
        elif hue < 255:                 return "blue"
        elif hue < 285:                 return "purple"
        else:                           return "magenta"


    def lab_to_marker_rgb(self, lab):
        # Convert LAB centroid back to BGR, then return normalised RGB for RViz marker
        lab_pixel = np.array(lab, dtype=np.float32).reshape(1, 1, 3)
        bgr = cv2.cvtColor(lab_pixel, cv2.COLOR_Lab2BGR)
        bgr = np.clip(bgr, 0, 1)
        b, g, r = bgr[0, 0]
        return float(r), float(g), float(b)
            

def main():
    print('Ring detection node starting.')

    rclpy.init(args=None)
    node = detect_rings()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()