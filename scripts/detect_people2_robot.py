#!/usr/bin/env python3

"""
shrani koordinate obrazov
se uporablja z greet_people.py, saj nato izračuna potrebne pozicije obiskov
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import rclpy.duration

from sensor_msgs.msg import Image, CameraInfo

from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from ultralytics import YOLO

from tf2_ros import TransformListener, Buffer, TransformException
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool

# Message synchronization for temporal alignment of RGB and depth
import message_filters


class detect_faces(Node):

	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		faces_topic = "/face_detections"

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()

		# TF2 listener for transform base_link -> map
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		# df for my detections
		#self.detections_df = pd.DataFrame(columns=['x', 'y', 'z', 'person_id'])
		self.detections = []    # [person_id][x,y,z]  2D array
		self.coords_published = False
		self.face_t = 0.7
		self.n_faces = 5
		self.camera_intrinsics = None # To store camera intrinsic parameters
		self.depth_units_converted = False  # Flag to log depth unit conversion once

		# RGB subscriber with queue
		rgb_sub = message_filters.Subscriber(self, Image, "/gemini/color/image_raw")
		# Depth subscriber with queue
		depth_sub = message_filters.Subscriber(self, Image, "/gemini/depth/image_raw")
		# Temporal synchronizer with 0.05s slop tolerance and queue size of 10
		self.ts = message_filters.ApproximateTimeSynchronizer(
			[rgb_sub, depth_sub],
			queue_size=10,
			slop=0.05
		)
		self.ts.registerCallback(self.synced_callback)
		
		# Camera Info subscription (one-time, unaffected by synchronization)
		self.camera_info_sub = self.create_subscription(CameraInfo, "/gemini/color/camera_info", self.camera_info_callback, qos_profile_sensor_data)
		
		# Load YOLO model for face detection
		self.model = YOLO("yolo26n-face.pt")

		self.pose_pub = self.create_publisher(PoseStamped, faces_topic, 10)
		self.marker_pub = self.create_publisher(Marker, "/face_markers", 10)
		self.get_logger().info(f"Will publish face coordinates to {faces_topic}.")
		self.get_logger().info("Will publish person markers to /face_markers.")
		self.finished_pub = self.create_publisher(Bool, "/finished", 10)
		self.get_logger().info("Will publish finished trigger to /finished.")



	def camera_info_callback(self, msg):
		# Store camera intrinsic parameters and stop subscribing (one-time only)
		if self.camera_intrinsics is None:
			self.camera_intrinsics = {
				'fx': msg.k[0],
				'fy': msg.k[4],
				'cx': msg.k[2],
				'cy': msg.k[5]
			}
			self.get_logger().info(f"Camera intrinsics received: {self.camera_intrinsics}")
			self.destroy_subscription(self.camera_info_sub) # We only need this once


	def synced_callback(self, rgb_msg, depth_msg):
		# Synchronized callback for temporally-aligned RGB and depth messages.
		# This ensures face detection and depth lookup are from the same instant in time.
		try:
			cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
			(h, w) = cv_image.shape[:2]

			# Convert depth image (32-bit floating point)
			depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
			depth_frame_id = depth_msg.header.frame_id
			depth_timestamp = depth_msg.header.stamp

			# If median depth > 100, assume values are in millimeters
			valid_depths = depth_image[depth_image > 0.1]
			if len(valid_depths) > 0:
				median_depth = np.median(valid_depths)
				if median_depth > 100:
					# Convert from millimeters to meters
					depth_image = depth_image / 1000.0
					self.get_logger().info("Depth image converted from millimeters to meters")
				else:
					self.get_logger().info(f"Depth values appear to be in meters (median: {median_depth:.3f}m)")

			# Run YOLO inference for face detection
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, device=self.device)

			# Process each detected face
			for result in res:
				if result.boxes is None or len(result.boxes) == 0:
					continue

				for box in result.boxes:
					bbox = box.xyxy[0]
					conf = float(box.conf)

					# Skip detections below confidence threshold
					if conf < self.face_t:
						continue

					self.get_logger().info("Person has been detected!")

					# Compute center of bounding box
					cx = int((bbox[0] + bbox[2]) / 2)
					cy = int((bbox[1] + bbox[3]) / 2)

					# Process detection with depth image and timestamp
					self.process_detection(cx, cy, depth_image, depth_frame_id, depth_timestamp)

					# Draw bounding box and center point on image
					cv_image = cv2.rectangle(cv_image,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),self.detection_color,3)
					cv_image = cv2.circle(cv_image, (cx, cy), 5, self.detection_color, -1)

			# Display annotated image
			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27:
				self.get_logger().info("ESC pressed, shutting down")
				exit()
			
		except CvBridgeError as e:
			self.get_logger().error(f"CvBridge error: {e}")
		except Exception as e:
			self.get_logger().error(f"Error in synced_callback: {e}")


	def get_stable_depth(self, depth_image, cx, cy, radius=5):
		# Extract a 5x5 patch around (cx, cy) and compute median of valid pixels.
		# This smooths out noisy or invalid depth measurements.
		# Returns None if fewer than 5 valid pixels are found.
		h, w = depth_image.shape
		
		# Define patch boundaries
		x_min = max(0, cx - radius)
		x_max = min(w, cx + radius + 1)
		y_min = max(0, cy - radius)
		y_max = min(h, cy + radius + 1)
		
		# Extract patch and filter valid depths (> 0.1 and finite)
		patch = depth_image[y_min:y_max, x_min:x_max]
		valid_pixels = patch[(patch > 0.1) & np.isfinite(patch)]
		
		# Need at least 5 valid pixels for robust median
		if len(valid_pixels) < 5:
			self.get_logger().warn(f"Not enough valid depth pixels at ({cx}, {cy}): {len(valid_pixels)}")
			return None
		
		# Return median of valid depths
		return np.median(valid_pixels)


	def process_detection(self, cx, cy, depth_image, depth_frame_id, depth_timestamp):
		# Process a face detection: extract 3D coordinates from depth and project to map frame.

		if self.camera_intrinsics is None:
			self.get_logger().warn("Camera intrinsics not available yet.")
			return

		h, w = depth_image.shape
		if not (0 <= cy < h and 0 <= cx < w):
			self.get_logger().warn(f"Center of detection ({cx}, {cy}) is outside the depth image dimensions ({w}x{h}).")
			return

		# Get stable depth using median of 5x5 patch to reduce noise
		distance = self.get_stable_depth(depth_image, cx, cy, radius=5)
		if distance is None:
			self.get_logger().warn("Could not get stable depth at detection center.")
			return

		if not np.isfinite(distance) or distance <= 0.1 or distance > 2.0:
			self.get_logger().warn(f"Invalid distance from depth image: {distance}")
			return

		self.get_logger().info(f"DEPTH = {distance}")


		# Use camera intrinsics to project 2D pixel to 3D point in camera frame
		cam_fx = self.camera_intrinsics['fx']
		cam_fy = self.camera_intrinsics['fy']
		cam_cx = self.camera_intrinsics['cx']
		cam_cy = self.camera_intrinsics['cy']

		# Create 3D point in camera frame using intrinsics
		point_camera = PointStamped()
		point_camera.header.frame_id = depth_frame_id
		point_camera.header.stamp = depth_timestamp  # Use actual message timestamp, not node time
		
		# Unprojection formulas: convert pixel + depth to 3D camera coordinates
		point_camera.point.x = (cx - cam_cx) * distance / cam_fx
		point_camera.point.y = (cy - cam_cy) * distance / cam_fy
		point_camera.point.z = distance

		# Transform the point from camera frame to map frame
		try:
			# Use the actual message timestamp and add 0.1s timeout for transform lookup
			transform = self.tf_buffer.lookup_transform(
				'map',
				depth_frame_id,
				depth_timestamp,
				timeout=rclpy.duration.Duration(seconds=0.1)
			)
			point_map = do_transform_point(point_camera, transform)

			self.get_logger().info(f"Face at map coordinates: ({point_map.point.x:.2f}, {point_map.point.y:.2f})")
			self.match_person(point_map.point.x, point_map.point.y, point_map.point.z)

		except TransformException as ex:
			self.get_logger().warn(f"Could not transform point from {depth_frame_id} to map: {ex}")



	def match_person(self, x, y, z):
		# Match detected face to an existing person or create new person entry.
		# Use spatial distance (Euclidean) to associate new detections with tracked persons.

		if z > 0.5:
			self.get_logger().info("Detection too high, skipping.")
			return

		n = len(self.detections)
		person_id = n  # Default: assume new person

		self.publish_person_marker(person_id, x, y, z)

		# Try to match with existing persons using median position
		for p_id in range(n):
			person_detections = np.array(self.detections[p_id], dtype=float)
			# Use median instead of mean for more robust position estimation
			xp, yp, zp = np.median(person_detections, axis=0)

			# Euclidean distance in 3D
			d = np.linalg.norm([x-xp, y-yp, z-zp])
			if d < 0.6:
				person_id = p_id  # Found a matching person
				self.get_logger().info(f"Matched person {person_id}")
				break

		# Store detection
		is_new_person = person_id >= len(self.detections)
		if is_new_person:
			self.detections.append([])
			self.get_logger().info(f"New person found! (ID {person_id})")
		self.detections[person_id].append([x, y, z])

		if is_new_person:
			# Publish marker for new person
			self.publish_person_marker(person_id, x, y, z)

		# Check if we have detected enough people to publish results
		if len(self.detections) >= self.n_faces and not self.coords_published:
			self.publish_people()


	def publish_person_marker(self, person_id, x, y, z):
		msg = Marker()
		msg.header.frame_id = "map"
		msg.header.stamp = self.get_clock().now().to_msg()
		msg.ns = "detected_people"
		msg.id = int(person_id)
		msg.type = Marker.SPHERE
		msg.action = Marker.ADD

		msg.pose.position.x = float(x)
		msg.pose.position.y = float(y)
		msg.pose.position.z = float(z)
		msg.pose.orientation.w = 1.0

		msg.scale.x = 0.25
		msg.scale.y = 0.25
		msg.scale.z = 0.25

		msg.color.r = 1.0
		msg.color.g = 0.2
		msg.color.b = 0.1
		msg.color.a = 0.9
		self.marker_pub.publish(msg)
		self.get_logger().info(f"Published marker for person {person_id}")



	def publish_people(self):
		# Publish all detected persons and signal completion.
		# This is called once we have detected the minimum number of faces (self.n_faces).

		if self.coords_published:
			return  # Already published, don't repeat
		
		# Verify we have enough people detected
		if len(self.detections) < self.n_faces:
			return
		
		# Verify each person has at least 2 detections for averaging
		if any(len(self.detections[person_id]) < 2 for person_id in range(self.n_faces)):
			return
		
		self.coords_published = True

		# Publish poses in reverse person order (person 9 first, then 8, etc.)
		for person_id in range(self.n_faces):
			# Correct reverse indexing: access the last person first
			reverse_idx = self.n_faces - 1 - person_id
			person_detections = np.array(self.detections[reverse_idx], dtype=float)
			# Use median instead of mean for robust position estimation (less affected by outliers)
			x, y, z = np.median(person_detections, axis=0)

			# Create PoseStamped message
			msg = PoseStamped()
			msg.header.frame_id = "map"
			msg.header.stamp = self.get_clock().now().to_msg()

			msg.pose.position.x = x
			msg.pose.position.y = y
			msg.pose.position.z = z
			msg.pose.orientation.w = 1.0

			self.pose_pub.publish(msg)
			self.get_logger().info(f"Published person {person_id} at ({x:.2f}, {y:.2f}, {z:.2f})")

		# Signal completion
		self.finished_pub.publish(Bool(data=True))
		self.get_logger().info("Done. Published /finished=True. Shutting down node.")
		rclpy.shutdown()



def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	
	try:
		rclpy.shutdown()
	except Exception as e:
		print(f"Error during shutdown: {e}")
	
if __name__ == '__main__':
	main()
