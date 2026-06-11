#!/usr/bin/env python3

"""
shrani koordinate robota ob zaznavi obraza
se uporablja z greet_people1_robot.py, obišče točno te koordinate
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import rclpy.duration

from sensor_msgs.msg import Image, CameraInfo, CompressedImage

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
from std_msgs.msg import String

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

		# za robot speaking
		self.speak_pub = self.create_publisher(String, "/speak", 10)

		faces_topic = "/face_detections"

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()

		# TF2 listener for transform base_link -> map
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		# Face detections storage - hybrid approach
		# Each person stores two coordinate types:
		# - 'B': face positions in map frame (detected face location)
		# - 'A': robot positions when detection occurred (if close enough)
		self.detections = []  # [person_id] = {'B': [...], 'A': [...]}
		self.coords_published = False
		self.face_t = 0.7
		self.n_faces = 5
		self.face_radius = 0.3
		self.detect_distance = 0.85  # Maximum distance to record robot position, in meters
		self.camera_intrinsics = None # To store camera intrinsic parameters
		self.depth_units_converted = False  # Flag to log depth unit conversion once

		# RGB subscriber with queue
		rgb_sub = message_filters.Subscriber(self, CompressedImage, "/gemini/color/image_raw/compressed")
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


	def get_robot_position(self, timestamp):
		# Get the robot's position (base_link) in the map frame at a given timestamp.
		# Returns [x, y, z] or None if transform fails.
		try:
			transform = self.tf_buffer.lookup_transform(
				'map',
				'base_link',
				timestamp,
				timeout=rclpy.duration.Duration(seconds=0.1)
			)
			# Extract translation and rotation (quaternion)
			x = transform.transform.translation.x
			y = transform.transform.translation.y
			z = transform.transform.translation.z
			rx = transform.transform.rotation.x
			ry = transform.transform.rotation.y
			rz = transform.transform.rotation.z
			rw = transform.transform.rotation.w
			# Return full pose (position + quaternion)
			return [x, y, z, rx, ry, rz, rw]
		except TransformException as ex:
			self.get_logger().warn(f"Could not get robot position: {ex}")
			return None


	def synced_callback(self, rgb_msg, depth_msg):
		# Synchronized callback for temporally-aligned RGB and depth messages.
		# This ensures face detection and depth lookup are from the same instant in time.
		try:
			# 1. Pretvori podatke iz CompressedImage v numpy polje
			np_arr = np.frombuffer(rgb_msg.data, np.uint8)
			
			# 2. Dekompresiraj sliko neposredno v BGR format (OpenCV)
			cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
			
			if cv_image is None:
				self.get_logger().error("Dekompresija slike ni uspela.")
				return

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
		# Process face detection: extract 3D face coordinates (B) and robot position (A).
		# If face distance <= self.detect_distance, also record robot's position in map frame.

		if self.camera_intrinsics is None:
			self.get_logger().warn("Camera intrinsics not available yet.")
			return

		h, w = depth_image.shape
		if not (0 <= cy < h and 0 <= cx < w):
			self.get_logger().warn(f"Center of detection ({cx}, {cy}) is outside depth dimensions ({w}x{h}).")
			return

		# Get stable depth using median of 5x5 patch to reduce noise
		distance = self.get_stable_depth(depth_image, cx, cy, radius=5)
		if distance is None:
			self.get_logger().warn("Could not get stable depth at detection center.")
			return

		if not np.isfinite(distance) or distance <= 0.1 or distance > 1.5:
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
		point_camera.header.stamp = depth_timestamp
		
		# Unprojection formulas: convert pixel + depth to 3D camera coordinates
		point_camera.point.x = (cx - cam_cx) * distance / cam_fx
		point_camera.point.y = (cy - cam_cy) * distance / cam_fy
		point_camera.point.z = distance

		# Transform face position from camera frame to map frame (Coordinate B)
		try:
			# Get transform for face detection
			transform = self.tf_buffer.lookup_transform(
				'map',
				depth_frame_id,
				depth_timestamp,
				timeout=rclpy.duration.Duration(seconds=0.1)
			)
			point_map_B = do_transform_point(point_camera, transform)
			
			
			if distance <= self.detect_distance:
				# Get robot position (Coordinate A) at detection time
				robot_pos = self.get_robot_position(depth_timestamp)
				if robot_pos is not None:
					self.get_logger().info(f"Face detected at map coords B: ({point_map_B.point.x:.2f}, {point_map_B.point.y:.2f}), depth: {distance:.2f}m")
					# Pass both coordinates (B and full A pose) to matching function
					self.match_person(
						point_map_B.point.x, point_map_B.point.y, point_map_B.point.z,
						robot_pos[0], robot_pos[1], robot_pos[2], robot_pos[3], robot_pos[4], robot_pos[5], robot_pos[6]
					)
				else:
					self.get_logger().warn("Could not get robot position for recording.")
			else:
				self.get_logger().info(f"Face too far ({distance:.2f}m > {self.detect_distance}m), skipping.")

		except TransformException as ex:
			self.get_logger().warn(f"Could not transform face point: {ex}")



	def match_person(self, bx, by, bz, ax, ay, az, aqx, aqy, aqz, aqw):
		# Match detected face (B: face_x, face_y, face_z) to existing person.
		# Also store robot position (A: robot_x, robot_y, robot_z).
		# Detection storage format: {'B': [...], 'A': [...]}

		if bz > 0.5:
			self.get_logger().info("Detection too high, skipping.")
			return

		n = len(self.detections)
		person_id = n  # Default: assume new person

		self.publish_person_marker(person_id, bx, by, bz)

		# Try to match with existing persons using median of B coordinates
		for p_id in range(n):
			if len(self.detections[p_id]['B']) == 0:
				continue
			
			person_b_detections = np.array(self.detections[p_id]['B'], dtype=float)
			# Use median of previous face detections for this person
			bx_prev, by_prev, bz_prev = np.median(person_b_detections, axis=0)

			# Euclidean distance in 3D
			d = np.linalg.norm([bx - bx_prev, by - by_prev, bz - bz_prev])
			if d < self.face_radius:
				person_id = p_id  # Found a matching person
				self.get_logger().info(f"Matched person {person_id}")
				break

		# Create new person entry if needed
		is_new_person = person_id >= len(self.detections)
		if is_new_person:
			self.detections.append({'B': [], 'A': []})
			self.get_logger().info(f"New person found! (ID {person_id})")
			self.speak(f"I found a new person!")

		# Store both coordinates: store full A pose including quaternion
		self.detections[person_id]['B'].append([bx, by, bz])  # Face position
		self.detections[person_id]['A'].append([ax, ay, az, aqx, aqy, aqz, aqw])  # Robot pose (A)

		if is_new_person:
			# Publish marker for new person at face location B
			self.publish_person_marker(person_id, bx, by, bz)

		# Check if we have detected enough people to publish results
		if len(self.detections) >= self.n_faces and not self.coords_published:
			self.publish_people()

	def speak(self, text):
		msg = String()
		msg.data = text
		self.speak_pub.publish(msg)
		self.get_logger().info(f'Said: "{text}"')


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
		# Publishes median of face coordinates B on faces_topic.

		if self.coords_published:
			return  # Already published, don't repeat
		
		# Verify we have enough people detected
		if len(self.detections) < self.n_faces:
			return
		
		# Verify each person has at least 2 robot-pose (A) detections for averaging
		if any(len(self.detections[person_id]['A']) < 2 for person_id in range(self.n_faces)):
			return
		
		self.coords_published = True

		# Publish poses in reverse person order (person 9 first, then 8, etc.)
		# Using median of face coordinates B
		for person_id in range(self.n_faces):
			# Correct reverse indexing: access the last person first
			reverse_idx = self.n_faces - 1 - person_id
			
			# Extract robot-pose A detections and compute median for position + orientation
			person_a_detections = np.array(self.detections[reverse_idx]['A'], dtype=float)
			median_a = np.median(person_a_detections, axis=0)
			ax, ay, az = median_a[0], median_a[1], median_a[2]
			aqx, aqy, aqz, aqw = median_a[3], median_a[4], median_a[5], median_a[6]

			# Normalize quaternion
			q_norm = np.linalg.norm([aqx, aqy, aqz, aqw])
			if q_norm > 0:
				aqx, aqy, aqz, aqw = aqx / q_norm, aqy / q_norm, aqz / q_norm, aqw / q_norm

			# Create PoseStamped message with median robot pose A (position+orientation)
			msg = PoseStamped()
			msg.header.frame_id = "map"
			msg.header.stamp = self.get_clock().now().to_msg()

			msg.pose.position.x = float(ax)
			msg.pose.position.y = float(ay)
			msg.pose.position.z = float(az)
			msg.pose.orientation.x = float(aqx)
			msg.pose.orientation.y = float(aqy)
			msg.pose.orientation.z = float(aqz)
			msg.pose.orientation.w = float(aqw)

			self.pose_pub.publish(msg)

			# Also log median face position B for reference
			person_b_detections = np.array(self.detections[reverse_idx]['B'], dtype=float)
			bx_med, by_med, bz_med = np.median(person_b_detections, axis=0)
			self.get_logger().info(
				f"Published person {person_id}: Goal A at ({ax:.2f}, {ay:.2f}, {az:.2f}, quat=[{aqx:.3f},{aqy:.3f},{aqz:.3f},{aqw:.3f}]), "
				f"Face B median at ({bx_med:.2f}, {by_med:.2f}, {bz_med:.2f})"
			)

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
