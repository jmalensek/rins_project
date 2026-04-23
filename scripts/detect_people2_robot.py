#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

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
		self.n_faces = 10
		self.latest_depth_image = None
		self.depth_frame_id = None
		self.camera_intrinsics = None # To store camera intrinsic parameters

		# NEW SUBS
		# RGB slika
		self.rgb_image_sub = self.create_subscription(Image,"/gemini/color/image_raw",self.rgb_callback, 10)
		# Depth slika
		self.depth_sub = self.create_subscription(Image, "/gemini/depth/image_raw", self.depth_callback, 10)
		# Camera Info
		self.camera_info_sub = self.create_subscription(CameraInfo, "/gemini/color/camera_info", self.camera_info_callback, qos_profile_sensor_data)
		
		self.model = YOLO("yolo26n-face.pt") # for face detection
		self.faces = []
		self.latest_depth = None
		self.depth_frame_id = None

		self.pose_pub = self.create_publisher(PoseStamped, faces_topic, 10)
		self.marker_pub = self.create_publisher(Marker, "/face_markers", 10)
		self.get_logger().info(f"Will publish face coordinates to {faces_topic}.")
		self.get_logger().info("Will publish person markers to /face_markers.")
		self.finished_pub = self.create_publisher(Bool, "/finished", 10)
		self.get_logger().info("Will publish finished trigger to /finished.")



	# detects faces
	def rgb_callback(self, data):

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
			(h, w) = cv_image.shape[:2]

			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, device=self.device)

			# iterate over results
			for result in res:
				if result.boxes is None or len(result.boxes) == 0:
					continue

				for box in result.boxes:
					bbox = box.xyxy[0]

					conf = float(box.conf)
					if conf < self.face_t:
						continue

					self.get_logger().info("Person has been detected!")

					cx = int((bbox[0] + bbox[2]) / 2)
					cy = int((bbox[1] + bbox[3]) / 2)

					self.process_detection(cx, cy)

					# draw rectangle
					cv_image = cv2.rectangle(cv_image,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),self.detection_color,3)
					# draw the center of bounding box
					cv_image = cv2.circle(cv_image, (cx, cy), 5, self.detection_color, -1)

			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27:
				print("exiting")
				exit()
			
		except CvBridgeError as e:
			print(e)

	
	def camera_info_callback(self, msg):
		# Store camera intrinsic parameters and stop subscribing
		if self.camera_intrinsics is None:
			self.camera_intrinsics = {
				'fx': msg.k[0],
				'fy': msg.k[4],
				'cx': msg.k[2],
				'cy': msg.k[5]
			}
			self.get_logger().info(f"Camera intrinsics received: {self.camera_intrinsics}")
			self.destroy_subscription(self.camera_info_sub) # We only need this once

	def depth_callback(self, msg):
		try:
			self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
			self.depth_frame_id = msg.header.frame_id
		except CvBridgeError as e:
			self.get_logger().error(f"Could not convert depth image: {e}")


	def process_detection(self, cx, cy):
		if self.latest_depth_image is None:
			self.get_logger().warn("No depth image received yet.")
			return
		
		if self.camera_intrinsics is None:
			self.get_logger().warn("Camera intrinsics not available yet.")
			return

		h, w = self.latest_depth_image.shape
		if not (0 <= cy < h and 0 <= cx < w):
			self.get_logger().warn(f"Center of detection ({cx}, {cy}) is outside the depth image dimensions ({w}x{h}).")
			return

		# Get distance from the depth image
		distance = self.latest_depth_image[cy, cx]

		if not np.isfinite(distance) or distance <= 0.1:
			self.get_logger().warn("Invalid distance from depth image.")
			return

		# Use camera intrinsics to project 2D pixel to 3D point
		cam_fx = self.camera_intrinsics['fx']
		cam_fy = self.camera_intrinsics['fy']
		cam_cx = self.camera_intrinsics['cx']
		cam_cy = self.camera_intrinsics['cy']

		# Convert pixel coordinates to camera coordinates
		point_camera = PointStamped()
		point_camera.header.frame_id = self.depth_frame_id
		point_camera.header.stamp = self.get_clock().now().to_msg()
		
		# Unprojection formulas
		point_camera.point.x = (cx - cam_cx) * distance / cam_fx
		point_camera.point.y = (cy - cam_cy) * distance / cam_fy
		point_camera.point.z = distance

		# Transform the point to the map frame
		try:
			transform = self.tf_buffer.lookup_transform('map', self.depth_frame_id, rclpy.time.Time())
			point_map = do_transform_point(point_camera, transform)

			self.get_logger().info(f"Face at map coordinates: ({point_map.point.x:.2f}, {point_map.point.y:.2f})")
			self.match_person(point_map.point.x, point_map.point.y, point_map.point.z)

		except TransformException as ex:
			self.get_logger().warn(f"Could not transform point from {self.depth_frame_id} to map: {ex}")



	def match_person(self, x, y, z):
		if z > 0.5:
			self.get_logger().info("Too high!")
			return  # not on a fence!

		n = len(self.detections)
		person_id = n  # default: assume new person

		for p_id in range(n):  # try to match with existing person
			person_detections = np.array(self.detections[p_id], dtype=float)
			xp, yp, zp = np.mean(person_detections, axis=0)

			d = np.linalg.norm([x-xp, y-yp, z-zp])
			if d < 0.6:
				person_id = p_id  # found a match
				self.get_logger().info(f"Matched person {person_id}")
				break

		# save
		is_new_person = person_id >= len(self.detections)
		if person_id >= len(self.detections):
			self.detections.append([])
			self.get_logger().info(f"New person found! (ID {person_id})")
		self.detections[person_id].append([x, y, z])

		if is_new_person:
			self.publish_person_marker(person_id, x, y, z)

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
		if self.coords_published: return
		if len(self.detections) < self.self.n_faces:
			return
		if any(len(self.detections[person_id]) < 2 for person_id in range(self.self.n_faces)):
			return
		self.coords_published = True

		for person_id in range(self.self.n_faces):
			person_detections = np.array(self.detections[self.self.n_faces - person_id], dtype=float)  # visit in reverse order
			x, y, z = np.mean(person_detections, axis=0)

			msg = PoseStamped()
			msg.header.frame_id = "map"
			msg.header.stamp = self.get_clock().now().to_msg()

			msg.pose.position.x = x
			msg.pose.position.y = y
			msg.pose.position.z = z
			msg.pose.orientation.w = 1.0

			self.pose_pub.publish(msg)
			self.get_logger().info(f"Published person {person_id}")


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
