#!/usr/bin/env python3

import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

from ultralytics import YOLO

from tf2_ros import TransformListener, Buffer, TransformException
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PoseStamped

from std_msgs.msg import String

from RECOGNIZE_PEOPLE_2s import PeopleRecognizer
from robot_commander import RobotCommander

MODEL_PATH = "yolo26n-face.pt"



class detect_faces(Node):

	def __init__(self, rc: RobotCommander):
		super().__init__('detect_faces')
		self.rc = rc

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value
		self.standoff_distance = 0.65
		self.min_attempt_interval_s = 0.5
		self.merge_radius_m = 0.5

		self.bridge = CvBridge()

		# TF2 listener for transform base_link -> map
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.face_t = 0.7
		self.latest_image = None
		self.faces = []  # list of dicts: {cx, cy, bbox}

		# tracked locations (map frame):
		# {"x": float, "y": float, "samples": [(name, gender), ...], "done": bool}
		self.tracks = []

		# Notify other nodes (e.g., QR scanning) when we have arrived at a person
		self.current_person_pub = self.create_publisher(String, "/current_person", 10)

		self._last_attempt_time = 0.0
		self._pending_goal = None
		self._pending_track = None
		self._navigating = False

		try:
			self.recognizer = PeopleRecognizer("./embeddings_db")
			self.get_logger().info("Loaded recognition DB from: ./embeddings_db")
		except Exception as e:
			self.recognizer = None
			self.get_logger().error(f"Could not load recognizer DB from ./embeddings_db: {e}")

		# subscribe to image + pointcloud
		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

		self.model = YOLO(MODEL_PATH)


	# detects faces
	def rgb_callback(self, data):
		self.faces = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
			self.latest_image = cv_image

			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, device=self.device)

			# iterate over results
			for result in res:
				if result.boxes is None or len(result.boxes) == 0:
					continue

				for box in result.boxes:
					bbox = box.xyxy[0]

					x = float(box.conf)
					if x < self.face_t:
						continue

					self.get_logger().info("Person has been detected!")

					# draw rectangle
					cv_image = cv2.rectangle(cv_image,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),self.detection_color,3)

					cx = int((bbox[0] + bbox[2]) / 2)
					cy = int((bbox[1] + bbox[3]) / 2)

					# draw the center of bounding box
					cv_image = cv2.circle(cv_image, (cx, cy), 5, self.detection_color, -1)

					x1 = int(bbox[0]); y1 = int(bbox[1]); x2 = int(bbox[2]); y2 = int(bbox[3])
					self.faces.append({"cx": cx, "cy": cy, "bbox": (x1, y1, x2, y2)})
				
			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key==27:
				self.get_logger().info("Exiting (ESC pressed)")
				rclpy.shutdown()
				return

			
		except CvBridgeError as e:
			print(e)


	def _find_track_ix(self, x_map: float, y_map: float):
		for i, t in enumerate(self.tracks):
			dx = float(t["x"]) - float(x_map)
			dy = float(t["y"]) - float(y_map)
			if (dx * dx + dy * dy) <= (self.merge_radius_m * self.merge_radius_m):
				return i
		return None

	def _majority_name_gender(self, samples):
		counts = {}
		for name, gender in samples:
			if not name:
				continue
			key = (name, gender or "?")
			counts[key] = counts.get(key, 0) + 1
		if not counts:
			return None, "?"
		(name, gender), _ = max(counts.items(), key=lambda kv: kv[1])
		return name, gender


	def _crop_bbox(self, image: np.ndarray, bbox):
		x1, y1, x2, y2 = bbox
		h, w = image.shape[:2]
		x1 = max(0, min(w - 1, x1))
		x2 = max(0, min(w, x2))
		y1 = max(0, min(h - 1, y1))
		y2 = max(0, min(h, y2))
		if x2 <= x1 or y2 <= y1:
			return None
		return image[y1:y2, x1:x2].copy()


	def _queue_navigation(self, track_ix: int, x_map: float, y_map: float):
		if self._pending_goal is not None or self._navigating:
			return
		if track_ix is None:
			return
		if not getattr(self.rc, "initial_pose_received", True) or getattr(self.rc, "current_pose", None) is None:
			self.get_logger().warn("Robot pose not available yet (waiting for amcl_pose)")
			return

		rx = float(self.rc.current_pose.pose.position.x)
		ry = float(self.rc.current_pose.pose.position.y)

		dx = x_map - rx
		dy = y_map - ry
		dist = math.sqrt(dx * dx + dy * dy)
		if dist > 1e-6:
			ux = dx / dist
			uy = dy / dist
		else:
			ux, uy = 0.0, 0.0

		# stop standoff_distance before the target
		stop = max(0.0, float(self.standoff_distance))
		goal_x = x_map - ux * stop
		goal_y = y_map - uy * stop
		yaw = math.atan2(dy, dx)

		goal = PoseStamped()
		goal.header.frame_id = "map"
		goal.header.stamp = self.get_clock().now().to_msg()
		goal.pose.position.x = float(goal_x)
		goal.pose.position.y = float(goal_y)
		goal.pose.position.z = 0.0
		goal.pose.orientation = self.rc.YawToQuaternion(float(yaw))

		self._pending_goal = goal
		self._pending_track = int(track_ix)
		self.get_logger().info(f"Queued navigation to track#{track_ix} at ({goal_x:.2f}, {goal_y:.2f})")


	def process_pending(self):
		if self._pending_goal is None:
			return
		if self._navigating:
			if self.rc.isTaskComplete():
				track_ix = self._pending_track
				name = None
				gender = "?"
				if track_ix is not None and 0 <= int(track_ix) < len(self.tracks):
					track = self.tracks[int(track_ix)]
					name, gender = self._majority_name_gender(track.get("samples", []))
					track["done"] = True
				if name:
					msg = String()
					msg.data = json.dumps({"person": name, "gender": gender})
					self.current_person_pub.publish(msg)
					self.get_logger().info(f"Arrived. Majority person={name} gender={gender}. Published /current_person.")
				else:
					self.get_logger().warn("Arrived, but no recognition samples to publish.")
				self._navigating = False
				self._pending_goal = None
				self._pending_track = None
			return

		# start navigation
		self._navigating = True
		self.get_logger().info(f"Going to track#{self._pending_track}...")
		self.rc.goToPose(self._pending_goal)


	def pointcloud_callback(self, data):
		if self.latest_image is None or self.recognizer is None:
			return
		if not self.faces:
			return

		# get point cloud attributes
		height = data.height
		width = data.width	

		# get 3-channel representation of the point cloud in numpy format (once)
		a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
		a = a.reshape((height, width, 3))


		# iterate over face coordinates
		for face in self.faces:
			x = int(face["cx"])
			y = int(face["cy"])
			bbox = face["bbox"]
			if x < 0 or x >= width or y < 0 or y >= height:
				continue

			now = time.monotonic()
			if (now - self._last_attempt_time) < self.min_attempt_interval_s:
				continue
			self._last_attempt_time = now

			crop = self._crop_bbox(self.latest_image, bbox)
			if crop is None:
				continue

			# depth lookup + TF to map (needed for location tracking)
			d = a[y, x, :]
			if not np.isfinite(d[0]) or not np.isfinite(d[1]) or not np.isfinite(d[2]):
				continue
			point_stamped = PointStamped()
			point_stamped.header.frame_id = data.header.frame_id
			point_stamped.header.stamp = data.header.stamp
			point_stamped.point.x = float(d[0])
			point_stamped.point.y = float(d[1])
			point_stamped.point.z = float(d[2])

			try:
				transform = self.tf_buffer.lookup_transform("map", data.header.frame_id, data.header.stamp)
			except TransformException:
				return

			point_map = do_transform_point(point_stamped, transform)
			x1 = float(point_map.point.x)
			y1 = float(point_map.point.y)

			track_ix = self._find_track_ix(x1, y1)
			if track_ix is not None and self.tracks[track_ix].get("done"):
				continue
			if track_ix is None:
				self.tracks.append({"x": x1, "y": y1, "samples": [], "done": False})
				track_ix = len(self.tracks) - 1
				self._queue_navigation(track_ix, x1, y1)

			result = self.recognizer.recognize(crop)
			if result is None:
				continue
			name = result.get("name")
			gender = result.get("gender")
			if not name:
				continue
			self.tracks[track_ix]["samples"].append((name, gender))




def main():
	print('Face detection + recognition + navigation node starting.')

	rclpy.init(args=None)
	rc = RobotCommander()

	# bring up Nav2 if available
	try:
		rc.waitUntilNav2Active()
	except Exception:
		pass

	# optional undock (if supported)
	try:
		while getattr(rc, "is_docked", None) is None:
			rclpy.spin_once(rc, timeout_sec=0.5)
		if getattr(rc, "is_docked", False):
			rc.undock()
	except Exception:
		pass

	node = detect_faces(rc)

	while rclpy.ok():
		rclpy.spin_once(node, timeout_sec=0.1)
		rclpy.spin_once(rc, timeout_sec=0.1)
		node.process_pending()

	node.destroy_node()
	try:
		rc.destroyNode()
	except Exception:
		pass
	try:
		rclpy.shutdown()
	except Exception as e:
		print(f"Error during shutdown: {e}")
	
if __name__ == '__main__':
	main()
