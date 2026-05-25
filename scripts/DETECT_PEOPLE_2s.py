#!/usr/bin/env python3

import json
import math
import os
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

from ament_index_python.packages import get_package_share_directory

from ultralytics import YOLO

from tf2_ros import TransformListener, Buffer, TransformException
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PoseStamped

from std_msgs.msg import String, Bool
from visualization_msgs.msg import Marker

from recognize_people_2s import PeopleRecognizer
from robot_commander import RobotCommander

MODEL_PATH = "yolo26n-face.pt"



class detect_faces(Node):

	def __init__(self, rc: RobotCommander):
		super().__init__('detect_faces')
		self.rc = rc
		self.package_share_dir = get_package_share_directory("rins_project")

		self.detection_color = (0,0,255)
		self.device = 0
		self.max_visit_distance_m = 2.0
		self.standoff_distance = 0.65
		self.merge_radius_m = 0.7
		self.min_location_samples = 5

		# Map-frame locations already visited (x, y). If a new face is detected within
		# merge_radius_m of any visited point, we skip it (no navigation).
		self.visited = []  # list of (float x, float y)

		self.bridge = CvBridge()

		# cache latest pointcloud as numpy (height, width, 3) in base_link frame
		self.latest_pc = None
		self.latest_pc_header = None
		self.pc_height = None
		self.pc_width = None

		# TF2 listener for transform base_link -> map
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.face_t = 0.7
		self.latest_image = None
		self.faces = []  # list of dicts: {cx, cy, bbox}

		# Expand the drawn bbox so it frames the whole portrait/picture, not only the face.
		# These are multipliers of the detected face bbox size.
		self.draw_pad_left = 1.0
		self.draw_pad_right = 1.0
		self.draw_pad_top = 0.7
		self.draw_pad_bottom = 2.7

		# Lightweight live recognition label (updated at most every N seconds)
		self.live_recognition = True
		self._live_recog_period_s = 0.6
		self._last_live_recog_t = 0.0
		self._live_label = None  # (name, gender) or None

		# tracked locations (map frame):
		# {"x": float, "y": float, "done": bool, "n": int, "sum_x": float, "sum_y": float, "marker_published": bool}
		self.tracks = []

		# Notify other nodes (e.g., QR scanning) when we have arrived at a person
		self.current_person_pub = self.create_publisher(String, "/current_person", 10)
		# Signals when we're actively approaching/handling a person (nav + recognition)
		self.approach_active_pub = self.create_publisher(Bool, "/approach_active", 10)
		# RViz marker for stabilized person locations (map frame)
		self.marker_pub = self.create_publisher(Marker, "/face_markers", 10)

		self._pending_goal = None
		self._pending_track = None
		self._navigating = False
		self._recognizing_track = None
		self._recognition_started_t = 0.0
		self._recognition_timeout_s = 20.0
		self._last_skip_log_t = 0.0

		try:
			db_dir = os.path.join(self.package_share_dir, "embeddings_db")
			self.recognizer = PeopleRecognizer(db_dir)
			self.get_logger().info(f"Loaded recognition DB from: {db_dir}")
		except Exception as e:
			self.recognizer = None
			self.get_logger().error(f"Could not load recognizer DB: {e}")

		# subscribe to image + pointcloud
		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

		model_path = os.path.join(self.package_share_dir, MODEL_PATH)
		self.model = YOLO(model_path)

	def _expand_bbox_for_draw(self, bbox, image_shape):
		x1, y1, x2, y2 = bbox
		h, w = image_shape[:2]
		bw = max(1, int(x2 - x1))
		bh = max(1, int(y2 - y1))
		dx1 = int(self.draw_pad_left * bw)
		dx2 = int(self.draw_pad_right * bw)
		dy1 = int(self.draw_pad_top * bh)
		dy2 = int(self.draw_pad_bottom * bh)
		nx1 = max(0, int(x1) - dx1)
		ny1 = max(0, int(y1) - dy1)
		nx2 = min(w - 1, int(x2) + dx2)
		ny2 = min(h - 1, int(y2) + dy2)
		if nx2 <= nx1 or ny2 <= ny1:
			return (int(x1), int(y1), int(x2), int(y2))
		return (int(nx1), int(ny1), int(nx2), int(ny2))

	def _publish_person_marker(self, person_id: int, x: float, y: float, z: float = 0.0) -> None:
		msg = Marker()
		msg.header.frame_id = "map"
		msg.header.stamp = self.get_clock().now().to_msg()
		msg.ns = "detected_people"
		msg.id = int(person_id)
		msg.type = Marker.SPHERE
		msg.action = Marker.ADD

		msg.pose.position.x = float(x)
		msg.pose.position.y = float(y)
		msg.pose.position.z = float(z) + 0.25
		msg.pose.orientation.w = 1.0

		msg.scale.x = 0.25
		msg.scale.y = 0.25
		msg.scale.z = 0.25

		msg.color.r = 1.0
		msg.color.g = 0.2
		msg.color.b = 0.1
		msg.color.a = 0.9

		self.marker_pub.publish(msg)

	def _is_visited(self, x_map: float, y_map: float) -> bool:
		r2 = float(self.merge_radius_m) * float(self.merge_radius_m)
		for (vx, vy) in self.visited:
			dx = float(x_map) - float(vx)
			dy = float(y_map) - float(vy)
			if (dx * dx + dy * dy) <= r2:
				return True
		return False

	def _remember_visited(self, x_map: float, y_map: float) -> None:
		# Avoid duplicate entries.
		if self._is_visited(x_map, y_map):
			return
		self.visited.append((float(x_map), float(y_map)))


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

					#self.get_logger().info("Person has been detected!")

					x1 = int(bbox[0]); y1 = int(bbox[1]); x2 = int(bbox[2]); y2 = int(bbox[3])
					face_bbox = (x1, y1, x2, y2)
					draw_bbox = self._expand_bbox_for_draw(face_bbox, cv_image.shape)

					# draw expanded rectangle (frames the whole portrait/picture)
					cv_image = cv2.rectangle(
						cv_image,
						(int(draw_bbox[0]), int(draw_bbox[1])),
						(int(draw_bbox[2]), int(draw_bbox[3])),
						self.detection_color,
						3,
					)

					cx = int((bbox[0] + bbox[2]) / 2)
					cy = int((bbox[1] + bbox[3]) / 2)

					# draw the center of bounding box
					cv_image = cv2.circle(cv_image, (cx, cy), 5, self.detection_color, -1)

					self.faces.append({"cx": cx, "cy": cy, "bbox": face_bbox, "draw_bbox": draw_bbox})

			# Optional: run lightweight live recognition (label overlay), throttled.
			if self.live_recognition and self.recognizer is not None and self.faces:
				now = time.monotonic()
				if (now - float(self._last_live_recog_t)) >= float(self._live_recog_period_s):
					self._last_live_recog_t = now
					face = max(self.faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
					crop = self._crop_bbox(self.latest_image, face["bbox"])
					label = None
					if crop is not None:
						try:
							result = self.recognizer.recognize(crop)
						except Exception:
							result = None
						if result is not None and result.get("name"):
							label = (result.get("name"), result.get("gender"))
					self._live_label = label

			# Draw label above each bbox (if available)
			if self._live_label is not None:
				name, gender = self._live_label
				label_text = f"{name} ({gender})" if gender else f"{name}"
				for f in self.faces:
					draw_bbox = f.get("draw_bbox") or f.get("bbox")
					x1, y1, x2, y2 = [int(v) for v in draw_bbox]
					tx = x1
					ty = max(0, y1 - 10)
					cv2.putText(
						cv_image,
						label_text,
						(tx, ty),
						cv2.FONT_HERSHEY_SIMPLEX,
						0.7,
						self.detection_color,
						2,
						cv2.LINE_AA,
					)

			# If we're at a person, try recognition on every RGB frame (no extra waiting)
			if self._recognizing_track is not None and self.recognizer is not None and self.faces:
				track_ix = int(self._recognizing_track)
				if 0 <= track_ix < len(self.tracks) and not self.tracks[track_ix].get("done"):
					if (time.monotonic() - float(self._recognition_started_t)) <= float(self._recognition_timeout_s):
						face = max(self.faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
						crop = self._crop_bbox(self.latest_image, face["bbox"])
						if crop is not None:
							result = self.recognizer.recognize(crop)
							if result is not None and result.get("name"):
								name = result.get("name")
								gender = result.get("gender")
								self._live_label = (name, gender)
								self.tracks[track_ix]["done"] = True
								self._remember_visited(float(self.tracks[track_ix].get("x", 0.0)), float(self.tracks[track_ix].get("y", 0.0)))
								msg = String()
								msg.data = json.dumps({"person": name, "gender": gender})
								self.current_person_pub.publish(msg)
								self.get_logger().info(f"Recognition done. person={name} gender={gender}. Published /current_person.")
								self._recognizing_track = None
								self.approach_active_pub.publish(Bool(data=False))
					else:
						self.get_logger().warn("Recognition timeout (no match).")
						self.tracks[track_ix]["done"] = True
						self._remember_visited(float(self.tracks[track_ix].get("x", 0.0)), float(self.tracks[track_ix].get("y", 0.0)))
						self._recognizing_track = None
						self.approach_active_pub.publish(Bool(data=False))
				
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

	def _add_location_sample(self, track_ix: int, x_map: float, y_map: float):
		t = self.tracks[int(track_ix)]
		# Keep a running average for stability
		n = int(t.get("n", 0))
		sum_x = float(t.get("sum_x", 0.0))
		sum_y = float(t.get("sum_y", 0.0))
		n += 1
		sum_x += float(x_map)
		sum_y += float(y_map)
		t["n"] = n
		t["sum_x"] = sum_x
		t["sum_y"] = sum_y
		t["x"] = sum_x / float(n)
		t["y"] = sum_y / float(n)

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


	def _rotate_vector_by_quaternion(self, v, q):
		"""Rotate 3-vector v (iterable) by geometry_msgs Quaternion q."""
		qx, qy, qz, qw = q.x, q.y, q.z, q.w
		qvec = np.array([qx, qy, qz], dtype=float)
		v = np.array(v, dtype=float)
		t = 2.0 * np.cross(qvec, v)
		vprime = v + qw * t + np.cross(qvec, t)
		return float(vprime[0]), float(vprime[1]), float(vprime[2])

	def _estimate_plane_normal_from_track(self, track_ix: int, window: int = 15, min_points: int = 50):
		"""Estimate plane point and normal in map frame for the track using cached pointcloud.
		Returns ((px,py,pz), (nx,ny,nz)) or None on failure.
		"""
		if self.latest_pc is None:
			return None
		if track_ix < 0 or track_ix >= len(self.tracks):
			return None
		t = self.tracks[int(track_ix)]
		if 'img_cx' not in t or 'img_cy' not in t:
			return None
		cx = int(t['img_cx'])
		cy = int(t['img_cy'])
		h = int(self.pc_height)
		w = int(self.pc_width)
		# bounds
		x0 = max(0, cx - window)
		x1 = min(w - 1, cx + window)
		y0 = max(0, cy - window)
		y1 = min(h - 1, cy + window)
		sub = self.latest_pc[y0:y1+1, x0:x1+1, :].reshape(-1, 3)
		# filter finite
		mask = np.isfinite(sub).all(axis=1)
		pts = sub[mask]
		if pts.shape[0] < int(min_points):
			return None
		# fit plane by SVD
		centroid = np.mean(pts, axis=0)
		A = pts - centroid
		U, S, Vt = np.linalg.svd(A, full_matrices=False)
		normal = Vt[-1, :]
		# create PointStamped in base_link
		ps = PointStamped()
		ps.header.frame_id = 'base_link'
		ps.header.stamp = self.latest_pc_header.stamp if self.latest_pc_header is not None else self.get_clock().now().to_msg()
		ps.point.x = float(centroid[0])
		ps.point.y = float(centroid[1])
		ps.point.z = float(centroid[2])
		# transform point to map
		try:
			transform = self.tf_buffer.lookup_transform('map', 'base_link', ps.header.stamp)
		except TransformException:
			try:
				transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
			except TransformException:
				return None
		point_map = do_transform_point(ps, transform)
		# rotate normal into map frame
		nx, ny, nz = self._rotate_vector_by_quaternion(normal.tolist(), transform.transform.rotation)
		# normalize
		norm = math.sqrt(nx * nx + ny * ny + nz * nz)
		if norm < 1e-6:
			return None
		nx /= norm; ny /= norm; nz /= norm
		return ((point_map.point.x, point_map.point.y, point_map.point.z), (nx, ny, nz))

	def _build_goal_from_plane_nearest_side(self, point_map, normal_map):
		"""Build a standoff goal on the normal side that is closest to the robot.
		Returns (goal_x, goal_y, goal_yaw).
		"""
		if getattr(self.rc, "current_pose", None) is None:
			return None
		r_x = float(self.rc.current_pose.pose.position.x)
		r_y = float(self.rc.current_pose.pose.position.y)
		stop = max(0.0, float(self.standoff_distance))
		px = float(point_map[0])
		py = float(point_map[1])
		nx = float(normal_map[0])
		ny = float(normal_map[1])

		# Two possible standoff points: one on each side of the wall normal.
		g1x = px - nx * stop
		g1y = py - ny * stop
		g2x = px + nx * stop
		g2y = py + ny * stop

		d1 = (g1x - r_x) * (g1x - r_x) + (g1y - r_y) * (g1y - r_y)
		d2 = (g2x - r_x) * (g2x - r_x) + (g2y - r_y) * (g2y - r_y)

		# Always choose side closer to robot to avoid navigating around the wall.
		if d1 <= d2:
			goal_x, goal_y = g1x, g1y
		else:
			goal_x, goal_y = g2x, g2y

		# Face the image plane (toward wall point).
		goal_yaw = math.atan2(py - goal_y, px - goal_x)
		return goal_x, goal_y, goal_yaw


	def _queue_navigation(self, track_ix: int, x_map: float, y_map: float):
		if self._pending_goal is not None or self._navigating:
			return
		if self._recognizing_track is not None:
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

		# Try to compute plane normal from recent pointcloud around the track's image
		goal = None
		try:
			plane_result = self._estimate_plane_normal_from_track(track_ix)
			if plane_result is not None:
				(point_map, normal_map) = plane_result
				selected = self._build_goal_from_plane_nearest_side(point_map, normal_map)
				if selected is not None:
					goal_x, goal_y, yaw = selected
					goal = PoseStamped()
					goal.header.frame_id = "map"
					goal.header.stamp = self.get_clock().now().to_msg()
					goal.pose.position.x = float(goal_x)
					goal.pose.position.y = float(goal_y)
					goal.pose.position.z = 0.0
					goal.pose.orientation = self.rc.YawToQuaternion(float(yaw))
		except Exception:
			goal = None
		# fallback: stop_standoff along current approach direction
		if goal is None:
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
				# Start recognition phase only after arrival
				self._recognizing_track = self._pending_track
				self._recognition_started_t = time.monotonic()
				# As soon as we arrive at a person, remember the location so we don't revisit.
				try:
					ix = int(self._recognizing_track)
					if 0 <= ix < len(self.tracks):
						self._remember_visited(float(self.tracks[ix].get("x", 0.0)), float(self.tracks[ix].get("y", 0.0)))
				except Exception:
					pass
				self.get_logger().info(f"Arrived at track#{self._recognizing_track}. Starting recognition...")
				self._navigating = False
				self._pending_goal = None
				self._pending_track = None
			return

		# start navigation
		self._navigating = True
		self.get_logger().info(f"Going to track#{self._pending_track}...")
		self.approach_active_pub.publish(Bool(data=True))
		self.rc.goToPose(self._pending_goal)

	def pointcloud_callback(self, data):
		if self.latest_image is None or self.recognizer is None:
			return
		if not self.faces:
			return
		if self._recognizing_track is not None:
			return

		# get point cloud attributes
		height = data.height
		width = data.width	

		# get 3-channel representation of the point cloud in numpy format (once)
		a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
		a = a.reshape((height, width, 3))
		# cache latest pointcloud
		self.latest_pc = a
		self.latest_pc_header = data.header
		self.pc_height = height
		self.pc_width = width

		# lookup TF once per callback: base_link -> map (same as detect_people2.py)
		try:
			transform = self.tf_buffer.lookup_transform("map", "base_link", data.header.stamp)
		except TransformException:
			try:
				transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
			except TransformException:
				return

		# iterate over face coordinates
		for face in self.faces:
			x = int(face["cx"])
			y = int(face["cy"])
			bbox = face["bbox"]
			if x < 0 or x >= width or y < 0 or y >= height:
				continue

			crop = self._crop_bbox(self.latest_image, bbox)
			if crop is None:
				continue

			# depth lookup + TF to map (needed for location tracking)
			d = a[y, x, :]
			if not np.isfinite(d[0]) or not np.isfinite(d[1]) or not np.isfinite(d[2]):
				continue
			dist = float(np.linalg.norm(d))
			if dist > float(self.max_visit_distance_m):
				continue
			point_stamped = PointStamped()
			point_stamped.header.frame_id = "base_link"
			point_stamped.header.stamp = data.header.stamp
			point_stamped.point.x = float(d[0])
			point_stamped.point.y = float(d[1])
			point_stamped.point.z = float(d[2])

			point_map = do_transform_point(point_stamped, transform)
			x1 = float(point_map.point.x)
			y1 = float(point_map.point.y)

			# If we have already visited a person at this location, skip.
			if self._is_visited(x1, y1):
				# Throttle log spam
				now = time.monotonic()
				if (now - float(self._last_skip_log_t)) > 2.0:
					#self.get_logger().info(f"Skipping face near visited location ({x1:.2f}, {y1:.2f})")
					self._last_skip_log_t = now
				continue

			track_ix = self._find_track_ix(x1, y1)
			if track_ix is not None:
				# Always refine the averaged location for an existing track.
				self._add_location_sample(track_ix, x1, y1)
				# update image coordinates for this track (used for plane estimation)
				try:
					self.tracks[track_ix]["img_cx"] = int(face["cx"])
					self.tracks[track_ix]["img_cy"] = int(face["cy"])
					self.tracks[track_ix]["pc_stamp"] = data.header.stamp
				except Exception:
					pass
				if self.tracks[track_ix].get("done"):
					# Ensure the visited list stays aligned with the refined track.
					self._remember_visited(float(self.tracks[track_ix].get("x", 0.0)), float(self.tracks[track_ix].get("y", 0.0)))
					continue
			else:
				self.tracks.append({"x": x1, "y": y1, "done": False, "n": 0, "sum_x": 0.0, "sum_y": 0.0, "marker_published": False, "img_cx": int(face["cx"]), "img_cy": int(face["cy"]), "pc_stamp": data.header.stamp})
				track_ix = len(self.tracks) - 1
				self._add_location_sample(track_ix, x1, y1)

			# if this track isn't done yet, make sure we go to it (but never spam the same goal)
			if not self.tracks[track_ix].get("done"):
				if int(self.tracks[track_ix].get("n", 0)) >= int(self.min_location_samples):
					# Stabilized mean (running average) is now available: publish marker once, then approach.
					if not bool(self.tracks[track_ix].get("marker_published", False)):
						self._publish_person_marker(int(track_ix), float(self.tracks[track_ix]["x"]), float(self.tracks[track_ix]["y"]), 0.0)
						self.tracks[track_ix]["marker_published"] = True
						self.get_logger().info(
							f"Track#{track_ix} stabilized (n={int(self.tracks[track_ix].get('n', 0))}). Published marker + queuing approach.")
					# Use averaged location for the goal
					self._queue_navigation(track_ix, float(self.tracks[track_ix]["x"]), float(self.tracks[track_ix]["y"]))

			# recognition happens only after arrival (see process_recognition)




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
