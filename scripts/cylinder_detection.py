#!/usr/bin/env python3

'''
Has to detect barrels -> count the number of barrels and the colors of barrels.

Finish the node when leaving the first room - at the same time send the barrel report 
to the node, that will make a final report

So this node should run passively

workflow:
1. detect a barrel and its color
    check this barrel was already detected
2. detect, whether the barrel is horizontal or vertical
    if it is horizontal, check whether it is leaking (alert)
    (leaking can be checked with the comparison of floor colors)
3. keep a count of the barrels and their color and orientation
4. for each barrel put a marker of the right color, so if we detect it again, we dont count it again
'''

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import subprocess

from tf2_ros import TransformListener, Buffer, TransformException
from tf2_geometry_msgs import do_transform_point

from std_msgs.msg import Bool, String

try:
    import pcl
    HAS_PCL = True
except ImportError:
    HAS_PCL = False

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False


class detect_barrels(Node):

    FLOOR_LAB_REF = np.array([75.0, 0.0, 5.0])

    def __init__(self):
        super().__init__('detect_barrels')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('device', ''),

                ('fx', 277.0),
                ('fy', 277.0),
                ('cx', 160.0),
                ('cy', 120.0),

                ('ransac_distance_threshold', 0.025),
                ('ransac_max_iterations', 500),

                ('cylinder_min_radius', 0.10),
                ('cylinder_max_radius', 0.45),

                ('cluster_threshold', 0.20),
                ('min_detections', 5),

                ('leak_lab_distance_threshold', 18.0),
                ('leak_sample_radius_px', 30),
            ]
        )

        marker_topic = "/barrels_marker"

        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.fx = self.get_parameter('fx').get_parameter_value().double_value

        self.fy = self.get_parameter('fy').get_parameter_value().double_value

        self.cx = self.get_parameter('cx').get_parameter_value().double_value

        self.cy = self.get_parameter('cy').get_parameter_value().double_value

        self.ransac_dist = self.get_parameter('ransac_distance_threshold').get_parameter_value().double_value

        self.ransac_iters = self.get_parameter('ransac_max_iterations').get_parameter_value().integer_value

        self.cyl_r_min = self.get_parameter('cylinder_min_radius').get_parameter_value().double_value

        self.cyl_r_max = self.get_parameter('cylinder_max_radius').get_parameter_value().double_value

        self.cluster_threshold = self.get_parameter('cluster_threshold').get_parameter_value().double_value

        self.min_detections = self.get_parameter('min_detections').get_parameter_value().integer_value

        self.leak_lab_thresh = self.get_parameter('leak_lab_distance_threshold').get_parameter_value().double_value

        self.leak_sample_r = self.get_parameter('leak_sample_radius_px').get_parameter_value().integer_value

        self.bridge = CvBridge()

        self.latest_rgb = None

        self.detected_colors = {}

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)

        self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

        self.cam_info_sub = self.create_subscription(CameraInfo, "/oakd/rgb/preview/camera_info", self.camera_info_callback, 10)

        self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.RELIABLE)

        self.debug_image_pub = self.create_publisher(Image, "/barrels_debug_image", 10)

        self.finished_pub = self.create_publisher(Bool, "/finished", 10)

        self.report_pub = self.create_publisher(String, "/barrel_report", 10)

        self.barrels_clusters = []

        self.ema_alpha = 0.1

        self.create_timer(1.0, self.publish_clusters)

        self.get_logger().info(f"Node initialized. Publishing markers to {marker_topic}.")

        if not HAS_PCL:
            self.get_logger().warn('python-pcl not found.')

        if not HAS_O3D:
            self.get_logger().warn('open3d not found.')

    # ==========================================================
    # Speech
    # ==========================================================

    def say(self, text):

        try:
            subprocess.run(["spd-say", "-r", "-60", "-p", "-55", "-t", "male3", text], check=False)

        except FileNotFoundError:

            print(f"spd-say not found. Would say: {text}")

    # ==========================================================
    # Camera info
    # ==========================================================

    def camera_info_callback(self, msg: CameraInfo):

        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

        self.destroy_subscription(self.cam_info_sub)

        self.get_logger().info(f"Updated camera intrinsics.")

    # ==========================================================
    # RGB callback
    # ==========================================================

    def rgb_callback(self, data):

        try:

            self.latest_rgb = self.bridge.imgmsg_to_cv2(data, "bgr8")

        except CvBridgeError as e:

            self.get_logger().error(f"CV Bridge error: {e}")

    # ==========================================================
    # Main pointcloud callback
    # ==========================================================

    def pointcloud_callback(self, data):

        if self.latest_rgb is None:
            return
        gen = pc2.read_points(
            data,
            field_names=("x", "y", "z"),
            skip_nans=True
        )
        
        pts_list = []
        
        for p in gen:
        
            x = float(p[0])
            y = float(p[1])
            z = float(p[2])
        
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
        
                pts_list.append([x, y, z])
        
        if len(pts_list) < 50:
            return

        pts_map = self._transform_points_to_map(pts_camera, data.header)

        if pts_map is None:
            return

        pts_no_floor = self._remove_floor(pts_map)

        if pts_no_floor is None or len(pts_no_floor) < 30:
            return

        pts_down = self._voxel_downsample(pts_no_floor, voxel=0.02)

        cylinders = self._segment_cylinders(pts_down)

        for cyl in cylinders:

            cx3, cy3, cz3 = cyl["centroid"]

            axis = cyl["axis"]

            radius = cyl["radius"]

            centroid_camera = self._map_to_camera_point(np.array([cx3, cy3, cz3]), data.header.frame_id)

            if centroid_camera is None:
                continue

            ccam_x, ccam_y, ccam_z = centroid_camera

            color = self.detect_barrel_color(self.latest_rgb, ccam_x, ccam_y, ccam_z)

            orientation = self.detect_orientation(axis)

            leaking = False

            if orientation == "horizontal":

                leaking = self.check_leak(self.latest_rgb, ccam_x, ccam_y, ccam_z)

                if leaking:

                    color_name = (color or {}).get("name", "unknown")

                    self.get_logger().warn(f"Leak detected on {color_name} barrel!")

                    self.say(f"Alert! Alert! Leak detected on {color_name} barrel!")

            self.add_to_clusters(
                cx3,
                cy3,
                cz3,
                color=color,
                axis=axis,
                radius=radius,
                orientation=orientation,
                leaking=leaking
            )

        # Publish debug image
        '''
        Green circles: Center points of detected barrels
        Blue circles: Projected barrel radius in image space
        Text labels: Radius and orientation (V for vertical, H for horizontal)
        '''
        if self.latest_rgb is not None and len(cylinders) > 0:
            vis_img = self.visualize_detections(self.latest_rgb, cylinders)
            try:
                msg = self.bridge.cv2_to_imgmsg(vis_img, "bgr8")
                self.debug_image_pub.publish(msg)
            except CvBridgeError as e:
                self.get_logger().error(f"CV Bridge error: {e}")

    # ==========================================================
    # TF helpers
    # ==========================================================

    def _transform_points_to_map(self, pts, header):

        try:

            transform = self.tf_buffer.lookup_transform('map', header.frame_id, rclpy.time.Time())

        except TransformException:

            return pts

        q = transform.transform.rotation

        t = transform.transform.translation

        R = self._quat_to_rot(q.x, q.y, q.z, q.w)

        translation = np.array([t.x, t.y, t.z])

        return (R @ pts.T).T + translation

    def _map_to_camera_point(self, point_map, camera_frame):

        try:

            transform = self.tf_buffer.lookup_transform(camera_frame, 'map', rclpy.time.Time())

            p = PointStamped()

            p.header.frame_id = "map"

            p.point.x = float(point_map[0])
            p.point.y = float(point_map[1])
            p.point.z = float(point_map[2])

            transformed = do_transform_point(p, transform)

            return np.array([transformed.point.x, transformed.point.y, transformed.point.z])

        except Exception:

            return None

    @staticmethod
    def _quat_to_rot(x, y, z, w):

        return np.array([
            [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
            [2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])

    # ==========================================================
    # Floor removal
    # ==========================================================

    def _remove_floor(self, pts):

        if HAS_O3D:

            pcd = o3d.geometry.PointCloud()

            pcd.points = o3d.utility.Vector3dVector(pts)

            _, inliers = pcd.segment_plane(
                distance_threshold=0.03,
                ransac_n=3,
                num_iterations=200
            )

            mask = np.ones(len(pts), dtype=bool)

            mask[inliers] = False

            return pts[mask]

        floor_z = np.percentile(pts[:, 2], 5)

        return pts[pts[:, 2] > floor_z + 0.05]

    # ==========================================================
    # Downsampling
    # ==========================================================

    def _voxel_downsample(self, pts, voxel=0.02):

        if HAS_O3D:

            pcd = o3d.geometry.PointCloud()

            pcd.points = o3d.utility.Vector3dVector(pts)

            pcd = pcd.voxel_down_sample(voxel_size=voxel)

            return np.asarray(pcd.points, dtype=np.float32)

        if len(pts) > 1000:

            idx = np.random.choice(len(pts), 1000, replace=False)

            return pts[idx]

        return pts

    # ==========================================================
    # Cylinder segmentation
    # ==========================================================

    def _segment_cylinders(self, pts):

        if HAS_O3D:

            pcd = o3d.geometry.PointCloud()

            pcd.points = o3d.utility.Vector3dVector(pts)

            labels = np.asarray(pcd.cluster_dbscan(eps=0.08, min_points=60))

            cylinders = []

            max_label = labels.max()

            for cluster_id in range(max_label + 1):

                indices = np.where(labels == cluster_id)[0]

                if len(indices) < 80:
                    continue

                cluster_pts = np.asarray(pts[indices], dtype=np.float32)

                centroid = np.mean(cluster_pts, axis=0)

                cov = np.cov(cluster_pts.T)

                eigvals, eigvecs = np.linalg.eigh(cov)

                axis = eigvecs[:, np.argmax(eigvals)]

                centered = cluster_pts - centroid

                projected = centered - np.outer(centered @ axis, axis)

                radii = np.linalg.norm(projected, axis=1)

                radius = np.median(radii)

                if radius < self.cyl_r_min:
                    continue

                if radius > self.cyl_r_max:
                    continue

                cylinders.append({
                    "centroid": centroid.tolist(),
                    "axis": axis.tolist(),
                    "radius": float(radius),
                    "inlier_pts": cluster_pts
                })

            return cylinders

        return []
    


    # ==========================================================
    # Visualization
    # ==========================================================

    def visualize_detections(self, cv_image, cylinders):
        """Draw detected cylinders on RGB image"""
        
        vis_image = cv_image.copy()
        
        for cyl in cylinders:
            centroid_camera = self._map_to_camera_point(
                np.array(cyl["centroid"]), 
                "oakd_rgb_camera_frame"
            )
            
            if centroid_camera is None:
                continue
            
            x, y, z = centroid_camera
            
            if z <= 0:
                continue
            
            # Project to image
            u = int(self.fx * x / z + self.cx)
            v = int(self.fy * y / z + self.cy)
            
            h, w = cv_image.shape[:2]
            if u < 0 or u >= w or v < 0 or v >= h:
                continue
            
            # Draw circle at barrel center
            cv2.circle(vis_image, (u, v), 10, (0, 255, 0), 2)
            
            # Draw radius
            radius_px = int(self.fx * cyl["radius"] / z)
            cv2.circle(vis_image, (u, v), radius_px, (255, 0, 0), 1)
            
            # Add text with info
            orientation = self.detect_orientation(cyl["axis"])
            text = f"R:{cyl['radius']:.2f}m {orientation[0].upper()}"
            cv2.putText(vis_image, text, (u - 30, v - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return vis_image

    # ==========================================================
    # Orientation detection
    # ==========================================================

    def detect_orientation(self, axis):

        axis = np.array(axis)

        axis /= np.linalg.norm(axis) + 1e-9

        z_axis = np.array([0.0, 0.0, 1.0])

        cos_angle = np.abs(np.dot(axis, z_axis))

        tilt_deg = np.degrees(np.arccos(cos_angle))

        return (
            "vertical"
            if tilt_deg <= 45.0
            else "horizontal"
        )

    # ==========================================================
    # Leak detection
    # ==========================================================

    def check_leak(self, cv_image, x, y, z):

        if cv_image is None or z <= 0:
            return False

        h, w = cv_image.shape[:2]

        u = int(self.fx * x / z + self.cx)
        v = int(self.fy * y / z + self.cy)

        r = self.leak_sample_r

        p = 12

        sample_offsets = [
            (0, r),
            (0, -r),
            (-r, 0),
            (r, 0),
        ]

        for du, dv in sample_offsets:

            su = u + du
            sv = v + dv

            u0 = max(0, su - p)
            u1 = min(w, su + p)

            v0 = max(0, sv - p)
            v1 = min(h, sv + p)

            if u1 <= u0 or v1 <= v0:
                continue

            patch = cv_image[v0:v1, u0:u1]

            if patch.size == 0:
                continue

            patch_f32 = patch.astype(np.float32) / 255.0

            patch_lab = cv2.cvtColor(patch_f32, cv2.COLOR_BGR2Lab)

            mean_lab = patch_lab.mean(axis=(0, 1))

            delta_e = np.linalg.norm(mean_lab - self.FLOOR_LAB_REF)

            if delta_e > self.leak_lab_thresh:
                return True

        return False

    # ==========================================================
    # Cluster handling
    # ==========================================================

    def add_to_clusters(self, x, y, z, color=None, axis=None, radius=0.0, orientation='vertical', leaking=False):

        new_point = np.array([x, y, z])

        now_sec = (self.get_clock().now().nanoseconds / 1e9)

        for cluster in self.barrels_clusters:

            centroid = np.array(cluster["centroid"])

            distance = np.linalg.norm(new_point[:2] - centroid[:2])

            if distance < self.cluster_threshold:

                cluster["count"] += 1

                cluster["last_seen"] = now_sec

                cluster["centroid"] = ((1 - self.ema_alpha) * centroid + self.ema_alpha * new_point).tolist()

                if color:

                    cluster["color"] = color

                cluster["orientation"] = orientation

                if leaking:
                    cluster["leaking"] = True

                return

        self.barrels_clusters.append({

            "centroid": new_point.tolist(),

            "axis": axis or [0.0, 0.0, 1.0],

            "radius": radius,

            "count": 1,

            "color": color,

            "orientation": orientation,

            "leaking": leaking,

            "last_seen": now_sec,

            "published": False
        })

    # ==========================================================
    # Marker publishing
    # ==========================================================

    def publish_clusters(self):

        for i, cluster in enumerate(
            self.barrels_clusters
        ):

            if cluster["count"] < self.min_detections:
                continue

            color = cluster.get("color")

            name = (color or {}).get("name", "unknown")

            if not cluster["published"]:

                self.detected_colors[name] = (self.detected_colors.get(name, 0) + 1)

            marker = Marker()

            marker.header.frame_id = "map"

            marker.header.stamp = (self.get_clock().now().to_msg())

            marker.ns = "barrels"

            marker.id = i

            marker.type = Marker.CYLINDER

            marker.action = Marker.ADD

            cx, cy, cz = cluster["centroid"]

            marker.pose.position.x = cx
            marker.pose.position.y = cy
            marker.pose.position.z = cz

            marker.pose.orientation.w = 1.0

            r = max(float(cluster.get("radius", 0.2)), 0.1)

            marker.scale.x = r * 2
            marker.scale.y = r * 2
            marker.scale.z = 0.85

            if color:

                r_val, g_val, b_val = self.lab_to_marker_rgb(color["lab"])

            else:

                r_val, g_val, b_val = (0.5, 0.5, 0.5)

            marker.color.r = r_val
            marker.color.g = g_val
            marker.color.b = b_val
            marker.color.a = 0.85

            if cluster.get("leaking"):

                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0

            marker.lifetime = Duration(seconds=5.0).to_msg()

            self.marker_pub.publish(marker)

            cluster["published"] = True

    # ==========================================================
    # Color detection
    # ==========================================================

    def detect_barrel_color(self, cv_image, x, y, z, patch_size=20):

        if cv_image is None or z <= 0:
            return None

        h, w = cv_image.shape[:2]

        u = int(self.fx * x / z + self.cx)
        v = int(self.fy * y / z + self.cy)

        half = patch_size // 2

        u0 = max(0, u - half)
        u1 = min(w, u + half)

        v0 = max(0, v - half)
        v1 = min(h, v + half)

        if u1 <= u0 or v1 <= v0:
            return None

        patch = cv_image[v0:v1, u0:u1]

        if patch.size == 0:
            return None

        patch_f32 = patch.astype(np.float32) / 255.0

        patch_lab = cv2.cvtColor(patch_f32, cv2.COLOR_BGR2Lab)

        mean_lab = patch_lab.mean(axis=(0, 1))

        L, A, B = mean_lab

        name = self.classify_lab(L, A, B)

        return {
            "lab": mean_lab.tolist(),
            "name": name
        }

    # ==========================================================
    # LAB classification
    # ==========================================================

    def classify_lab(self, L, A, B):

        chroma = np.sqrt(A**2 + B**2)

        if chroma < 15:

            if L < 20:
                return 'black'

            if L > 90:
                return 'white'

            return 'unknown'

        hue = (
            np.degrees(np.arctan2(B, A))
        ) % 360.0

        if hue < 20 or hue >= 340:
            return 'red'

        if 20 <= hue < 50:
            return 'orange'

        if 50 <= hue < 80:
            return 'yellow'

        if 80 <= hue < 155:
            return 'green'

        if 155 <= hue < 265:
            return 'blue'

        if 265 <= hue < 340:
            return 'red'

        return 'unknown'

    # ==========================================================
    # LAB -> RViz color
    # ==========================================================

    def lab_to_marker_rgb(self, lab):

        lab_pixel = np.array(
            lab,
            dtype=np.float32
        ).reshape(1, 1, 3)

        bgr = cv2.cvtColor(
            lab_pixel,
            cv2.COLOR_Lab2BGR
        )

        bgr = np.clip(bgr, 0, 1)

        b, g, r = bgr[0, 0]

        return (
            float(r),
            float(g),
            float(b)
        )

    # ==========================================================
    # Final report
    # ==========================================================

    def publish_final_report(self):

        confirmed = [

            c for c in self.barrels_clusters

            if c["count"] >= self.min_detections
        ]

        total = len(confirmed)

        vertical = sum(1 for c in confirmed if c["orientation"] == "vertical")

        horizontal = total - vertical

        leaking = sum(1 for c in confirmed if c.get("leaking"))

        color_summary = ', '.join(f'{cnt} {col}' for col, cnt in self.detected_colors.items())

        report = f'''
=== Barrel Report ===
Total barrels : {total}
Vertical      : {vertical}
Horizontal    : {horizontal}
Leaking       : {leaking}
Colors        : {color_summary}
'''

        self.get_logger().info(report)

        msg = String()

        msg.data = report

        self.report_pub.publish(msg)


# ==============================================================
# MAIN
# ==============================================================

def main():

    print('Barrel detection node starting.')

    rclpy.init(args=None)

    node = detect_barrels()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.publish_final_report()

        node.destroy_node()

        try:

            rclpy.shutdown()

        except Exception as e:

            print(f'Shutdown error: {e}')


if __name__ == '__main__':
    main()
