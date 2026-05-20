#!/usr/bin/env python3

import sys
import numpy as np
import rclpy
from rclpy.node import Node

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Header

import tf2_ros
import tf2_geometry_msgs

import open3d as o3d


# Parameters
ERROR_MARGIN = 0.04          # 4 cm margin for radius error
TARGET_RADIUS = 0.11         # 11 cm radius
VERBOSE = False

# Cloud filtering
X_LIMIT_LOW = 0.0
X_LIMIT_HIGH = 3.0
Z_LIMIT_LOW = -0.2
Z_LIMIT_HIGH = 0.3

# RANSAC
RANSAC_MAX_ITERATIONS = 50
RANSAC_NORMAL_DISTANCE_WEIGHT = 0.3
RANSAC_DISTANCE_THRESHOLD = 0.005

MARKER_HEIGHT = 0.4
MAX_DETECTED_CYLINDERS = 3
MIN_CYLINDER_SIZE = 500


def numpy_to_pointcloud2(points: np.ndarray, header: Header) -> PointCloud2:
    """Convert an Nx3 numpy array to a PointCloud2 message."""
    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    cloud_msg = pc2.create_cloud(header, fields, points.astype(np.float32))
    return cloud_msg


def fit_cylinder_ransac(points: np.ndarray,
                        radius_min: float,
                        radius_max: float,
                        distance_threshold: float = 0.005,
                        max_iterations: int = 50):
    """
    Fit a vertical cylinder to a point cloud using RANSAC via Open3D.

    Returns (center_x, center_y, radius, inlier_mask) or None if fitting fails.
    The cylinder axis is assumed to be aligned with Z (vertical).
    """
    if len(points) < 10:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50)
    )

    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    best_inliers = []
    best_radius = None
    best_cx = None
    best_cy = None

    rng = np.random.default_rng()

    for _ in range(max_iterations):
        # Sample 3 points
        idx = rng.choice(len(pts), 3, replace=False)
        p1, p2, p3 = pts[idx[0], :2], pts[idx[1], :2], pts[idx[2], :2]

        # Circumcircle of the three XY projections
        ax, ay = p1
        bx, by = p2
        cx, cy = p3

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            continue

        ux = ((ax**2 + ay**2) * (by - cy) +
              (bx**2 + by**2) * (cy - ay) +
              (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) +
              (bx**2 + by**2) * (ax - cx) +
              (cx**2 + cy**2) * (bx - ax)) / d

        radius = np.sqrt((ax - ux)**2 + (ay - uy)**2)

        if not (radius_min <= radius <= radius_max):
            continue

        # Count inliers: distance from point (x,y) to circle centre
        dists = np.sqrt((pts[:, 0] - ux)**2 + (pts[:, 1] - uy)**2)
        inlier_mask = np.abs(dists - radius) < distance_threshold

        if inlier_mask.sum() > len(best_inliers):
            best_inliers = np.where(inlier_mask)[0]
            best_radius = radius
            best_cx = ux
            best_cy = uy

    if best_radius is None or len(best_inliers) == 0:
        return None

    return best_cx, best_cy, best_radius, best_inliers


class CylinderSegmentation(Node):

    def __init__(self):
        super().__init__('cylinder_segmentation')

        self.get_logger().info('cylinder_segmentation started')

        # Declare and read parameter
        self.declare_parameter('topic_pointcloud_in', '/oakd/rgb/preview/depth/points')
        topic_in = self.get_parameter('topic_pointcloud_in').get_parameter_value().string_value

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publishers
        self.planes_pub = self.create_publisher(PointCloud2, 'filtered_point_cloud', 1)
        self.cylinder_pub = self.create_publisher(PointCloud2, 'cylinder_point_cloud', 1)
        self.marker_pub = self.create_publisher(Marker, 'cylinder_markers', 1)

        # Subscriber
        self.subscription = self.create_subscription(
            PointCloud2,
            topic_in,
            self.cloud_cb,
            10
        )

    # ------------------------------------------------------------------
    def cloud_cb(self, msg: PointCloud2):
        now = msg.header.stamp
        from_frame = msg.header.frame_id
        to_frame = 'map'

        # ---- Convert PointCloud2 → numpy ----
        gen = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points = np.array(list(gen), dtype=np.float32)  # (N, 3)

        if len(points) == 0:
            self.get_logger().warn('Received empty point cloud.')
            return

        if VERBOSE:
            self.get_logger().info(f'PointCloud has: {len(points)} data points.')

        # ---- PassThrough filter ----
        mask_x = (points[:, 0] >= X_LIMIT_LOW) & (points[:, 0] <= X_LIMIT_HIGH)
        mask_z = (points[:, 2] >= Z_LIMIT_LOW) & (points[:, 2] <= Z_LIMIT_HIGH)
        points = points[mask_x & mask_z]

        if VERBOSE:
            self.get_logger().info(f'PointCloud after filtering has: {len(points)} data points.')

        # Publish filtered cloud
        filtered_header = Header()
        filtered_header.stamp = now
        filtered_header.frame_id = from_frame
        self.planes_pub.publish(numpy_to_pointcloud2(points, filtered_header))

        # ---- Iterative cylinder extraction ----
        remaining = points.copy()
        all_cylinders = []
        detected_cylinders = 0
        marker_id = 0

        while detected_cylinders <= MAX_DETECTED_CYLINDERS:
            if len(remaining) < MIN_CYLINDER_SIZE:
                break

            result = fit_cylinder_ransac(
                remaining,
                radius_min=TARGET_RADIUS - ERROR_MARGIN,
                radius_max=TARGET_RADIUS + ERROR_MARGIN,
                distance_threshold=RANSAC_DISTANCE_THRESHOLD,
                max_iterations=RANSAC_MAX_ITERATIONS,
            )

            if result is None:
                break

            cx, cy, detected_radius, inlier_idx = result
            cylinder_points_count = len(inlier_idx)

            # Extract cylinder points
            cloud_cylinder = remaining[inlier_idx]

            # ---- TF transform ----
            point_camera = PointStamped()
            point_camera.header.frame_id = from_frame
            point_camera.header.stamp = now
            point_camera.point.x = float(cx)
            point_camera.point.y = float(cy)
            point_camera.point.z = float(MARKER_HEIGHT)

            try:
                transform = self.tf_buffer.lookup_transform(to_frame, from_frame, now)
                point_map = tf2_geometry_msgs.do_transform_point(point_camera, transform)
            except tf2_ros.TransformException as ex:
                self.get_logger().warn(str(ex))
                break

            # ---- Accept cylinder if within margin and large enough ----
            if (abs(detected_radius - TARGET_RADIUS) <= ERROR_MARGIN and
                    cylinder_points_count >= MIN_CYLINDER_SIZE):

                if VERBOSE:
                    self.get_logger().info(f'Cylinder radius: {detected_radius}')
                    self.get_logger().info(f'Cylinder_points_count: {cylinder_points_count}')

                # Publish marker
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.header.stamp = now
                marker.ns = 'cylinder'
                marker.id = marker_id
                marker_id += 1

                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD

                marker.pose.position.x = point_map.point.x
                marker.pose.position.y = point_map.point.y
                marker.pose.position.z = MARKER_HEIGHT / 2.0
                marker.pose.orientation.w = 1.0

                marker.scale.x = detected_radius * 2
                marker.scale.y = detected_radius * 2
                marker.scale.z = MARKER_HEIGHT

                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                marker.lifetime.nanosec = 1  # ~1 ns (effectively immediate)

                self.marker_pub.publish(marker)

                all_cylinders.append(cloud_cylinder)
                detected_cylinders += 1

            # Remove inliers from remaining cloud
            mask = np.ones(len(remaining), dtype=bool)
            mask[inlier_idx] = False
            remaining = remaining[mask]

        self.get_logger().info(f'Detected {detected_cylinders} cylinders.')

        # Publish combined cylinder cloud
        if all_cylinders:
            combined = np.vstack(all_cylinders)
            cyl_header = Header()
            cyl_header.stamp = now
            cyl_header.frame_id = from_frame
            self.cylinder_pub.publish(numpy_to_pointcloud2(combined, cyl_header))


# -----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = CylinderSegmentation()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
