#include <iostream>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/centroid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2/convert.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/msg/marker.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <map>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>

#include "rins_interfaces/msg/barrels_results.hpp"

/*
ZA DODAT:
- to potem publisha na nek topic
- za preverit kako se orientacijo zamenja - se pravi X ali Y axis?
*/

bool verbose = true; // debug outout

// Function to play text-to-speech alert
static void say(const std::string& message) {
    try {

        std::string cmd = "espeak \"" + message + "\" 2>/dev/null &";
        
        int result = std::system(cmd.c_str());
        if (result != 0 && verbose) {
            std::cerr << "Warning: Text-to-speech may have failed (exit code: " << result << ")" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in say: " << e.what() << std::endl;
    }
}

rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr viz_image_pub;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr planes_pub;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cylinder_pub;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub;
rclcpp::Publisher<rins_interfaces::msg::BarrelsResults>::SharedPtr barrels_results_pub;

std::shared_ptr<rclcpp::Node> node;
std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

typedef pcl::PointXYZRGB PointT;

// Camera intrinsics
float fx = 383.5f, fy = 383.5f;  // focal lengths (update based on camera_info)
float cx = 320.0f, cy = 240.0f;  // principal point
int image_width = 640, image_height = 480;

// parameters
float error_margin = 0.04;  // 4 cm margin for radius error
float target_radius = 0.11; // 11cm radius

// cloud filtering
float x_limit_low = 0; // only process points 0-3m in X direction
float x_limit_high = 3;
float z_limit_low = -0.2; // keep points between -0.2 ro 0.3m in Z
float z_limit_high = 0.3;

// RANSAC
int ransac_max_iterations = 50;
float ransac_normal_distance_weight = 0.3; // how much to trust normals
float ransac_distance_threshold = 0.005; // 5mm inlier threshold

float marker_height = 0.4;
//int max_detected_cylinders = 20;
int min_cylinder_size = 500;

struct ColorStats {
    float avg_r, avg_g, avg_b;
    float std_r, std_g, std_b;
};

struct CylinderRecord {
    int cylinder_id = -1;
    rclcpp::Time stamp;
    geometry_msgs::msg::Point position_map;
    std::string color;

    // Placeholders for later (requested: don't implement now)
    std::string orientation = "UNKNOWN"; // e.g. HORIZONTAL/VERTICAL later
    bool leakage = false;

    ColorStats color_stats;
};

static std::vector<CylinderRecord> saved_cylinders;
static int next_cylinder_id = 0;
static int max_saved_cylinders = 20; // keeps at least 10

struct SaveResult {
    int cylinder_id = -1;
    bool inserted = false;
};

static SaveResult upsertCylinder(
    const rclcpp::Time& stamp,
    const geometry_msgs::msg::Point& position_map,
    const std::string& color,
    const ColorStats& color_stats,
    const std::string& orientation,
    bool leakage) {

    constexpr double kMergeDistance = 0.5; // meters
    const double merge_distance_sq = kMergeDistance * kMergeDistance;

    for (auto& existing : saved_cylinders) {
        const double dx = existing.position_map.x - position_map.x;
        const double dy = existing.position_map.y - position_map.y;
        const double dz = existing.position_map.z - position_map.z;
        const double dist_sq = dx * dx + dy * dy + dz * dz;

        if (dist_sq <= merge_distance_sq) {
            existing.stamp = stamp;
            existing.position_map = position_map;
            existing.color = color;
            existing.color_stats = color_stats;
            existing.orientation = orientation;
            existing.leakage = leakage;
            // existing.orientation stays as-is (UNKNOWN for now)
            // existing.leakage stays as-is (false for now)
            return SaveResult{existing.cylinder_id, false};
        }
    }

    CylinderRecord rec;
    rec.cylinder_id = next_cylinder_id++;
    rec.stamp = stamp;
    rec.position_map = position_map;
    rec.color = color;
    rec.color_stats = color_stats;
    rec.orientation = orientation;
    rec.leakage = leakage;

    saved_cylinders.push_back(rec);

    const int keep = std::max(10, max_saved_cylinders);
    if ((int)saved_cylinders.size() > keep) {
        saved_cylinders.erase(saved_cylinders.begin()); // drop oldest
    }

    return SaveResult{rec.cylinder_id, true};
}




static std::string formatTimeSec(const rclcpp::Time& t) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    oss << t.seconds();
    return oss.str();
}

static void printSavedCylindersReport() {
    std::cout << "\n=== Saved Cylinders Report ===\n";
    std::cout << "count=" << saved_cylinders.size() << "\n";

    for (const auto& c : saved_cylinders) {
        std::cout << "- id=" << c.cylinder_id
                  << " t=" << formatTimeSec(c.stamp)
                  << " pos_map=[" << c.position_map.x << ", " << c.position_map.y << ", " << c.position_map.z << "]"
                  << " color=" << c.color
                  << " orientation=" << c.orientation
                  << " leakage=" << (c.leakage ? "true" : "false")
                  << " avg_rgb=[" << (int)c.color_stats.avg_r << ", " << (int)c.color_stats.avg_g << ", " << (int)c.color_stats.avg_b << "]"
                  << "\n";
    }
    std::cout << "=== End Saved Cylinders Report ===\n";
    
    // Compose and publish BarrelsResults message
    if (barrels_results_pub) {
        rins_interfaces::msg::BarrelsResults msg;
        msg.total = saved_cylinders.size();
        
        for (const auto& c : saved_cylinders) {
            msg.barve.push_back(c.color);
            msg.orientacija.push_back(c.orientation);
            msg.leak.push_back(c.leakage);
        }
        
        barrels_results_pub->publish(msg);
        std::cout << "Published BarrelsResults: total=" << msg.total 
                  << " colors=[";
        for (size_t i = 0; i < msg.barve.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << msg.barve[i];
        }
        std::cout << "] orientations=[";
        for (size_t i = 0; i < msg.orientacija.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << msg.orientacija[i];
        }
        std::cout << "] leaks=[";
        for (size_t i = 0; i < msg.leak.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << (msg.leak[i] ? "true" : "false");
        }
        std::cout << "]\n";
    }
}


int marker_id = 0;
int detected_cylinders = 0;

ColorStats analyzeColors(const pcl::PointCloud<PointT>::Ptr& cloud) {
    ColorStats stats = {0, 0, 0, 0, 0, 0};
    if (cloud->empty()) return stats;
    
    float sum_r = 0, sum_g = 0, sum_b = 0;
    int n = cloud->points.size();
    
    for (const auto& point : cloud->points) {
        sum_r += static_cast<float>(point.r);
        sum_g += static_cast<float>(point.g);
        sum_b += static_cast<float>(point.b);
    }
    
    stats.avg_r = sum_r / n;
    stats.avg_g = sum_g / n;
    stats.avg_b = sum_b / n;
    return stats;
}

std::string identifyColor(const ColorStats& stats) {
    float r = stats.avg_r, g = stats.avg_g, b = stats.avg_b;
    float max_val = std::max({r, g, b});
    if (max_val < 10) return "BLACK";
    
    r /= max_val; g /= max_val; b /= max_val;
    
    if (b > 0.7 && g < 0.5 && r < 0.5) return "RED";
    if (g > 0.7 && b < 0.3 && r < 0.3) return "GREEN";
    if (r > 0.8 && b < 0.5 && g < 0.8) return "BLUE";
    if (b > 0.6 && g > 0.6 && r < 0.3) return "YELLOW";
    return "OTHER";
}

// leakage se začne tuki
static std::string identifyColorFromRgb(uint8_t r, uint8_t g, uint8_t b) {
    ColorStats stats;
    stats.avg_r = static_cast<float>(r);
    stats.avg_g = static_cast<float>(g);
    stats.avg_b = static_cast<float>(b);
    stats.std_r = stats.std_g = stats.std_b = 0.0f;
    return identifyColor(stats);
}

struct LeakageResult {
    bool leaking = false;
    std::string dominant1;
    std::string dominant2;
    int total_samples = 0;
};

static LeakageResult detectLeakageHorizontal(
    const pcl::PointCloud<PointT>::Ptr& reference_cloud,
    const Eigen::Vector4f& cylinder_centroid_cam,
    const Eigen::Vector3f& cylinder_axis_cam,
    const std::string& barrel_color) {

    LeakageResult result;
    if (!reference_cloud || reference_cloud->empty()) {
        return result;
    }

    // Build a floor-parallel ROI:
    // - along the (horizontal) cylinder axis, elongated by height/2 on each side -> total length = 2*marker_height
    // - width = radius
    const float half_length = marker_height;      // (marker_height + marker_height/2 + marker_height/2) / 2
    const float barrel_half_length = marker_height * 0.5f; // exclude the barrel body itself (keep only front/back)
    const float half_width = target_radius;       // radius
    const float z_center = cylinder_centroid_cam[2];

    // Project axis onto the floor plane (XY) so ROI is floor-parallel.
    Eigen::Vector3f axis_xy(cylinder_axis_cam.x(), cylinder_axis_cam.y(), 0.0f);
    const float axis_xy_norm = axis_xy.norm();
    if (axis_xy_norm < 1e-6f) {
        return result;
    }
    axis_xy /= axis_xy_norm;
    Eigen::Vector3f perp_xy(-axis_xy.y(), axis_xy.x(), 0.0f);

    // Focus on points below the barrel centroid (puddle on the floor).
    const float max_z_for_floor = z_center - (target_radius * 0.2f);

    std::map<std::string, int> counts;

    for (const auto& p : reference_cloud->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        if (p.z > max_z_for_floor) {
            continue;
        }

        Eigen::Vector3f delta(p.x - cylinder_centroid_cam[0], p.y - cylinder_centroid_cam[1], 0.0f);
        const float along = delta.dot(axis_xy);
        const float perp = delta.dot(perp_xy);

        if (std::abs(along) > half_length || std::abs(perp) > half_width) {
            continue;
        }

        // Exclude the barrel itself: keep only points behind/in front of the barrel along its axis.
        if (std::abs(along) < barrel_half_length) {
            continue;
        }

        // Per-point “pixel” classification using the same thresholds as barrel color classification.
        const std::string c = identifyColorFromRgb(p.r, p.g, p.b);
        if (c == "OTHER") {
            continue;
        }

        counts[c]++;
        result.total_samples++;
    }

    // Need enough samples to be meaningful.
    if (result.total_samples < 50) {
        return result;
    }

    // Extract top-2 dominant colors.
    std::pair<std::string, int> best1{"", 0};
    std::pair<std::string, int> best2{"", 0};
    for (const auto& kv : counts) {
        if (kv.second > best1.second) {
            best2 = best1;
            best1 = kv;
        } else if (kv.second > best2.second) {
            best2 = kv;
        }
    }

    result.dominant1 = best1.first;
    result.dominant2 = best2.first;
    result.leaking = (!barrel_color.empty()) &&
                     (barrel_color == result.dominant1 || barrel_color == result.dominant2);
    return result;
}


/*
X JE MIRRORED, TREBA POPRAVIT (ZA VIZUALIZACIJO)
*/
void colorizePointCloud(pcl::PointCloud<PointT>::Ptr& cloud, const cv::Mat& rgb_image) {
    int colored_count = 0;
    int out_of_bounds = 0;

    for (auto& point : cloud->points) {
        if (point.x <= 0) {
            point.r = point.g = point.b = 0;
            continue;
        }
        
        int u = (int)(fx * (-point.y) / point.x + cx);
        int v = (int)(fy * (-point.z) / point.x + cy);

        
        if (u >= 0 && u < image_width && v >= 0 && v < image_height) {
            cv::Vec3b bgr = rgb_image.at<cv::Vec3b>(v, u);
            point.r = bgr[2];  // BGR to RGB
            point.g = bgr[1];
            point.b = bgr[0];
            colored_count++;
        } else {
            point.r = point.g = point.b = 128;
            out_of_bounds++;
        }
    }
    /*
    if (verbose) {
        std::cerr << "Colorized " << colored_count << " points, " 
                  << out_of_bounds << " out of bounds" << std::endl;
    }*/
}


void publishVisualization(
    const pcl::PointCloud<PointT>::Ptr& cloud_cylinder,
    const cv::Mat& rgb_image,
    const rclcpp::Time& timestamp) {
    
    //cv::Mat viz = rgb_image.clone();

    cv::Mat viz;
    cv::cvtColor(rgb_image, viz, cv::COLOR_RGB2BGR);
    
    for (const auto& point : cloud_cylinder->points) {
        if (point.x <= 0) continue;
        
        int u = (int)(fx * (-point.y) / point.x + cx);
        int v = (int)(fy * (-point.z) / point.x + cy);

        if (u >= 0 && u < viz.cols && v >= 0 && v < viz.rows) {
            // Draw green circles at detected points
            cv::circle(viz, cv::Point(u, v), 3, cv::Scalar(50, 0, 50), -1);
        }
    }

    cv::imshow("Cylinder Detection", viz);
    cv::waitKey(1);
    
    // Convert Mat to ROS 2 image message and publish
    std_msgs::msg::Header header;
    header.stamp = timestamp;
    header.frame_id = "rgb_frame";
    
    auto msg = cv_bridge::CvImage(header, "bgr8", viz).toImageMsg();
    viz_image_pub->publish(*msg);
}

// Pointcloud callback
void cloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg, const sensor_msgs::msg::Image::SharedPtr image_msg) {
    // save timestamp from message
    rclcpp::Time now = (*msg).header.stamp;

    // set up PCL objects
    pcl::PassThrough<PointT> pass; // removes points outside bounds
    pcl::NormalEstimation<PointT, pcl::Normal> ne; // calculates surface normals
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg; // RANSAC cylinder finder
    pcl::PCDWriter writer; 
    pcl::ExtractIndices<PointT> extract; // extracts specific points from cloud
    pcl::ExtractIndices<pcl::Normal> extract_normals;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

    // set up pointers
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::PCLPointCloud2::Ptr pcl_pc(new pcl::PCLPointCloud2);
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered2(new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients), coefficients_cylinder(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices), inliers_cylinder(new pcl::PointIndices);
    pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());

    // convert ROS msg to PointCloud2
    pcl_conversions::toPCL(*msg, *pcl_pc);

    // convert PointCloud2 to templated PointCloud
    pcl::fromPCLPointCloud2(*pcl_pc, *cloud);

    cv::Mat rgb_image;
    try {
        rgb_image = cv_bridge::toCvShare(image_msg, "rgb8")->image;
    } catch (cv_bridge::Exception& e) {
        std::cerr << "cv_bridge exception: " << e.what() << std::endl;
        return;
    }

    colorizePointCloud(cloud, rgb_image);

    /*
    if (verbose) {
        std::cerr << "PointCloud has: " << cloud->points.size() << " data points." << std::endl;
    }*/

    // Build a passthrough filter to remove spurious NaNs
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(x_limit_low, x_limit_high);
    pass.filter(*cloud_filtered);

    pass.setInputCloud(cloud_filtered);    
    pass.setFilterFieldName("z");
    pass.setFilterLimits(z_limit_low, z_limit_high);
    pass.filter(*cloud_filtered);
    
    /*
    if (verbose) {
        std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl;
    }*/

    // Estimate point normals: for each point calculates the surface normal
    // uses 50 nearest neighbours to estimate this -crucial for ransac -> they help distinguish
    // cylinders from random poin distributions
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud_filtered);
    ne.setKSearch(50);  // look at 50 nearest neighbours
    ne.compute(*cloud_normals);

    // limit to upwards orientation
    Eigen::Vector3f axis(0.0, 0.0, 1.0); // expect cylinders pointing UP (Z axis)
    seg.setAxis(axis);
    seg.setEpsAngle(0.8); // allow 0.8 radians (~46) deviation from vertical

    // Create the segmentation object for cylinder segmentation and set all the parameters
    // veritcal, match points within 5mm dist from the cylinder surface
    // only accept cylinders with radius between 7-15cm
    // use surface noramls to guide the fitting
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);  // fit cylinders, not planes/spheres
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(ransac_normal_distance_weight); // weight normal accuracy
    seg.setMaxIterations(ransac_max_iterations); // try up to 50 times
    seg.setDistanceThreshold(ransac_distance_threshold); // 5mm tolerance
    seg.setRadiusLimits(target_radius-error_margin, target_radius+error_margin); // 7-15cm
    seg.setInputCloud(cloud_filtered);
    seg.setInputNormals(cloud_normals);
    seg.setAxis(axis);

    // Obtain the cylinder inliers and coefficients
    seg.segment(*inliers_cylinder, *coefficients_cylinder);

    // Copy remaining cloud for iterative extraction
    pcl::PointCloud<PointT>::Ptr remaining_cloud(new pcl::PointCloud<PointT>(*cloud_filtered));
    pcl::PointCloud<pcl::Normal>::Ptr remaining_normals(new pcl::PointCloud<pcl::Normal>(*cloud_normals));

    pcl::PointCloud<PointT>::Ptr all_cylinders(new pcl::PointCloud<PointT>());

    // convert to pointcloud2, then to ROS2 message
    sensor_msgs::msg::PointCloud2 plane_out_msg;
    pcl::PCLPointCloud2::Ptr outcloud_plane(new pcl::PCLPointCloud2());
    pcl::toPCLPointCloud2(*cloud_filtered, *outcloud_plane);
    pcl_conversions::fromPCL(*outcloud_plane, plane_out_msg);
    planes_pub->publish(plane_out_msg);

    //int marker_id = 0;
    //int detected_cylinders = 0;

    // each loop iteration: finds the best-fitting cylinder in the remaining cloud
    // extracts cylinder points and stores them
    // removes those points from remaining_cloud
    // repeats up to 3 times or until no more cylinders found
    while (true) {

        pcl::PointIndices::Ptr inliers_cylinder(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);

        seg.setInputCloud(remaining_cloud);
        seg.setInputNormals(remaining_normals);
        seg.segment(*inliers_cylinder, *coefficients_cylinder);

        if (coefficients_cylinder->values.empty() || inliers_cylinder->indices.empty()) {
            break;
        }

        // extracts the cylinder's radius and number of inlier points from the RANSAC results
        float detected_radius = coefficients_cylinder->values[6];
        int cylinder_points_count = inliers_cylinder->indices.size();

        // Extract cylinder
        pcl::PointCloud<PointT>::Ptr cloud_cylinder(new pcl::PointCloud<PointT>());
        extract.setInputCloud(remaining_cloud);
        extract.setIndices(inliers_cylinder);
        extract.setNegative(false);
        extract.filter(*cloud_cylinder);

        geometry_msgs::msg::PointStamped point_camera, point_map;
        visualization_msgs::msg::Marker marker;

        std::string toFrameRel = "map";
        std::string fromFrameRel = (*msg).header.frame_id;

        // coordinate transform
        point_camera.header.frame_id = fromFrameRel;
        point_camera.header.stamp = now;
        point_camera.point.x = coefficients_cylinder->values[0];
        point_camera.point.y = coefficients_cylinder->values[1];
        point_camera.point.z = marker_height;

        try {
            auto tss = tf_buffer_->lookupTransform(toFrameRel, fromFrameRel, now);
            tf2::doTransform(point_camera, point_map, tss);
        } catch (tf2::TransformException& ex) {
            std::cout << ex.what() << std::endl;
            break;
        }

        // accept cylinders within margin: radius within 4cm of the 11cm target
        // have at least 500 inlier points
        if ((std::abs(detected_radius - target_radius) <= error_margin) && (cylinder_points_count>=min_cylinder_size)) {
            
            ColorStats color_stats = analyzeColors(cloud_cylinder);
            std::string color_name = identifyColor(color_stats);

            if(color_name != "OTHER") {

                // Save cylinder (map coords already computed as point_map)
                const SaveResult save_result = upsertCylinder(now, point_map.point, color_name, color_stats, "VERTICAL", false);
                const int saved_id = save_result.cylinder_id;

                if (verbose) {
                    std::cerr << "Cylinder radius: " << detected_radius << std::endl;
                    std::cout << "Cylinder color: " << color_name << std::endl;
                    //std::cout << "RGB: " << (int)color_stats.avg_r << ", " 
                            // << (int)color_stats.avg_g << ", " 
                            // << (int)color_stats.avg_b << std::endl;
                    std::cout << "Orientationally: VERTICAL" << std::endl;
                    std::cout << "Saved cylinder_id: " << saved_id
                            << (save_result.inserted ? " (new)" : " (merged)")
                            << " (saved_cylinders=" << saved_cylinders.size() << ")" << std::endl;
                }

                // NEW: Visualize which points were used
                publishVisualization(cloud_cylinder, rgb_image, now);

                // Publish marker
                marker.header.frame_id = "map";
                marker.header.stamp = now;
                marker.ns = "cylinder";
                marker.id = saved_id;

                marker.type = visualization_msgs::msg::Marker::CYLINDER;
                marker.action = visualization_msgs::msg::Marker::ADD;

                marker.pose.position.x = point_map.point.x;
                marker.pose.position.y = point_map.point.y;
                marker.pose.position.z = marker_height/2;
                marker.pose.orientation.w = 1.0;

                marker.scale.x = detected_radius * 2;
                marker.scale.y = detected_radius * 2;
                marker.scale.z = marker_height;

                // Set marker color to match detected cylinder
                marker.color.r = color_stats.avg_b / 255.0f;
                marker.color.g = color_stats.avg_g / 255.0f;
                marker.color.b = color_stats.avg_r / 255.0f;
                marker.color.a = 1.0f;

                marker.lifetime = rclcpp::Duration(0, 0);

                marker_pub->publish(marker);
            }
            

            // Publish cylinder cloud
            sensor_msgs::msg::PointCloud2 cylinder_msg;
            pcl::PCLPointCloud2::Ptr pcl_out(new pcl::PCLPointCloud2());
            pcl::toPCLPointCloud2(*cloud_cylinder, *pcl_out);
            pcl_conversions::fromPCL(*pcl_out, cylinder_msg);
            *all_cylinders += *cloud_cylinder;
            detected_cylinders++;
        }

        // Remove extracted cylinder from cloud, so the next iteration searches for a diff cylinder
        extract.setNegative(true);
        pcl::PointCloud<PointT>::Ptr temp_cloud(new pcl::PointCloud<PointT>());
        extract.filter(*temp_cloud);

        pcl::ExtractIndices<pcl::Normal> extract_normals_iter;
        extract_normals_iter.setInputCloud(remaining_normals);
        extract_normals_iter.setIndices(inliers_cylinder);
        extract_normals_iter.setNegative(true);

        pcl::PointCloud<pcl::Normal>::Ptr temp_normals(new pcl::PointCloud<pcl::Normal>());
        extract_normals_iter.filter(*temp_normals);

        remaining_cloud.swap(temp_cloud);
        remaining_normals.swap(temp_normals);
    }

    // Additional pass: detect horizontal cylinders (barrels laying on the floor)
    // This does not change the existing vertical-detection logic; it runs after it on the leftover points.
    if (!remaining_cloud->empty() && !remaining_normals->empty()) {
        const Eigen::Vector3f axis_x(1.0, 0.0, 0.0);
        const Eigen::Vector3f axis_y(0.0, 1.0, 0.0);
        const float horizontal_eps_angle = 0.8f; // same tolerance as vertical pass

        auto detectHorizontalWithAxis = [&](const Eigen::Vector3f& axis_h) {
            seg.setAxis(axis_h);
            seg.setEpsAngle(horizontal_eps_angle);

            while (true) {
                pcl::PointIndices::Ptr inliers_cylinder_h(new pcl::PointIndices);
                pcl::ModelCoefficients::Ptr coefficients_cylinder_h(new pcl::ModelCoefficients);

                seg.setInputCloud(remaining_cloud);
                seg.setInputNormals(remaining_normals);
                seg.segment(*inliers_cylinder_h, *coefficients_cylinder_h);

                if (coefficients_cylinder_h->values.empty() || inliers_cylinder_h->indices.empty()) {
                    break;
                }

                float detected_radius_h = coefficients_cylinder_h->values[6];
                int cylinder_points_count_h = inliers_cylinder_h->indices.size();

                pcl::PointCloud<PointT>::Ptr cloud_cylinder_h(new pcl::PointCloud<PointT>());
                extract.setInputCloud(remaining_cloud);
                extract.setIndices(inliers_cylinder_h);
                extract.setNegative(false);
                extract.filter(*cloud_cylinder_h);

                if (cloud_cylinder_h->empty()) {
                    break;
                }

                // For horizontal cylinders the SAC coefficients provide an arbitrary point on the infinite axis,
                // which can slide along the axis and cause ~meters of position error.
                // Use the centroid of inlier points as a stable representative position (also preserves height).
                Eigen::Vector4f centroid_h;
                pcl::compute3DCentroid(*cloud_cylinder_h, centroid_h);

                geometry_msgs::msg::PointStamped point_camera_h, point_map_h;
                visualization_msgs::msg::Marker marker_h;

                std::string toFrameRel_h = "map";
                std::string fromFrameRel_h = (*msg).header.frame_id;

                point_camera_h.header.frame_id = fromFrameRel_h;
                point_camera_h.header.stamp = now;
                point_camera_h.point.x = centroid_h[0];
                point_camera_h.point.y = centroid_h[1];
                point_camera_h.point.z = centroid_h[2];

                try {
                    auto tss_h = tf_buffer_->lookupTransform(toFrameRel_h, fromFrameRel_h, now);
                    tf2::doTransform(point_camera_h, point_map_h, tss_h);
                } catch (tf2::TransformException& ex) {
                    std::cout << ex.what() << std::endl;
                    break;
                }

                if ((std::abs(detected_radius_h - target_radius) <= error_margin) && (cylinder_points_count_h >= min_cylinder_size)) {
                    ColorStats color_stats_h = analyzeColors(cloud_cylinder_h);
                    std::string color_name_h = identifyColor(color_stats_h);

                    
                    if(color_name_h != "OTHER") {

                        // Leakage check: look for dominant colors on the floor-parallel ROI around the horizontal barrel.
                        // Uses per-point RGB “pixel” classification and compares top-2 ROI colors with barrel color.
                        const Eigen::Vector3f axis_cam_h(
                            coefficients_cylinder_h->values[3],
                            coefficients_cylinder_h->values[4],
                            coefficients_cylinder_h->values[5]);
                        const LeakageResult leak = detectLeakageHorizontal(
                            cloud_filtered,
                            centroid_h,
                            axis_cam_h,
                            color_name_h);

                        const bool is_leaking = leak.leaking;

                        if (is_leaking) {
                            say("Alert, Alert, warning, the barrel is leaking");
                        }

                        const SaveResult save_result_h = upsertCylinder(
                            now, point_map_h.point, color_name_h, color_stats_h, "HORIZONTAL", is_leaking);
                        const int saved_id_h = save_result_h.cylinder_id;

                        if (verbose) {
                            std::cerr << "[H] Cylinder radius: " << detected_radius_h << std::endl;
                            std::cout << "[H] Cylinder color: " << color_name_h << std::endl;
                            std::cout << "Orientationally: HORIZONTAL" << std::endl;
                                std::cout << "[H] Leakage: " << (is_leaking ? "true" : "false")
                                      << " (dominant=[" << leak.dominant1 << ", " << leak.dominant2
                                      << "] samples=" << leak.total_samples << ")" << std::endl;
                            std::cout << "[H] Saved cylinder_id: " << saved_id_h
                                    << (save_result_h.inserted ? " (new)" : " (merged)")
                                    << " (saved_cylinders=" << saved_cylinders.size() << ")" << std::endl;
                        }

                        publishVisualization(cloud_cylinder_h, rgb_image, now);


                        marker_h.header.frame_id = "map";
                        marker_h.header.stamp = now;
                        marker_h.ns = "cylinder";
                        marker_h.id = saved_id_h;

                        marker_h.type = visualization_msgs::msg::Marker::CYLINDER;
                        marker_h.action = visualization_msgs::msg::Marker::ADD;

                        marker_h.pose.position.x = point_map_h.point.x;
                        marker_h.pose.position.y = point_map_h.point.y;
                        marker_h.pose.position.z = marker_height / 2;
                        marker_h.pose.orientation.w = 1.0;

                        marker_h.scale.x = detected_radius_h * 2;
                        marker_h.scale.y = detected_radius_h * 2;
                        marker_h.scale.z = marker_height;

                        marker_h.color.r = color_stats_h.avg_b / 255.0f;
                        marker_h.color.g = color_stats_h.avg_g / 255.0f;
                        marker_h.color.b = color_stats_h.avg_r / 255.0f;
                        marker_h.color.a = 1.0f;

                        marker_h.lifetime = rclcpp::Duration(0, 0);
                        marker_pub->publish(marker_h);
                    }

                    sensor_msgs::msg::PointCloud2 cylinder_msg_h;
                    pcl::PCLPointCloud2::Ptr pcl_out_h(new pcl::PCLPointCloud2());
                    pcl::toPCLPointCloud2(*cloud_cylinder_h, *pcl_out_h);
                    pcl_conversions::fromPCL(*pcl_out_h, cylinder_msg_h);
                    *all_cylinders += *cloud_cylinder_h;
                    detected_cylinders++;
                }

                // Remove extracted cylinder from remaining (always), continue searching
                extract.setNegative(true);
                pcl::PointCloud<PointT>::Ptr temp_cloud_h(new pcl::PointCloud<PointT>());
                extract.filter(*temp_cloud_h);

                pcl::ExtractIndices<pcl::Normal> extract_normals_iter_h;
                extract_normals_iter_h.setInputCloud(remaining_normals);
                extract_normals_iter_h.setIndices(inliers_cylinder_h);
                extract_normals_iter_h.setNegative(true);

                pcl::PointCloud<pcl::Normal>::Ptr temp_normals_h(new pcl::PointCloud<pcl::Normal>());
                extract_normals_iter_h.filter(*temp_normals_h);

                remaining_cloud.swap(temp_cloud_h);
                remaining_normals.swap(temp_normals_h);
            }
        };

        detectHorizontalWithAxis(axis_x);
        detectHorizontalWithAxis(axis_y);

        // Restore expected axis (not strictly necessary, but keeps intent clear)
        seg.setAxis(Eigen::Vector3f(0.0, 0.0, 1.0));
        seg.setEpsAngle(0.8);
    }

    //std::cout << "Detected " << detected_cylinders << " cylinders." << std::endl;

    // publish cylinder-filtered point cloud
    if (!all_cylinders->empty()) {
        sensor_msgs::msg::PointCloud2 cylinder_msg;
        pcl::PCLPointCloud2::Ptr pcl_out(new pcl::PCLPointCloud2());

        pcl::toPCLPointCloud2(*all_cylinders, *pcl_out);
        pcl_conversions::fromPCL(*pcl_out, cylinder_msg);

        cylinder_msg.header = msg->header;  // preserve frame + timestamp
        cylinder_pub->publish(cylinder_msg);
    }    
}

/*
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    std::cout << "cylinder_segmentation started" << std::endl;

    node = rclcpp::Node::make_shared("cylinder_segmentation");

    // Print a final report on shutdown (Ctrl+C / rclcpp::shutdown)
    rclcpp::on_shutdown([]() {
        printSavedCylindersReport();
    });

    // create subscriber
    node->declare_parameter<std::string>("topic_pointcloud_in", "/oakd/rgb/preview/depth/points");
    std::string param_topic_pointcloud_in = node->get_parameter("topic_pointcloud_in").as_string();
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription = node->create_subscription<sensor_msgs::msg::PointCloud2>(param_topic_pointcloud_in, 10, &cloud_cb);

    // setup tf listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // create publishers
    planes_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_point_cloud", 1);
    cylinder_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("cylinder_point_cloud", 1);
    marker_pub = node->create_publisher<visualization_msgs::msg::Marker>("cylinder_markers", 1);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}*/

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    std::cout << "cylinder_segmentation started" << std::endl;

    node = rclcpp::Node::make_shared("cylinder_segmentation");

    // Print a final report on shutdown (Ctrl+C / rclcpp::shutdown)
    rclcpp::on_shutdown([]() {
        printSavedCylindersReport();
    });

    // create subscriber
    node->declare_parameter<std::string>("topic_pointcloud_in", "/oakd/rgb/preview/depth/points");
    std::string param_topic_pointcloud_in = node->get_parameter("topic_pointcloud_in").as_string();
    
    // CHANGED: Use message_filters to sync point cloud with RGB image
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub(
        node.get(), param_topic_pointcloud_in);
    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub(
        node.get(), "/oakd/rgb/preview/image_raw");
    message_filters::TimeSynchronizer<sensor_msgs::msg::PointCloud2, 
        sensor_msgs::msg::Image> sync(cloud_sub, image_sub, 10);
    sync.registerCallback(&cloud_cb);

    // Subscribe to camera_info to get intrinsics
    auto camera_info_sub = node->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/oakd/rgb/preview/camera_info", 10, 
        [](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
            fx = msg->k[0];
            fy = msg->k[4];
            cx = msg->k[2];
            cy = msg->k[5];
            /*
            if (verbose) {
                std::cerr << "Updated camera intrinsics: fx=" << fx << " fy=" << fy 
                          << " cx=" << cx << " cy=" << cy << std::endl;
            }*/
        });

    // setup tf listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // create publishers
    planes_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_point_cloud", 1);
    cylinder_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("cylinder_point_cloud", 1);
    marker_pub = node->create_publisher<visualization_msgs::msg::Marker>("cylinder_markers", 1);
    viz_image_pub = node->create_publisher<sensor_msgs::msg::Image>("cylinder_viz_image", 1);  // NEW
    barrels_results_pub = node->create_publisher<rins_interfaces::msg::BarrelsResults>("barrels_results", 10);


    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
