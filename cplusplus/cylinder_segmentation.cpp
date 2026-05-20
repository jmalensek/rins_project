#include <iostream>
#include <pcl/ModelCoefficients.h>
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

/*
ZA DODAT:
- ko detecta barrel, naredi color recognition
- vse barelle shranjuje - se pravi njegov id, barvo, orientation, lokacija, ali leaka, ali je bil published?
- to potem publisha na nek topic
- za preverit kako se orientacijo zamenja - se pravi X ali Y axis

*/

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr planes_pub;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cylinder_pub;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub;

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
bool verbose = true; // debug outout

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
int max_detected_cylinders = 3;
int min_cylinder_size = 500;

struct ColorStats {
    float avg_r, avg_g, avg_b;
    float std_r, std_g, std_b;
};

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
    
    if (r > 0.7 && g < 0.3 && b < 0.3) return "RED";
    if (g > 0.7 && r < 0.3 && b < 0.3) return "GREEN";
    if (b > 0.7 && r < 0.3 && g < 0.3) return "BLUE";
    if (r > 0.6 && g > 0.6 && b < 0.3) return "YELLOW";
    return "OTHER";
}

void colorizePointCloud(pcl::PointCloud<PointT>::Ptr& cloud, const cv::Mat& rgb_image) {
    int colored_count = 0;
    int out_of_bounds = 0;

    for (auto& point : cloud->points) {
        if (point.z <= 0) {
            point.r = point.g = point.b = 0;
            continue;
        }
        
        int u = (int)(fx * point.x / point.z + cx);
        int v = (int)(fy * point.y / point.z + cy);
        
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


void visualizeDetectedPoints(
    const pcl::PointCloud<PointT>::Ptr& cloud_cylinder,
    const cv::Mat& rgb_image,
    const std::string& output_filename) {
    
    cv::Mat viz = rgb_image.clone();
    
    for (const auto& point : cloud_cylinder->points) {
        if (point.z <= 0) continue;
        
        int u = (int)(fx * point.x / point.z + cx);
        int v = (int)(fy * point.y / point.z + cy);
        
        if (u >= 0 && u < viz.cols && v >= 0 && v < viz.rows) {
            // Draw a small circle at each point
            cv::circle(viz, cv::Point(u, v), 2, cv::Scalar(0, 255, 0), -1);
        }
    }
    
    // Save the image
    cv::imwrite(output_filename, viz);
    std::cout << "Saved visualization to: " << output_filename << std::endl;
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

    int marker_id = 0;
    int detected_cylinders = 0;

    // each loop iteration: finds the best-fitting cylinder in the remaining cloud
    // extracts cylinder points and stores them
    // removes those points from remaining_cloud
    // repeats up to 3 times or until no more cylinders found
    while (detected_cylinders <= max_detected_cylinders) {

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

            if (verbose) {
                std::cerr << "Cylinder radius: " << detected_radius << std::endl;
                std::cout << "Cylinder_points_count: " << cylinder_points_count << std::endl;
                std::cout << "Cylinder color: " << color_name << std::endl;
                std::cout << "RGB: " << (int)color_stats.avg_r << ", " 
                            << (int)color_stats.avg_g << ", " 
                            << (int)color_stats.avg_b << std::endl;
            }

            // NEW: Visualize which points were used
            static int cylinder_count = 0;
            std::string output_file = "/tmp/cylinder_" + std::to_string(cylinder_count) + ".jpg";
            visualizeDetectedPoints(cloud_cylinder, rgb_image, output_file);
            cylinder_count++;

            // Publish marker
            marker.header.frame_id = "map";
            marker.header.stamp = now;
            marker.ns = "cylinder";
            marker.id = marker_id++;

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
            marker.color.r = color_stats.avg_r / 255.0f;
            marker.color.g = color_stats.avg_g / 255.0f;
            marker.color.b = color_stats.avg_b / 255.0f;
            marker.color.a = 1.0f;

            marker.lifetime = rclcpp::Duration(0, 0);

            marker_pub->publish(marker);

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

    std::cout << "Detected " << detected_cylinders << " cylinders." << std::endl;

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

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
