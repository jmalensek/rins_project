#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import time
import math
import subprocess
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from robot_commander import RobotCommander


class greet_people(Node):

    def __init__(self, rc):
        super().__init__('greet_people')

        self.rc = rc

        # Subscribe to face detections
        self.create_subscription(PoseStamped, '/face_detections', self.detections_callback, 10)

        # Start greeting only after two /finished triggers
        self.create_subscription(Bool, '/finished', self.finished_callback, 10)

        # za robot speaking
        self.speak_pub = self.create_publisher(String, "/speak", 10)

        # location markers
        self.location_pub = self.create_publisher(Marker, '/location_marker', 10)


        self.queue = []
        self.greet_started = False
        self.processing = False
        self.finished = False

        self.n_finished = 1
        self.finished_true_count = 0

        self.n_persons = 6
        self.stop_distance = 0.65

        self.get_logger().info("Node started. Waiting for people detections...")

        self.text = ["Smile", "Hair", "Eyes", "Hairstyle", "Style", "Everything"]
        self.person_ix = 0



    # make sure rings and faces are both finished
    def finished_callback(self, msg: Bool):
        self.finished_true_count += 1
        self.get_logger().info(f"Received /finished=True trigger ({self.finished_true_count}/{self.n_finished})")
        self.maybe_start_greeting()


    # save people detections coords
    def detections_callback(self, msg):
        # Receive a PoseStamped that already contains the exact coordinates to visit.
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        qx = float(msg.pose.orientation.x)
        qy = float(msg.pose.orientation.y)
        qz = float(msg.pose.orientation.z)
        qw = float(msg.pose.orientation.w)
        self.get_logger().info(
            f"Received face detection goal at ({x:.2f}, {y:.2f}) with orientation "
            f"[{qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f}]"
        )
        # Store the full PoseStamped so it can be used as an exact goal later
        self.queue.append(msg)
        self.maybe_start_greeting()


    #  check if everything is ok to start
    def maybe_start_greeting(self):
        if self.greet_started:
            return
        if len(self.queue) < self.n_persons or self.finished_true_count < self.n_finished:
            return
        if not self.rc.initial_pose_received:
            self.get_logger().warn(
                f"Have {len(self.queue)} detections and {self.finished_true_count} /finished triggers; waiting for first amcl_pose before greeting")
            return

        self.greet_started = True


    def process_pending_greetings(self):
        if self.finished or not self.greet_started or self.processing:
            return

        self.processing = True
        try:
            self.greet()
            self.finished = True
            self.get_logger().info(f"Visited all {self.n_persons} faces. Shutting down.")
            rclpy.shutdown()
        finally:
            self.processing = False


    def greet(self):
        # Use the first N PoseStamped messages as exact goals
        for pose_msg in self.queue[:self.n_persons]:
            self.greet_person(pose_msg)


    def greet_person(self, target_pose: PoseStamped):
        # Accept the incoming PoseStamped as the exact goal to visit.
        # The pose orientation is preserved so the robot faces the intended direction.
        if not self.rc.initial_pose_received:
            self.get_logger().warn("Robot pose not available yet (waiting for amcl_pose)")
            return

        # Copy the incoming pose so we preserve both position and orientation exactly.
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = target_pose.header.frame_id if target_pose.header.frame_id else 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose = target_pose.pose

        self.get_logger().info(
            f"Going to exact goal at ({goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f}) "
            f"with orientation [{goal_pose.pose.orientation.x:.3f}, {goal_pose.pose.orientation.y:.3f}, "
            f"{goal_pose.pose.orientation.z:.3f}, {goal_pose.pose.orientation.w:.3f}]"
        )

        self.publish_location_marker(target_pose)
        # Send the exact goal pose to RobotCommander, including orientation.
        self.rc.goToPose(goal_pose)
        
        # Wait for completion
        while not self.rc.isTaskComplete():
            time.sleep(0.5)

        # say something
        self.speak(f"Hello person, I like your {self.text[self.person_ix]}!")
        time.sleep(4)
        self.person_ix += 1


    def say(self, text):
        """Use system spd-say for text-to-speech"""
        try:
            subprocess.run(["spd-say", "-r", "-60", "-p", "-55", "-t", "male3", text], check=False)
        except FileNotFoundError:
            print(f"spd-say not found. Would say: {text}")

    
    def speak(self, text):
        msg = String()
        msg.data = text
        self.speak_pub.publish(msg)
        self.get_logger().info(f'Said: "{text}"')

    
    def publish_location_marker(self, pose: PoseStamped):
        marker = Marker()
        
        # Nastavitev glave sporočila
        marker.header.frame_id = pose.header.frame_id if pose.header.frame_id else "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        
        # Nastavitev tipa markerja (npr. ARROW (puščica), SPHERE (sfera), CUBE (kocka))
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Pozicija in orientacija (kopirana iz vašega pose)
        marker.pose = pose.pose
        
        # Velikost markerja v metrih (x, y, z)
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        
        marker.color.r = 0.0
        marker.color.g = 1.0  # Polna zelena barva
        marker.color.b = 0.0
        marker.color.a = 1.0  # Neprosojno (0.0 je popolnoma prosojno)
        
        # ID markerja (če objavljate več markerjev, dajte vsakemu svoj ID, sicer se prepišejo)
        marker.id = 0
        
        # Objava na topiko
        self.location_pub.publish(marker)

        self.get_logger().info(
            f"Published GREEN location marker at ({marker.pose.position.x:.2f}, "
            f"{marker.pose.position.y:.2f})"
        )

        


def main():
    rclpy.init(args=None)
    rc = RobotCommander()

    rc.waitUntilNav2Active()

    """
    while rc.is_docked is None:
        rclpy.spin_once(rc, timeout_sec=0.5)

    if rc.is_docked:
        rc.undock()
    """

    node = greet_people(rc)

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        rclpy.spin_once(rc, timeout_sec=0.1)
        node.process_pending_greetings()

    node.destroy_node()
    rc.destroyNode()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
