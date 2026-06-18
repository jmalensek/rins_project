#!/usr/bin/env python3

'''
Poenostavljen red_green_cell_detection node z integriranim RobotCommanderjem.
Ko vozlišče prejme ukaz iz task_managerja in informacijo o barvi, se zaporedno
odpelje do hardkodiranega začetka, preveri barvo tal, pripravi tile detection
in se odpelje do konca celice.
'''

import math
import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

# Uvoz tvojega RobotCommanderja iz iste mape/paketa
from robot_commander import RobotCommander 

from robot_explorer import RobotExplorer


class RedGreenCellDetection(Node):
    def __init__(self):
        super().__init__('red_green_cell_detection')

        # Ker bomo uporabljali blokirajoče klicne funkcije (go_to_pose), 
        # potrebujemo ReentrantCallbackGroup, da se lahko drugi callbacki (npr. kamera) 
        # izvajajo vzporedno med vožnjo.
        self.callback_group = ReentrantCallbackGroup()

        # Hardkodirane koordinate: (x, y, yaw v radianih)
        # Nastavi dejanske končne in začetne koordinatne točke za tvoje okolje
        self.COORDINATES = {
            'cell 1': {
                'start': (-1.57, -4.8),
                'end': (0.59, -4.8)
            },
            'cell 2': {
                'start': (-4.76, 0.343),
                'end': (-4.76, -2.749)
            }
        }

        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_topic', '/oakd/rgb/preview/image_raw'),
                ('task_start_topic', '/task_manager/task_started'),
                ('tile_detection_start_topic', '/tile_detection/start'),
                ('tile_detection_stop_topic', '/tile_detection/stop'),
            ],
        )

        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.task_start_topic = self.get_parameter('task_start_topic').get_parameter_value().string_value
        self.tile_detection_start_topic = self.get_parameter('tile_detection_start_topic').get_parameter_value().string_value
        self.tile_detection_stop_topic = self.get_parameter('tile_detection_stop_topic').get_parameter_value().string_value

        # Inicializacija Robot Commanderja
        self.commander = RobotCommander(self)

        self.explo = RobotExplorer(self)

        # Interno stanje
        self.task_active = False
        self.color_of_the_cell = None
        self.latest_frame = None
        self.bridge = CvBridge()

        # Naročniki (Kamera mora biti v callback skupini, da konstantno osvežuje sliko!)
        self.camera_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.camera_callback,
            qos_profile_sensor_data,
            callback_group=self.callback_group
        )

        self.task_start_sub = self.create_subscription(
            Bool,
            self.task_start_topic,
            self.on_task_started,
            10,
            callback_group=self.callback_group
        )

        self.color_cell_sub = self.create_subscription(
            String,
            '/task_manager/color_of_the_cell',
            self.color_of_the_cell_callback,
            10,
            callback_group=self.callback_group
        )

        # Izdajatelji
        self.tile_start_pub = self.create_publisher(Bool, self.tile_detection_start_topic, 10)
        self.tile_stop_pub = self.create_publisher(Bool, self.tile_detection_stop_topic, 10)
        self.arm_mover_pub = self.create_publisher(String, '/arm_command', 10)

        self.get_logger().info('RedGreenCellDetection node z RobotCommanderjem uspešno zagnan.')

    def camera_callback(self, msg: Image):
        """Kamera konstantno teče v ozadju in shranjuje najnovejši frame."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f'CV Bridge napaka: {exc}')

    def on_task_started(self, msg: Bool):
        if not msg.data:
            return
        self.task_active = True
        self.get_logger().info('Signal za začetek naloge prejet.')
        self._check_and_execute_mission()

    def color_of_the_cell_callback(self, msg: String):
        self.color_of_the_cell = msg.data.lower().strip()
        self.get_logger().info(f"Prejeta barva celice iz task_managerja: {self.color_of_the_cell}")
        self._check_and_execute_mission()

    def _check_and_execute_mission(self):
        """
        Preveri, če imamo hkrati aktiven task in znano barvo. 
        Če sta pogoja izpolnjena, zaporedno izvede celotno misijo.
        """
        if self.task_active and (self.color_of_the_cell in ['red', 'green']):
            # Takoj ponastavimo pogoje, da se misija ne bi sprožila dvakrat
            self.task_active = False 

            color = self.color_of_the_cell
            
            for cell in self.COORDINATES:

                start_coords = cell['start']
                end_coords = cell['end']

                # 1. Premik do začetne točke celice
                self.get_logger().info(f"Potujem proti ZAČETKU celice: {start_coords}")
                # go_to_pose interno čaka na uspeh/konec premika (blokirajoč klic)

                ayaw = self.explo.compute_absolute_yaw(start_coords, end_coords)

                self.explo.go_to_pose(start_coords[0], start_coords[1], ayaw)

                distance = self.explo.compute_distance(start_coords, end_coords)

                # Počakamo sekundo, da se robot povsem umiri pred analizo slike
                time.sleep(1.0)

                # 2. Preverjanje barve tal
                self.get_logger().info("Prispeli na štart. Preverjam dejansko barvo tal...")
                if self.check_floor_color(color):
                    self.get_logger().info(f"Barva tal potrjena ({color}). Pripravljam detekcijo.")
                    
                    # Priprava senzorjev in roke
                    self.raise_top_camera()
                    self.start_tile_detection()

                    
                    # 3. Premik do končne točke celice
                    self.get_logger().info(f"Potujem proti KONCU celice: {end_coords}")
                    #self.commander.go_to_pose(end_coords[0], end_coords[1], end_coords[2])
                    self.explo.move_straight_odom(distance=distance)
                    
                    # Zaključek misije
                    self.get_logger().info("Prispeli na konec celice. Zaustavljam detekcijo ploščic.")
                    self.stop_tile_detection()
                    break
                else:
                    continue

    def check_floor_color(self, expected_color: str) -> bool:
        """Pregleda spodnji rob slike in potrdi, če barva ustreza."""
        if self.latest_frame is None:
            self.get_logger().warning("Ni slike iz kamere za preverjanje barve tal!")
            return False

        h, w = self.latest_frame.shape[:2]
        # Vzamemo spodnji pas slike (tla neposredno pred robotom)
        roi = self.latest_frame[h-100:h, :]

        # LAB barvni prostor za robustnejše filtriranje
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        A = lab[:, :, 1] - 128.0
        B = lab[:, :, 2] - 128.0
        chroma = np.sqrt(A**2 + B**2)
        hue = np.degrees(np.arctan2(B, A)) % 360

        if expected_color == 'red':
            mask = (chroma >= 15) & ((hue < 42) | (hue >= 330))
        elif expected_color == 'green':
            mask = (chroma >= 15) & (hue >= 105) & (hue < 140)
        else:
            return False

        pixel_count = np.count_nonzero(mask)
        self.get_logger().info(f"Zaznanih pikslov barve '{expected_color}': {pixel_count}")
        
        # Prag za potrditev (če je vsaj 500 slikovnih pik ustrezne barve)
        return pixel_count > 500

    def raise_top_camera(self):
        command = 'look_at_belt_right'
        self.arm_mover_pub.publish(String(data=command))
        self.get_logger().info('Poslan ukaz za roko: look_at_belt_right')
        time.sleep(5)

    def start_tile_detection(self):
        self.tile_start_pub.publish(Bool(data=True))
        self.get_logger().info('Tile detection START signal poslan.')

    def stop_tile_detection(self):
        self.tile_stop_pub.publish(Bool(data=True))
        self.get_logger().info('Tile detection STOP signal poslan.')
        
        # Vrne roko/kamero nazaj v začetni položaj
        command = 'look_for_qr'
        self.arm_mover_pub.publish(String(data=command))
        self.get_logger().info('Poslan zaključni ukaz za roko: look_for_qr')

        self.get_logger().info('Stopping red_green_cell_detection node.')
        try:
            self.destroy_node()
        except Exception as exc:
            self.get_logger().error(f'Error destroying node: {exc}')
        rclpy.shutdown()


def main():
    rclpy.init()
    node = RedGreenCellDetection()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            try:
                node.destroy_node()
            except Exception:
                pass
            rclpy.shutdown()


if __name__ == '__main__':
    main()