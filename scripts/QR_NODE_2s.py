#!/usr/bin/env python3

# DEMO DOKLER NE NAREDIM SPEECH SYNTESHIS

"""
qr_task_node.py

ROS2 node ki:
    1. Sprejme ime osebe + gender prek topica /current_person  (std_msgs/String, JSON ali plain)
  2. Bere QR kodo iz kamere /camera/image_raw
  3. Ko dekodira QR, poveže ime + nalogo in jo shrani
    4. Objavi rezultat na /person_task  (std_msgs/String, samo task)

Zapomnjena opravila so dostopna kadarkoli prek servica /get_task (ime → naloga).

Namestitev:
    pip install pyzbar --break-system-packages
    sudo apt install libzbar0

Uporaba:
    ros2 run my_robot qr_task_node
    
    # Nastavi trenutno osebo:
    ros2 topic pub /current_person std_msgs/String "data: 'Luka'" --once
    
    # Preberi shranjene naloge:
    ros2 service call /get_task std_srvs/srv/... 
    # (ali poslušaj /person_task topic)
"""

import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from cv_bridge import CvBridge
import cv2

try:
    from pyzbar.pyzbar import decode as qr_decode
    PYZBAR_OK = True
except ImportError:
    PYZBAR_OK = False
    print("⚠ pyzbar ni nameščen: pip install pyzbar --break-system-packages")
    print("⚠ sudo apt install libzbar0")


# Veljavne naloge – za normalizacijo QR vsebine
TASK_KEYWORDS = {
    "barrels":       ["barrel", "barrels", "inspection", "barrel inspection"],
    "rings":         ["ring",   "rings",   "count rings", "counting"],
    "anomaly_red":   ["anomaly red",  "red cell",   "red working cell",   "red"],
    "anomaly_green": ["anomaly green","green cell",  "green working cell", "green"],
    "nothing":       ["nothing", "none", "no task"],
}


def normalize_task(raw: str) -> str:
    """Pretvori surovo QR besedilo v standardizirano ime naloge."""
    text = raw.strip().lower()
    for task, keywords in TASK_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return task
    # Če ne prepoznamo – vrnemo surovo vsebino
    return raw.strip()


class QRTaskNode(Node):
    def __init__(self):
        super().__init__("qr_task_node")

        if not PYZBAR_OK:
            self.get_logger().error("pyzbar ni nameščen! Node ne bo deloval.")
            return

        self.bridge = CvBridge()

        # Trenutna oseba (nastavljena od zunaj) + pripadajoč gender
        self.current_person: str | None = None
        self.current_gender: str | None = None

        # Shramba: {ime_osebe: {task, gender}}
        self.task_memory: dict[str, dict[str, str]] = {}

        # Zaščita pred večkratnim branjem iste QR kode
        self.last_qr_content: str | None = None
        self.qr_cooldown_sec = 3.0
        self.last_qr_time    = self.get_clock().now()

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            String, "/current_person",
            self._person_cb, 10
        )
        self.create_subscription(
            Image, "/camera/image_raw",
            self._image_cb, 10
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.task_pub = self.create_publisher(String, "/person_task", 10)

        # ── Service: vrne nalogo za določeno osebo ────────────────────────────
        # Klic: ros2 service call /get_tasks std_srvs/srv/Trigger
        # Vrne JSON z vsemi shranjenimi nalogami
        self.create_service(Trigger, "/get_tasks", self._get_tasks_srv)

        self.get_logger().info("✓ QR Task Node zagnan")
        self.get_logger().info("  Pošlji ime osebe: ros2 topic pub /current_person std_msgs/String \"data: 'Ime'\" --once")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _person_cb(self, msg: String):
        """Nastavi trenutno aktivno osebo.

        Pričakovan format (std_msgs/String):
          - JSON: {"person": "Ime", "gender": "M/F/?"}
          - ali plain string: "Ime" (gender ostane None)
        """
        raw = msg.data.strip()
        person = None
        gender = None

        if raw.startswith("{"):
            try:
                payload = json.loads(raw)
                person = (payload.get("person") or payload.get("name") or "").strip() or None
                gender = (payload.get("gender") or "").strip() or None
            except Exception:
                person = raw
        else:
            person = raw

        self.current_person = person
        self.current_gender = gender
        self.last_qr_content = None  # reset cooldown za novo osebo

        if self.current_person:
            self.get_logger().info(f"👤 Trenutna oseba: {self.current_person} (gender={self.current_gender})")

    def _image_cb(self, msg: Image):
        """Procesira vsak frame – išče QR kodo."""
        if self.current_person is None:
            return  # Ne vemo za koga beremo QR

        # Throttle: ne procesiramo prepogosto
        now     = self.get_clock().now()
        elapsed = (now - self.last_qr_time).nanoseconds / 1e9
        if elapsed < 0.5:  # max 2 fps za QR detekcijo
            return

        self.last_qr_time = now

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self._scan_qr(frame)

    def _scan_qr(self, frame):
        """Poskusi dekodirati QR kodo v sliki."""
        # Pretvori v grayscale za boljšo detekcijo
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        decoded = qr_decode(gray)

        if not decoded:
            return

        for obj in decoded:
            raw_content = obj.data.decode("utf-8").strip()

            # Cooldown – ignoriraj če smo to že prebrali
            if raw_content == self.last_qr_content:
                return

            task = normalize_task(raw_content)

            self.get_logger().info(f"📦 QR prebrana: '{raw_content}' → naloga: '{task}'")

            # Shrani
            self.task_memory[self.current_person] = {
                "task": task,
                "gender": self.current_gender or "?",
            }
            self.last_qr_content = raw_content

            # Objavi
            out      = String()
            out.data = task
            self.task_pub.publish(out)

            self.get_logger().info(
                f"✅ Shranjeno: {self.current_person} → {task}\n"
                f"   Vse naloge: {self.task_memory}"
            )

            # Reset trenutne osebe (pripravi se na naslednjo)
            self.current_person = None
            self.current_gender = None
            break  # dovolj ena QR koda na frame

    def _get_tasks_srv(self, request, response):
        """Service ki vrne vse shranjene naloge kot JSON."""
        response.success = True
        response.message = json.dumps(self.task_memory, indent=2)
        self.get_logger().info(f"📋 Zahtevano: vse naloge → {self.task_memory}")
        return response

    def get_task_for_person(self, name: str) -> dict[str, str] | None:
        """Pomožna metoda za direktni dostop znotraj kode."""
        return self.task_memory.get(name)


def main(args=None):
    rclpy.init(args=args)
    node = QRTaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()