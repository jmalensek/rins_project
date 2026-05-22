#!/usr/bin/env python3

"""
task_node.py

ROS2 node ki:
  1. Sprejme ime osebe + gender prek /current_person (JSON: {"person": "Ime", "gender": "female"})
  2. Izgovori pozdrav in vpraša po nalogi
  3. Sprejme nalogo prek /task_input (std_msgs/String) – zamenjamo z mikrofonom ko bo na voljo
  4. Shrani {ime: task} in objavi na /person_task

Zamenjava za mikrofon: samo funkcijo _listen_for_task() zamenjamo z STT klicem.
"""

# ros2 topic pub /current_person std_msgs/msg/String "{data: '{\"person\": \"Jeff\", \"gender\": \"female\"}'}" --once
# ros2 topic pub /task_input std_msgs/String \"data: 'barrels'\" --once"

from datetime import datetime
import json
from pathlib import Path
import subprocess
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


TASK_KEYWORDS = {
    "barrels":       ["barrel", "barrels", "inspection", "barrel inspection"],
    "rings":         ["ring",   "rings",   "count rings", "counting"],
    "tiles":         ["tiles", "anomaly"],
}



def normalize_task(raw: str) -> str:
    text = raw.strip().lower()
    for task, keywords in TASK_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return task
    return raw.strip()


class TaskNode(Node):
    def __init__(self):
        super().__init__("task_node")

        # Trenutna oseba
        self.current_person: str | None = None
        self.current_gender: str | None = None

        # Shramba: {ime: task}
        self.task_memory: dict[str, str] = {}
        self._task_memory_lock = threading.Lock()

        # Čaka na nalogo (dialog poteka v ločenem threadu)
        self._task_input: str | None = None
        self._waiting_for_task = False
        self._task_input_event = threading.Event()

        self.task_pub = self.create_publisher(String, "/person_task", 10)
        self.create_subscription(String, "/current_person", self._person_cb, 10)
        # /task_input = simulacija mikrofona; zamenjaj z STT ko bo na voljo
        self.create_subscription(String, "/task_input", self._task_input_cb, 10)

        self.create_service(Trigger, "/get_tasks", self._get_tasks_srv)

        self.get_logger().info("Task_manager starting.")



    #  Say response text
    def say(self, text: str):
        """System TTS z spd-say."""
        self.get_logger().info(f"{text}")
        try:
            subprocess.run(
                ["spd-say", "-r", "-60", "-p", "-55", "-t", "male3", text],
                check=False
            )
        except FileNotFoundError:
            print(f"spd-say not found. Would say: {text}")


    # STT placeholder: simulacija mikrofona, preko topic
    def _listen_for_task(self) -> str | None:
        self._task_input    = None
        self._waiting_for_task = True
        self._task_input_event.clear()

        self.get_logger().info("Waiting on /task_input ...")

        # IMPORTANT: do not call rclpy.spin_once() here.
        # The node is already spinning in the main thread (rclpy.spin(node)).
        # We just wait for the subscriber callback to signal new input.
        while rclpy.ok():
            if self._task_input_event.wait(timeout=0.1):
                break

        if not rclpy.ok():
            self._waiting_for_task = False
            return None

        result = self._task_input
        self._task_input = None
        self._waiting_for_task = False
        self._task_input_event.clear()
        return result



    
    # make dialog, ask, receive tast (if female ask again)
    def _run_dialog(self, name: str, gender: str):
        """
        Celoten dialog z osebo. Teče v ločenem threadu.
        """
        if(name == "Jeff"): 
            self.report()
            return

        # Pozdrav + vprašanje
        self.say(
            f"Hello {name}. You are a {gender}. "
            f"What task should I perform? "
        )

        # Poslušaj odgovor
        raw = self._listen_for_task()
        if raw is None:
            self.get_logger().warn(f"No task received for {name}.")
            return

        self.get_logger().info(f"Received: '{raw}')")

        # If female ASK AGAIN
        if gender == "female":
            self.say("Are you sure?")
            raw = self._listen_for_task()
            if raw is None:
                self.get_logger().warn(f"No task received for {name}.")
                return
            
        task = normalize_task(raw)
        self.get_logger().info(f"Received: '{raw}')")
        self.get_logger().info(f"Task for {name}: {task} (raw: '{raw}')")

        # Shrani
        with self._task_memory_lock:
            self.task_memory[name] = task

        # Potrditev
        self.say(f"Understood. I will perform: {task.replace('_', ' ')}.")

        if name.strip().lower() == "jeff":
            pdf_path = self._generate_tasks_pdf()
            if pdf_path is not None:
                self.get_logger().info(f"Generated PDF: {pdf_path}")


    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _person_cb(self, msg: String):
        """Sprejme trenutno osebo in sproži dialog v ločenem threadu."""
        raw = msg.data.strip()

        person = None
        gender = "unknown"

        if raw.startswith("{"):
            try:
                payload = json.loads(raw)
                person  = (payload.get("person") or payload.get("name") or "").strip() or None
                gender  = (payload.get("gender") or "unknown").strip()
            except Exception:
                person = raw
        else:
            person = raw

        if not person:
            return

        # Če dialog že teče, ignoriraj
        if self._waiting_for_task:
            self.get_logger().warn(f"Dialog že teče, ignoriram novega: {person}")
            return

        self.current_person = person
        self.current_gender = gender
        self.get_logger().info(f"New person: {person} ({gender})")

        # Zaženi dialog v threadu da ne blokiramo spin loopa
        t = threading.Thread(target=self._run_dialog, args=(person, gender), daemon=True)
        t.start()

    def _task_input_cb(self, msg: String):
        """Sprejme simuliran govorni vnos."""
        if self._waiting_for_task:
            self._task_input = msg.data.strip()
            self._task_input_event.set()
            #self.get_logger().info(f"Prejeto: '{self._task_input}'")
        else:
            self.get_logger().warn("Prejeto /task_input ampak dialog ne čaka – ignoriram.")

    def _get_tasks_srv(self, request, response):
        response.success = True
        with self._task_memory_lock:
            snapshot = dict(self.task_memory)
        response.message = json.dumps(snapshot, indent=2)
        return response

    def get_task_for_person(self, name: str) -> str | None:
        with self._task_memory_lock:
            return self.task_memory.get(name)


    #-------------------------------------------------
    def report(self):
        self.say("Hello Sir, I am generating your report.")
        self._generate_tasks_pdf()

    #generate pdf
    def _generate_tasks_pdf(self) -> Path | None:
        """Generate a PDF with all stored names and tasks into ./pdf/ directory."""

        with self._task_memory_lock:
            snapshot = dict(self.task_memory)

        out_dir = Path.cwd() / "pdf"
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"tasks_{timestamp}.pdf"

        c = canvas.Canvas(str(out_path), pagesize=A4)
        width, height = A4

        left_margin = 50
        top_margin = 50
        y = height - top_margin

        c.setFont("Helvetica-Bold", 14)
        c.drawString(left_margin, y, "Tasks summary")
        y -= 24

        c.setFont("Helvetica", 10)
        c.drawString(left_margin, y, f"Generated: {datetime.now().isoformat(timespec='seconds')}")
        y -= 20

        if not snapshot:
            c.drawString(left_margin, y, "No tasks recorded.")
            c.save()
            return out_path

        for person_name in sorted(snapshot.keys(), key=lambda s: s.lower()):
            task = snapshot.get(person_name, "")
            line = f"{person_name}: {task}"
            if y < 60:
                c.showPage()
                y = height - top_margin
                c.setFont("Helvetica", 10)
            c.drawString(left_margin, y, line)
            y -= 14

        c.save()
        return out_path


def main(args=None):
    rclpy.init(args=args)
    node = TaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()