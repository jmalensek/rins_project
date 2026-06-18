#!/usr/bin/env python3

# pip install vosk --break-system-packages
# pip install reportlab --break-system-packages

# V RINS PROJECT DOWNLOAD VOSK MODEL (https://alphacephei.com/vosk/models) v src/rins_project/vosk-model-en-us-0.22

"""
task_node.py

ROS2 node ki:
  1. Sprejme ime osebe + gender prek /current_person (JSON: {"person": "Ime", "gender": "female"})
  2. Izgovori pozdrav in vpraša po nalogi
  3. Sprejme nalogo prek /task_input (std_msgs/String) – zamenjamo z mikrofonom ko bo na voljo
  4. Shrani {ime: task} in objavi na /person_task

Zamenjava za mikrofon: samo funkcijo _listen_for_task() zamenjamo z STT klicem.
"""

# Example usage:
#   ros2 topic pub --once /current_person std_msgs/msg/String "{data: '{\"person\": \"Jeff\", \"gender\": \"female\"}'}"
#   ros2 topic pub --once /task_input std_msgs/msg/String "{data: 'barrels'}"
#   ros2 topic pub --once /task_input std_msgs/msg/String "{data: 'Detect anomalies in the green cell'}"

from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import threading
import time

import json as _json

try:
    from rapidfuzz import fuzz as _fuzz  # type: ignore[import-not-found]
except Exception:
    _fuzz = None  # type: ignore[assignment]

try:
    from vosk import Model, KaldiRecognizer  # type: ignore[import-not-found]
except Exception:
    Model = None  # type: ignore[assignment]
    KaldiRecognizer = None  # type: ignore[assignment]

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger

from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)


TASK_KEYWORDS = {
    "Barrels inspection": ["barrel", "barrels", "cylinder", "cylinders", "inspection"],
    "Counting rings":     ["ring", "rings", "count rings", "circles"],
    "Anomaly detection":    ["tile", "tiles", "cell", "anomaly", "anomalies"],
}
MIN_TASK_SCORE = 0.45


def classify_task(raw: str) -> tuple[str | None, float]:
    """Return (task, confidence_score in [0..1]) using rapidfuzz token matching."""
    text = raw.strip().lower()
    tokens = re.findall(r"[a-z]+", text)

    if not text:
        return (None, 0.0)

    # Exact keyword substring match => confidence 1.0
    for task, keywords in TASK_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return (task, 1.0)

    # Fuzzy token match with rapidfuzz (or fallback to simple ratio without it)
    scores: dict[str, float] = {}
    for task, keywords in TASK_KEYWORDS.items():
        best = 0.0
        for tok in tokens:
            for kw in keywords:
                kw_tokens = re.findall(r"[a-z]+", kw)
                for kw_tok in kw_tokens:
                    if _fuzz is not None:
                        ratio = _fuzz.ratio(tok, kw_tok) / 100.0
                    else:
                        # simple overlap fallback if rapidfuzz not installed
                        common = sum((tok.count(c) for c in set(kw_tok)))
                        ratio = 2 * common / (len(tok) + len(kw_tok)) if (tok and kw_tok) else 0.0
                    best = max(best, ratio)
        scores[task] = best

    best_task = max(scores, key=scores.get)
    return best_task, scores[best_task]


class TaskNode(Node):
    def __init__(self):
        super().__init__("task_node")

        #self.listen = self._listen_for_task_voice
        self.listen = self._listen_for_task

        # Optional offline STT configuration (Vosk)
        self.model_path = "/home/lea/colcon_ws/vosk-model-en-us-0.22"
        #self.model_path = "./src/rins_project/vosk-model-en-us-0.22"
        self.declare_parameter("mic_device_index", -1)

        self._vosk_model = None
        if Model is None:
            self.get_logger().error("Vosk ni na voljo (import failed). Voice STT disabled.")
        else:
            try:
                self.get_logger().info(f"Loading Vosk model once: {self.model_path}")
                self._vosk_model = Model(self.model_path)
            except Exception as exc:
                self.get_logger().error(f"Failed to load Vosk model from '{self.model_path}': {exc}")
                self._vosk_model = None

        self._arecord_proc: subprocess.Popen | None = None
        self._arecord_lock = threading.Lock()
        self._mic_suppressed_until = 0.0

        # Trenutna oseba
        self.current_person: str | None = None
        self.current_gender: str | None = None

        # Shramba: {ime: task}
        self.task_memory: dict[str, str] = {}
        self._task_memory_lock = threading.Lock()
        self.task_person: dict[str, str] = {}

        # Čaka na nalogo (dialog poteka v ločenem threadu)
        self._task_input: str | None = None
        self._waiting_for_task = False
        self._task_input_event = threading.Event()

        self.task_pub = self.create_publisher(String, "/person_task", 10)
        self.create_subscription(String, "/current_person", self._person_cb, 10)
        # /task_input = simulacija mikrofona; zamenjaj z STT ko bo na voljo
        self.create_subscription(String, "/task_input", self._task_input_cb, 10)
        self.approach_active_pub = self.create_publisher(Bool, "/approach_active", 10)

        self.start_task_anomaly_pub = self.create_publisher(Bool, "/task_manager/task_started", 10)
        self.color_of_the_cell_pub = self.create_publisher(String, "/task_manager/color_of_the_cell", 10)
        
        self.color_of_the_cell = None

        # Mape s slikami (ime slike = ID objekta, npr. 1.jpg)
        self.barrels_img_dir = "./barrels"
        self.tiles_img_dir   = "./tiles"

        # Rezultati nalog: {ime_osebe: dict z rezultati}
        self.task_results: dict[str, dict] = {}

        self.create_service(Trigger, "/get_tasks", self._get_tasks_srv)

        self.get_logger().info("Task_manager starting.")


    def _kill_arecord(self):
        with self._arecord_lock:
            if self._arecord_proc is not None:
                try:
                    self._arecord_proc.kill()
                    self._arecord_proc.wait()
                except Exception:
                    pass
                self._arecord_proc = None



    def say(self, text: str):
        """System TTS z spd-say."""
        self.get_logger().info(f"{text}")
        try:
            subprocess.run(
                ["spd-say", "-r", "-10", "-p", "-55", "-t", "male3", text],
                check=False
            )
        except FileNotFoundError:
            print(f"spd-say not found. Would say: {text}")
        finally:
            # Cooldown — arecord bo preskočil ta čas preden začne brati
            self._mic_suppressed_until = time.monotonic() + 0.6


    def _listen_for_categorized_task(self) -> tuple[str, str] | None:
        """Listen until we can confidently categorize into TASK_KEYWORDS.
        Returns (task, raw_text) or None on failure."""
        MAX_ATTEMPTS = 3
        for attempt in range(MAX_ATTEMPTS):
            raw = self.listen()
            if raw is None:
                return None

            task, score = classify_task(raw)
            self.get_logger().info(f"Task classification score={score:.2f} -> {task}")

            if score >= MIN_TASK_SCORE and task is not None:
                return (task, raw)

            if attempt < MAX_ATTEMPTS - 1:
                self.say("I didn't catch that, can you please repeat?")
                #time.sleep(2)

        self.say("I'm sorry, I could not understand the task.")
        return None


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


    def _extract_color(self, text: str) -> str | None:
        t = text.lower()
        if "red" in t:
            return "red"
        if "green" in t:
            return "green"
        return None

    def _listen_for_task_voice(self) -> str | None:
        if Model is None or KaldiRecognizer is None:
            self.get_logger().error("Vosk ni na voljo. Padec nazaj na /task_input topic.")
            return self._listen_for_task()

        if self._vosk_model is None:
            self.get_logger().error("Vosk model ni naložen. Voice STT disabled.")
            return None

        SAMPLE_RATE    = 16000
        CHUNK_BYTES    = 8000   # 0.25s @ 16kHz 16-bit mono = 8000 bytes
        SILENCE_SEC    = 2.0
        MAX_LISTEN_SEC = 15.0

        recognizer = KaldiRecognizer(self._vosk_model, SAMPLE_RATE)

        cmd = ["arecord", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1", "-"]

        self.get_logger().info("🎤 Poslušam ... (govorite)")
        self._waiting_for_task = True

        collected_text: list[str] = []
        listen_start = time.monotonic()
        last_voice_time: float | None = None

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            with self._arecord_lock:
                self._arecord_proc = proc

            try:
                while rclpy.ok():
                    now = time.monotonic()

                    # Čakaj cooldown po TTS da ne posname robot samega sebe
                    if now < self._mic_suppressed_until:
                        time.sleep(0.05)
                        continue

                    if now - listen_start > MAX_LISTEN_SEC:
                        self.get_logger().warn("🎤 Timeout – prenehanje poslušanja.")
                        break

                    data = proc.stdout.read(CHUNK_BYTES)
                    if not data:
                        break

                    if recognizer.AcceptWaveform(data):
                        res = json.loads(recognizer.Result())
                        text = res.get("text", "").strip()
                        if text:
                            self.get_logger().info(f"🎤 Zaznano: '{text}'")
                            collected_text.append(text)
                            last_voice_time = now
                    else:
                        partial = json.loads(recognizer.PartialResult()).get("partial", "")
                        if partial:
                            last_voice_time = now

                    if collected_text and last_voice_time is not None and (now - last_voice_time) > SILENCE_SEC:
                        self.get_logger().info("🎤 Tišina zaznana – končujem.")
                        break

            finally:
                proc.kill()
                proc.wait()
                with self._arecord_lock:
                    self._arecord_proc = None

        except FileNotFoundError:
            self.get_logger().error("arecord ni nameščen. Padec nazaj na /task_input.")
            return self._listen_for_task()
        finally:
            self._waiting_for_task = False

        if not collected_text:
            return None

        full_text = " ".join(collected_text)
        self.get_logger().info(f"🎤 Končni rezultat: '{full_text}'")
        return full_text


    
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
            f"Hello {name}."
            f"What task should I perform? "
        )
        time.sleep(3)

        # Poslušaj odgovor
        result = self._listen_for_categorized_task()
        if result is None:
            self.get_logger().warn(f"No task received for {name}.")
            return
        task, raw_text = result

        self.get_logger().info(f"Task: {task}")

        # If female ASK AGAIN until she says yes
        if gender == "female":
            undecided = True
            while undecided:
                self.say(f"You want me to perform {task}. Are you sure?")
                time.sleep(3)
                raw = self.listen()
                if raw is None:
                    self.get_logger().warn(f"No task received for {name}.")
                    return
                self.get_logger().info(f"Received: '{raw}'")

                if "yes" in raw.lower() or "i am sure" in raw.lower():
                    undecided = False
                else:
                    new_task, score = classify_task(raw)
                    if score < MIN_TASK_SCORE or new_task is None:
                        self.say("I didn't catch that, can you please repeat?")
                        time.sleep(0.2)
                        continue
                    task = new_task
                    raw_text = raw

        self.get_logger().info(f"Task for {name}: {task}")

        # Shrani
        with self._task_memory_lock:
            self.task_memory[task] = name
            self.task_results[name] = self._empty_result(task)

        self.task_person[task] = name

        self.say(f"Understood. I will perform: {task}.")

        if task == "Anomaly detection":
            # Try to extract color from the utterance; ask if missing
            color = self._extract_color(raw_text)
            if color is None:
                self.say("Which cell should I inspect, red or green?")
                time.sleep(1)
                color_raw = self.listen()
                if color_raw:
                    color = self._extract_color(color_raw)

            self.color_of_the_cell = color
            self.get_logger().info(f"Cell color: {color}")
            self.start_task_anomaly_pub.publish(Bool(data=True))
            if color:
                self.color_of_the_cell_pub.publish(String(data=color))

        self.approach_active_pub.publish(Bool(data=False))
        


    # ── Rezultati ─────────────────────────────────────────────────────────────

    def _empty_result(self, task: str) -> dict:
        if task == "Counting rings":
            return {"task": task, "total": 0, "by_color": {}}
        if task == "Barrels inspection":
            return {"task": task, "total": 0, "barrels": []}
        if task == "Anomaly detection":
            return {"task": task, "total": 0, "tiles": []}
        return {"task": task}

    def set_rings_result(self, person: str, total: int, by_color: dict[str, int]) -> None:
        """Nastavi rezultat za Counting rings.
        by_color primer: {"red": 3, "blue": 2, "green": 1}
        """
        with self._task_memory_lock:
            if person in self.task_results:
                self.task_results[person].update({"total": total, "by_color": by_color})

    def set_barrels_result(self, person: str, total: int, barrels: list[dict]) -> None:
        """Nastavi rezultat za Barrels inspection.
        barrels primer: [{"id": 1, "colour": "red", "orientation": "vertical", "leak": True}, ...]
        """
        with self._task_memory_lock:
            if person in self.task_results:
                self.task_results[person].update({"total": total, "barrels": barrels})

    def set_tiles_result(self, person: str, total: int, tiles: list[dict]) -> None:
        """Nastavi rezultat za Anomaly detection.
        tiles primer: [{"id": 1, "status": "OK"}, {"id": 2, "status": "NOK"}, ...]
        """
        with self._task_memory_lock:
            if person in self.task_results:
                self.task_results[person].update({"total": total, "tiles": tiles})

    def _find_image(self, directory: str, obj_id) -> Path | None:
        base = Path(directory) / str(obj_id)
        for ext in (".jpg", ".jpeg", ".png"):
            p = base.with_suffix(ext)
            if p.exists():
                return p
        return None

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
        

    # DODAJ RESULTS CALLBACK
    """
    NPR RINGS CALLBACK:
    person = self.task_person["Counting Rings"]

    self.set_rings_result(
    person="John",
    total=6,
    by_color={
        "red": 3,
        "blue": 2,
        "green": 1
    }
    )"""

    """
    self.set_barrels_result(
    person="Anna",
    total=3,
    barrels=[
        {"id": 1, "colour": "red", "orientation": "vertical", "leak": True},
        {"id": 2, "colour": "blue", "orientation": "horizontal", "leak": False},
        {"id": 3, "colour": "green", "orientation": "vertical", "leak": False}
    ]
)
    """


    """
    self.set_tiles_result(
    person="Mark",
    total=5,
    tiles=[
        {"id": 1, "status": "OK"},
        {"id": 2, "status": "OK"},
        {"id": 3, "status": "NOK"},
        {"id": 4, "status": "OK"},
        {"id": 5, "status": "NOK"}
    ]
)
    """


    #-------------------------------------------------
    def report(self):
        self.say("Hello Sir, I am generating your report.")
        self._generate_tasks_pdf()

    def _generate_tasks_pdf(self) -> Path | None:
        """Generate a PDF report with tasks and results into ./pdf/ directory."""
        with self._task_memory_lock:
            snapshot_tasks   = dict(self.task_memory)
            snapshot_results = dict(self.task_results)

        out_dir = Path.cwd() / "pdf"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path  = out_dir / f"tasks_{timestamp}.pdf"

        styles = getSampleStyleSheet()
        doc    = SimpleDocTemplate(str(out_path), pagesize=A4,
                                   leftMargin=2*cm, rightMargin=2*cm,
                                   topMargin=2*cm, bottomMargin=2*cm)
        story  = []

        story.append(Paragraph("Tasks Report", styles["Title"]))
        story.append(Paragraph(
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            styles["Normal"],
        ))
        story.append(Spacer(1, 0.5*cm))

        if not snapshot_tasks:
            story.append(Paragraph("No tasks recorded.", styles["Normal"]))
            doc.build(story)
            return out_path

        for person_name in sorted(snapshot_tasks, key=str.lower):
            task   = snapshot_tasks[person_name]
            result = snapshot_results.get(person_name, {})

            story.append(Paragraph(person_name, styles["Heading1"]))
            story.append(Paragraph(f"Task: <b>{task}</b>", styles["Normal"]))
            story.append(Spacer(1, 0.3*cm))
            story.extend(self._build_result_section(task, result))
            story.append(Spacer(1, 0.8*cm))

        doc.build(story)
        return out_path

    def _build_result_section(self, task: str, result: dict) -> list:
        styles  = getSampleStyleSheet()
        section = []

        _HDR_CMDS = [
            ("BACKGROUND",    (0, 0), (-1, 0), rl_colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [rl_colors.white, rl_colors.HexColor("#f2f2f2")]),
            ("GRID",          (0, 0), (-1, -1), 0.4, rl_colors.grey),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ]

        if task == "Counting rings":
            total    = result.get("total", 0)
            by_color = result.get("by_color", {})
            section.append(Paragraph(f"Total rings: <b>{total}</b>", styles["Normal"]))
            if by_color:
                section.append(Spacer(1, 0.2*cm))
                data  = [["Color", "Count"]] + [[c, str(n)] for c, n in by_color.items()]
                tbl   = Table(data, colWidths=[5*cm, 3*cm])
                tbl.setStyle(TableStyle(_HDR_CMDS))
                section.append(tbl)

        elif task == "Barrels inspection":
            total   = result.get("total", 0)
            barrels = result.get("barrels", [])
            section.append(Paragraph(f"Total detected: <b>{total}</b>", styles["Normal"]))

            if barrels:
                section.append(Spacer(1, 0.2*cm))
                data = [["Barrel ID", "Colour", "Orientation", "Leak"]]
                for b in barrels:
                    data.append([
                        str(b.get("id", "")),
                        b.get("colour", ""),
                        b.get("orientation", ""),
                        "Yes" if b.get("leak") else "No",
                    ])

                style_cmds = list(_HDR_CMDS)
                for i, b in enumerate(barrels, start=1):
                    if b.get("leak"):
                        style_cmds += [
                            ("TEXTCOLOR", (3, i), (3, i), rl_colors.red),
                            ("FONTNAME",  (3, i), (3, i), "Helvetica-Bold"),
                        ]

                tbl = Table(data, colWidths=[3*cm, 4*cm, 4*cm, 3*cm])
                tbl.setStyle(TableStyle(style_cmds))
                section.append(tbl)

                # Slike za barrele z leak
                for b in barrels:
                    if b.get("leak"):
                        img_path = self._find_image(self.barrels_img_dir, b["id"])
                        if img_path:
                            section.append(Spacer(1, 0.3*cm))
                            section.append(Paragraph(
                                f"Barrel {b['id']} — leak detected", styles["Italic"]
                            ))
                            section.append(RLImage(str(img_path), width=8*cm, height=6*cm))

        elif task == "Anomaly detection":
            total = result.get("total", 0)
            tiles = result.get("tiles", [])
            section.append(Paragraph(f"Total tiles: <b>{total}</b>", styles["Normal"]))

            if tiles:
                section.append(Spacer(1, 0.2*cm))
                data = [["Tile ID", "Status"]]
                for t in tiles:
                    data.append([str(t.get("id", "")), t.get("status", "")])

                style_cmds = list(_HDR_CMDS)
                for i, t in enumerate(tiles, start=1):
                    if t.get("status") == "NOK":
                        style_cmds += [
                            ("TEXTCOLOR", (1, i), (1, i), rl_colors.red),
                            ("FONTNAME",  (1, i), (1, i), "Helvetica-Bold"),
                        ]

                tbl = Table(data, colWidths=[4*cm, 4*cm])
                tbl.setStyle(TableStyle(style_cmds))
                section.append(tbl)

                # Slike za NOK tiles
                for t in tiles:
                    if t.get("status") == "NOK":
                        img_path = self._find_image(self.tiles_img_dir, t["id"])
                        if img_path:
                            section.append(Spacer(1, 0.3*cm))
                            section.append(Paragraph(
                                f"Tile {t['id']} — NOK", styles["Italic"]
                            ))
                            section.append(RLImage(str(img_path), width=8*cm, height=6*cm))

        return section


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
