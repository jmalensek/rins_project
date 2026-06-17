#!/usr/bin/env python3

# pip install vosk --break-system-packages
# pip install sounddevice --break-system-packages
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
import os
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
    import sounddevice as sd  # type: ignore[import-not-found]
except Exception:
    sd = None  # type: ignore[assignment]

try:
    from vosk import Model, KaldiRecognizer  # type: ignore[import-not-found]
except Exception:
    Model = None  # type: ignore[assignment]
    KaldiRecognizer = None  # type: ignore[assignment]

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


TASK_KEYWORDS = {
    "Barrels inspection": ["barrel", "barrels", "cylinder", "cylinders", "battles"],
    "Counting rings":     ["ring", "rings", "count rings", "circles"],
    "Tile inspection":    ["tile", "tiles", "cell", "anomaly", "anomalies", "inspect"],
}
MIN_TASK_SCORE = 0.45

GRAMMAR = _json.dumps([
    # Barrels
    "barrel", "barrels", "cylinder", "cylinders", "inspection", "barrel inspection",
    # Rings
    "ring", "rings", "count rings", "count the rings",
    # Tile inspection
    "tile", "tiles", "cell", "anomaly", "anomalies",
    # Colors (for tile inspection follow-up)
    "red", "red cell", "green", "green cell",
    # Confirmation
    "yes", "i am sure", "sure", "correct",
    # Fallback
    "nothing", "none", "[unk]"
])


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

        # Optional offline STT configuration (Vosk)
        #self.model_path = "/home/lea/colcon_ws/vosk-model-en-us-0.22"
        self.model_path = "./src/rins_project/vosk-model-en-us-0.22"
        self.declare_parameter("mic_device_index", -1)

        self._vosk_model = None
        if Model is None:
            self.get_logger().error("Vosk ni na voljo (import failed). Voice STT disabled.")
        else:
            try:
                self.get_logger().info(f"🎤 Loading Vosk model once: {self.model_path}")
                self._vosk_model = Model(self.model_path)
            except Exception as exc:
                self.get_logger().error(f"Failed to load Vosk model from '{self.model_path}': {exc}")
                self._vosk_model = None

        self._vosk_grammar = GRAMMAR

        self._sd = None
        self._mic_stream = None
        self._mic_lock = threading.Lock()
        self._mic_suppressed_until = 0.0
        if sd is not None:
            try:
                self._sd = sd
            except Exception as exc:
                self.get_logger().error(f"Failed to initialize sounddevice: {exc}")
                self._sd = None

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

        self.start_task_anomaly_pub = self.create_publisher(Bool, "/task_manager/task_started", 10)
        self.color_of_the_cell_pub = self.create_publisher(String, "/task_manager/color_of_the_cell", 10)
        
        # DODANO ŠE DA SI SHRANI KATERA BARVA JE PRI ANOMALY DETECTION -
        # treba še actually inplementirat katera barva, tu je samo placeholder
        self.color_of_the_cell = None
        
        self.create_service(Trigger, "/get_tasks", self._get_tasks_srv)

        self.get_logger().info("Task_manager starting.")


    def _get_mic_stream(self, sample_rate: int, chunk: int):
        """Get (or open) a persistent microphone input stream."""
        if self._sd is None:
            return None

        with self._mic_lock:
            if self._mic_stream is not None:
                return self._mic_stream

            try:
                device_index = int(self.get_parameter("mic_device_index").value)
            except Exception:
                device_index = -1

            open_kwargs = {}
            if device_index >= 0:
                open_kwargs["device"] = device_index

            try:
                # Use RawInputStream so we get bytes similar to PyAudio's stream.read()
                raw = self._sd.RawInputStream(
                    samplerate=sample_rate,
                    blocksize=chunk,
                    channels=1,
                    dtype='int16',
                    **open_kwargs,
                )

                class SoundDeviceStream:
                    def __init__(self, raw_stream):
                        self._raw = raw_stream

                    def start_stream(self):
                        try:
                            self._raw.start()
                        except Exception:
                            pass

                    def read(self, frames, exception_on_overflow=False):
                        # RawInputStream.read(frames) -> (data, overflowed)
                        data, overflowed = self._raw.read(frames)
                        return data

                    def stop_stream(self):
                        try:
                            self._raw.stop()
                        except Exception:
                            pass

                    def close(self):
                        try:
                            self._raw.close()
                        except Exception:
                            pass

                self._mic_stream = SoundDeviceStream(raw)
                return self._mic_stream
            except Exception as exc:
                self.get_logger().error(f"Failed to open microphone input stream (sounddevice): {exc}")
                self._mic_stream = None
                return None


    def _reset_mic_stream(self):
        with self._mic_lock:
            if self._mic_stream is not None:
                try:
                    self._mic_stream.stop_stream()
                except Exception:
                    pass
                try:
                    self._mic_stream.close()
                except Exception:
                    pass
            self._mic_stream = None



    #  Say response text
    def say(self, text: str):
        """System TTS z spd-say."""
        self.get_logger().info(f"{text}")

        # Stop mic while speaking to avoid STT capturing TTS.
        with self._mic_lock:
            if self._mic_stream is not None:
                try:
                    self._mic_stream.stop_stream()
                except Exception:
                    pass
        try:
            subprocess.run(
                ["spd-say", "-r", "-10", "-p", "-55", "-t", "male3", text],
                check=False
            )
        except FileNotFoundError:
            print(f"spd-say not found. Would say: {text}")
        finally:
            # Small cooldown to reduce echo pickup.
            self._mic_suppressed_until = time.monotonic() + 0.6


    def _listen_for_categorized_task(self) -> tuple[str, str] | None:
        """Listen until we can confidently categorize into TASK_KEYWORDS.
        Returns (task, raw_text) or None on failure."""
        MAX_ATTEMPTS = 3
        for attempt in range(MAX_ATTEMPTS):
            raw = self._listen_for_task_voice()
            if raw is None:
                return None

            task, score = classify_task(raw)
            self.get_logger().info(f"Task classification score={score:.2f} -> {task}")

            if score >= MIN_TASK_SCORE and task is not None:
                return (task, raw)

            if attempt < MAX_ATTEMPTS - 1:
                self.say("I didn't catch that, can you please repeat?")
                time.sleep(2)

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
        if self._sd is None or Model is None or KaldiRecognizer is None:
            self.get_logger().error(
                "Vosk ali pyaudio nista na voljo. Namesti: pip install vosk pyaudio\n"
                "Padec nazaj na /task_input topic."
            )
            return self._listen_for_task()

        if self._vosk_model is None:
            self.get_logger().error("Vosk model ni naložen. Voice STT disabled.")
            return None

        SAMPLE_RATE   = 16000
        CHUNK         = 4000    # ~0.25s blokov
        SILENCE_SEC   = 2.0     # konec govora po toliko tišine
        MAX_LISTEN_SEC = 15.0   # absolutni timeout

        recognizer = KaldiRecognizer(self._vosk_model, SAMPLE_RATE)
        recognizer.SetGrammar(self._vosk_grammar)

        stream = self._get_mic_stream(SAMPLE_RATE, CHUNK)
        if stream is None:
            self.get_logger().error("Microphone stream not available. Falling back to /task_input.")
            return self._listen_for_task()

        self.get_logger().info("🎤 Poslušam ... (govorite)")
        self._waiting_for_task = True

        collected_text: list[str] = []
        listen_start = time.monotonic()
        last_voice_time: float | None = None

        try:
            try:
                stream.start_stream()
            except Exception:
                pass

            while rclpy.ok():
                now = time.monotonic()

                # If we just spoke, wait a moment before reading mic input.
                if now < self._mic_suppressed_until:
                    time.sleep(0.05)
                    continue

                # Absolutni timeout
                if now - listen_start > MAX_LISTEN_SEC:
                    self.get_logger().warn("🎤 Timeout – prenehanje poslušanja.")
                    break

                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                except Exception as exc:
                    self.get_logger().error(f"Microphone read failed: {exc}")
                    self._reset_mic_stream()
                    return None

                if recognizer.AcceptWaveform(data):
                    # Celotna poved zaključena
                    res = json.loads(recognizer.Result())
                    text = res.get("text", "").strip()
                    if text:
                        self.get_logger().info(f"🎤 Zaznano: '{text}'")
                        collected_text.append(text)
                        last_voice_time = now
                else:
                    # Delni rezultat – zaznaj tišino
                    partial = json.loads(recognizer.PartialResult()).get("partial", "")
                    if partial:
                        last_voice_time = now   # nekdo govori

                # Stop once we've heard something and then got enough silence.
                if collected_text and last_voice_time is not None and (now - last_voice_time) > SILENCE_SEC:
                    self.get_logger().info("🎤 Tišina zaznana – končujem.")
                    break

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
                raw = self._listen_for_task_voice()
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
            self.task_memory[name] = task

        self.say(f"Understood. I will perform: {task}.")

        if task == "Tile inspection":
            # Try to extract color from the utterance; ask if missing
            color = self._extract_color(raw_text)
            if color is None:
                self.say("Which cell should I inspect, red or green?")
                time.sleep(1)
                color_raw = self._listen_for_task_voice()
                if color_raw:
                    color = self._extract_color(color_raw)

            self.color_of_the_cell = color
            self.get_logger().info(f"Cell color: {color}")
            self.start_task_anomaly_pub.publish(Bool(data=True))
            if color:
                self.color_of_the_cell_pub.publish(String(data=color))


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
