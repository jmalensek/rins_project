#!/usr/bin/env python3

# INSTALL: vosk, pyaudio, (sounddevice)
# sudo apt install -y libportaudio2 portaudio19-dev

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
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

try:
    import pyaudio  # type: ignore[import-not-found]
except Exception:
    pyaudio = None  # type: ignore[assignment]

try:
    from vosk import Model, KaldiRecognizer  # type: ignore[import-not-found]
except Exception:
    Model = None  # type: ignore[assignment]
    KaldiRecognizer = None  # type: ignore[assignment]

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


TASK_KEYWORDS = {
    "Barrels inspection":       ["barrel", "barrels", "cylinder", "cylinders", "battles"],
    "Counting rings":         ["ring",   "rings",   "count rings", "circles"],
    "Anomaly detection":         ["tiles", "anomaly", "cell", "anomalies"],
}
MIN_TASK_SCORE = 0.45


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())

# similar sounding words to TASK_KEYWORDS
def _soundex(word: str) -> str:
    """Simple English-oriented Soundex (good enough for STT typos)."""
    w = re.sub(r"[^A-Za-z]", "", word)
    if not w:
        return ""
    w = w.upper()
    first = w[0]
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    digits: list[str] = []
    last = ""
    for ch in w[1:]:
        d = mapping.get(ch, "")
        if d == last:
            continue
        last = d
        if d:
            digits.append(d)

    code = (first + "".join(digits) + "000")[:4]
    return code


_TASK_PHONETIC: dict[str, set[str]] = {
    task: {_soundex(tok) for kw in keywords for tok in _tokenize(kw) if _soundex(tok)}
    for task, keywords in TASK_KEYWORDS.items()
}


def classify_task(raw: str) -> tuple[str, float]:
    """Return (task, confidence_score in [0..1])."""
    text = raw.strip().lower()
    tokens = _tokenize(text)

    if not text:
        # Nothing to classify
        return (next(iter(TASK_KEYWORDS)), 0.0)

    # Exact keyword substring match => high confidence
    for task, keywords in TASK_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return (task, 1.0)

    # Phonetic/fuzzy token match => medium-high confidence
    if tokens:
        token_soundex = {_soundex(t) for t in tokens if _soundex(t)}
        for task, keywords in TASK_KEYWORDS.items():
            if token_soundex and (token_soundex & _TASK_PHONETIC.get(task, set())):
                return (task, 0.9)

            for kw in keywords:
                for kw_tok in _tokenize(kw):
                    for t in tokens:
                        if SequenceMatcher(None, t, kw_tok).ratio() >= 0.84:
                            return (task, 0.85)

    # Soft fallback: score all tasks
    best_task: str | None = None
    best_score = 0.0

    for task, keywords in TASK_KEYWORDS.items():
        score = 0.0

        for kw in keywords:
            score = max(score, SequenceMatcher(None, text, kw.lower()).ratio())

        if tokens:
            for kw in keywords:
                kw_tokens = _tokenize(kw)
                for t in tokens:
                    for kw_tok in kw_tokens:
                        score = max(score, SequenceMatcher(None, t, kw_tok).ratio())

        if best_task is None or score > best_score:
            best_score = score
            best_task = task

    return (best_task or next(iter(TASK_KEYWORDS))), best_score



def normalize_task(raw: str) -> str:
    task, _score = classify_task(raw)
    return task


class TaskNode(Node):
    def __init__(self):
        super().__init__("task_node")

        # Optional offline STT configuration (Vosk)
        self.model_path = "/home/lea/colcon_ws/vosk-model-en-us-0.22"
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

        self._pa = None
        self._mic_stream = None
        self._mic_lock = threading.Lock()
        self._mic_suppressed_until = 0.0
        if pyaudio is not None:
            try:
                self._pa = pyaudio.PyAudio()
            except Exception as exc:
                self.get_logger().error(f"Failed to init PyAudio: {exc}")
                self._pa = None

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


    def _get_mic_stream(self, sample_rate: int, chunk: int):
        """Get (or open) a persistent microphone input stream."""
        if self._pa is None:
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
                open_kwargs["input_device_index"] = device_index

            try:
                self._mic_stream = self._pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk,
                    **open_kwargs,
                )
                return self._mic_stream
            except Exception as exc:
                self.get_logger().error(f"Failed to open microphone input stream: {exc}")
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


    def _listen_for_categorized_task(self) -> str | None:
        """Listen until we can confidently categorize into TASK_KEYWORDS."""
        while rclpy.ok():
            raw = self._listen_for_task_voice()
            if raw is None:
                return None

            task, score = classify_task(raw)
            self.get_logger().info(f"Task classification score={score:.2f} -> {task}")

            if score >= MIN_TASK_SCORE:
                return task

            self.say("I didn't catch that, can you please repeat?")
            time.sleep(2)

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


    def _listen_for_task_voice(self) -> str | None:
        if pyaudio is None or Model is None or KaldiRecognizer is None:
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
        task = self._listen_for_categorized_task()
        if task is None:
            self.get_logger().warn(f"No task received for {name}.")
            return

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
                self.get_logger().info(f"Received: '{raw}')")

                if "yes" in raw.lower() or "i am sure" in raw.lower():
                    # se je odločila
                    undecided = False
                else:
                    new_task, score = classify_task(raw)
                    if score < MIN_TASK_SCORE:
                        self.say("I didn't catch that, can you please repeat?")
                        time.sleep(0.2)
                        continue
                    task = new_task


        self.get_logger().info(f"Task for {name}: {task}")

        # Shrani
        with self._task_memory_lock:
            self.task_memory[name] = task

        # Potrditev
        spoken_task = task if task in TASK_KEYWORDS else normalize_task(task)
        self.say(f"Understood. I will perform: {spoken_task}.")

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