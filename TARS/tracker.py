"""ISRO-Astrotrack: conservative, GUI-ready vision/state pipeline.

Dependencies: opencv-python, numpy, ultralytics, mediapipe, pyttsx3 (Windows).
Place best.pt beside this file or pass its path to AstrotrackEngine.
"""
from collections import deque
import os
import queue
import threading
import time

import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO
import winsound
import mediapipe as mp

# Model contract: best.pt was checked and reports these exact names and IDs.
CLASS_MAIN, CLASS_RED, CLASS_YELLOW = 0, 1, 2
CLASS_NAMES = {CLASS_MAIN: "main-box", CLASS_RED: "red-box", CLASS_YELLOW: "yellow-box"}

# --- STRICT AI CONFIDENCE & SIZE FILTERS ---
# YOLO must run below the main-box acceptance threshold so the main box is not
# discarded before class-specific filtering is applied.
YOLO_INFER_CONFIDENCE = 0.20
MAIN_CONFIDENCE = 0.30
SMALL_CONFIDENCE = 0.72
MIN_DETECTION_SIDE_PX = 28

# Small objects are expected to be compact, rectangular colored payloads.
SMALL_MIN_COLOR_FRACTION = 0.42
SMALL_MIN_RECTANGULARITY = 0.50
# --------------------------------------------

MAIN_FALLBACK_MIN_AREA_RATIO = 0.025
MAIN_FALLBACK_MAX_AREA_RATIO = 0.65
MAIN_FALLBACK_MIN_RECTANGULARITY = 0.45
MAIN_FALLBACK_CENTER_X_MIN = 0.15
MAIN_FALLBACK_CENTER_X_MAX = 0.85
MAIN_FALLBACK_CENTER_Y_MIN = 0.35
MAIN_TOUCH_DISTANCE_FACTOR = 0.32
MAIN_TOUCH_MIN_PX = 28
NMS_IOU = 0.45
MAX_DETECTIONS = 40

# --- ANTI-SPEEDRUN FILTERS (Forces deliberate astronaut action) ---
TOUCH_MARGIN_PX = 15
TOUCH_SUSTAIN_FRAMES = 15       # Hand must touch for 0.5 seconds
ANOMALY_RELEASE_FRAMES = 15     
REGION_STABLE_FRAMES = 15       
UNLOAD_STABLE_FRAMES = 15       # Object must be unloaded for 0.5s before validating
LOAD_STABLE_FRAMES = 15         # Object must be loaded for 0.5s before validating
# Lid gestures are intentionally more tolerant than normal object-touch checks.
# Closing the physical lid can happen quickly, so do not require a long 15-frame
# touch streak before measuring the downward hand movement.
LID_MIN_VERTICAL_TRAVEL_PX = 14
LID_CLOSE_CONTACT_FRAMES = 1
LID_CLOSE_WINDOW_S = 1.5
LID_CLOSE_END_RATIO = 0.20
LID_CONTACT_WINDOW_S = 1.2
LID_QUALIFIED_WINDOW_S = 2.8
# ------------------------------------------------------------------

# Main-box tracker.
MAIN_TRACK_MAX_MISSED = 45

# Small-object temporal tracking.
SMALL_MAX_MAIN_AREA_RATIO = 0.12 # Max size ratio for payloads vs main box
OBJECT_MAX_MISSED_FRAMES = 18
OBJECT_REDETECT_DISTANCE_FACTOR = 1.35
OBJECT_COLOR_SEARCH_FACTOR = 1.35
OBJECT_MIN_COLOR_FRACTION = 0.48
OBJECT_COLOR_MIN_SATURATION = 120
OBJECT_COLOR_MIN_VALUE = 70
OBJECT_MAX_VELOCITY_PX = 90.0
OBJECT_VELOCITY_SMOOTHING = 0.55
OBJECT_MIN_TRACK_SIDE_PX = 8

# Human/object interaction.
OBJECT_HAND_DISTANCE_FACTOR = 0.90
OBJECT_HAND_DISTANCE_MIN_PX = 22

# Unload/load event confirmation.
UNLOAD_MIN_DISPLACEMENT_FACTOR = 0.50
UNLOAD_MIN_DISPLACEMENT_PX = 24

audio_library = {
    "startup": "Astrotrack online. Step 1: Touch the main box.",
    "success_0": "Step 1 completed correctly. Now Step 2: Open the main box.",
    "success_1": "Step 2 completed correctly. Now Step 3: Unload the yellow box.",
    "success_2": "Step 3 completed correctly. Now Step 4: Unload the red box.",
    "success_3": "Step 4 completed correctly. Now Step 5: Load the red box.",
    "success_4": "Step 5 completed correctly. Now Step 6: Load the yellow box.",
    "success_5": "Step 6 completed correctly. Now Step 7: Close the main box.",
    "success_6": "Step 7 completed correctly. Experiment sequence completed successfully.",
    "anomaly": "Anomaly detected. Incorrect action.",
}

experiment_steps = [
    {"class_id": CLASS_MAIN, "rule": "touch", "action": "Touch the main box"},
    {"class_id": CLASS_MAIN, "rule": "lid_open", "action": "Open the main box"},
    {"class_id": CLASS_YELLOW, "rule": "unload", "action": "Unload the yellow box"},
    {"class_id": CLASS_RED, "rule": "unload", "action": "Unload the red box"},
    {"class_id": CLASS_RED, "rule": "load", "action": "Load the red box"},
    {"class_id": CLASS_YELLOW, "rule": "load", "action": "Load the yellow box"},
    {"class_id": CLASS_MAIN, "rule": "lid_close", "action": "Close the main box"},
]

class VoiceAnnouncer:
    """Strict, Sequential Audio Player handling both WAVs and dynamic Text-to-Speech."""
    def __init__(self, library: dict, compile_files=False):
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def say(self, key):
        if key:
            self._queue.put(key)

    def say_text(self, text):
        if text:
            # Pass the actual detailed anomaly text into the queue
            self._queue.put(("__text__", text))

    def _run(self):
        # Initialize pyttsx3 inside the isolated thread so it doesn't freeze the GUI
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
        except Exception:
            engine = None

        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            try:
                if isinstance(item, tuple) and item[0] == "__text__":
                    # Speak dynamic anomaly explanations out loud
                    if engine is not None:
                        engine.say(item[1])
                        engine.runAndWait()
                else:
                    # Play pre-recorded zero-latency WAVs for success chimes
                    filepath = f"{item}.wav"
                    if os.path.exists(filepath):
                        winsound.PlaySound(filepath, winsound.SND_FILENAME)
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def stop(self):
        self._stop.set()


def play_instant_audio(key: str, announcer):
    """Compatibility helper: enqueue speech without blocking the vision loop."""
    if announcer is not None:
        announcer.say(key)


class StateManager:
    """Thread-safe state and FIFO event mailbox for GUI + voice announcements."""
    def __init__(self, steps):
        self._lock = threading.Lock()
        self._steps = steps
        self._index = 0
        self._audio_queue = deque()
        self._event_queue = deque()
        self._completed_history = deque(maxlen=7)
        self._last_anomaly = None

    def snapshot_for_hud(self):
        with self._lock:
            done = self._index >= len(self._steps)
            text = "EXPERIMENT COMPLETE" if done else self._steps[self._index]["action"]
            return self._index, text, done

    def current_step(self):
        with self._lock:
            return self._steps[self._index] if self._index < len(self._steps) else None

    def advance(self):
        with self._lock:
            if self._index >= len(self._steps):
                return False
            completed_index = self._index
            self._index += 1
            self._audio_queue.append(f"success_{completed_index}")
            completed = {
                "type": "step_completed",
                "step_index": completed_index,
                "action": self._steps[completed_index]["action"],
            }
            self._completed_history.append(completed)
            self._event_queue.append(completed)
            return True

    def flag_anomaly(self, message=None):
        with self._lock:
            if message:
                self._audio_queue.append(("__text__", message))
            else:
                self._audio_queue.append("anomaly")
            self._last_anomaly = message or "Anomaly detected. Incorrect action."
            self._event_queue.append({"type": "anomaly", "message": self._last_anomaly})

    def pop_audio(self):
        with self._lock:
            if not self._audio_queue:
                return None
            return self._audio_queue.popleft()

    def pop_event(self):
        with self._lock:
            if not self._event_queue:
                return None
            return self._event_queue.popleft()

    def completed_history(self):
        with self._lock:
            return list(self._completed_history)

    def last_anomaly(self):
        with self._lock:
            return self._last_anomaly

    def reset(self):
        with self._lock:
            self._index = 0
            self._audio_queue.clear()
            self._event_queue.clear()
            self._completed_history.clear()
            self._last_anomaly = None


class SmallObjectTracker:
    """Temporal tracker for the red/yellow experiment objects."""
    def __init__(self, class_id):
        self.class_id = class_id
        self.box = None
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.size = None
        self.missed = 0
        self.last_conf = 0.0
        self.source = "none"
        self.age = 0

    @staticmethod
    def _center(box):
        return np.array([
            (box[0] + box[2]) * 0.5,
            (box[1] + box[3]) * 0.5
        ], dtype=np.float32)

    @staticmethod
    def _area(box):
        return max(1.0, (box[2]-box[0]) * (box[3]-box[1]))

    @staticmethod
    def _iou(a, b):
        if a is None or b is None:
            return 0.0
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0.0, x2-x1) * max(0.0, y2-y1)
        return inter / max(1.0, SmallObjectTracker._area(a) +
                           SmallObjectTracker._area(b) - inter)

    def reset(self):
        self.__init__(self.class_id)

    def _accept(self, box, confidence, source):
        new_center = self._center(box)
        if self.box is not None:
            old_center = self._center(self.box)
            delta = new_center - old_center
            self.velocity = (
                OBJECT_VELOCITY_SMOOTHING * self.velocity +
                (1.0 - OBJECT_VELOCITY_SMOOTHING) * delta
            )
            speed = float(np.linalg.norm(self.velocity))
            if speed > OBJECT_MAX_VELOCITY_PX:
                self.velocity *= OBJECT_MAX_VELOCITY_PX / speed
        self.box = [float(v) for v in box]
        self.size = (max(1.0, box[2]-box[0]), max(1.0, box[3]-box[1]))
        self.missed = 0
        self.last_conf = float(confidence)
        self.source = source
        self.age += 1

    def _predicted_box(self):
        if self.box is None:
            return None
        cx, cy = self._center(self.box) + self.velocity
        bw, bh = self.size if self.size is not None else (
            self.box[2]-self.box[0], self.box[3]-self.box[1])
        return [
            float(cx-bw/2), float(cy-bh/2),
            float(cx+bw/2), float(cy+bh/2)
        ]

    def _color_mask(self, frame):
        """Strict color mask used only to recover an already tracked object.

        This is intentionally much stricter than the old mask because human skin
        can occupy the same red/orange HSV range.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, sat, val = cv2.split(hsv)

        if self.class_id == CLASS_YELLOW:
            mask = (
                (h >= 20) & (h <= 38) &
                (sat >= OBJECT_COLOR_MIN_SATURATION) &
                (val >= OBJECT_COLOR_MIN_VALUE)
            )
        else:
            mask = (
                (((h <= 8) | (h >= 172))) &
                (sat >= 135) &
                (val >= 65)
            )

        return (mask.astype(np.uint8) * 255)

    def _color_recovery(self, frame, predicted):
        # Never create a new object from color alone. Recovery is allowed only
        # after YOLO has already established a real object track.
        if predicted is None or self.age < 3:
            return None

        h, w = frame.shape[:2]
        bw = max(OBJECT_MIN_TRACK_SIDE_PX, predicted[2]-predicted[0])
        bh = max(OBJECT_MIN_TRACK_SIDE_PX, predicted[3]-predicted[1])

        pad_x = OBJECT_COLOR_SEARCH_FACTOR * bw
        pad_y = OBJECT_COLOR_SEARCH_FACTOR * bh

        rx1 = max(0, int(predicted[0] - pad_x))
        ry1 = max(0, int(predicted[1] - pad_y))
        rx2 = min(w, int(predicted[2] + pad_x))
        ry2 = min(h, int(predicted[3] + pad_y))
        if rx2 <= rx1 or ry2 <= ry1:
            return None

        mask = self._color_mask(frame)
        roi = mask[ry1:ry2, rx1:rx2].copy()
        kernel = np.ones((3, 3), np.uint8)
        roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel)
        roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        pred_center = self._center(predicted)
        best = None
        best_score = -1e9

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < max(25.0, 0.04 * bw * bh):
                continue

            x, y, cw, ch = cv2.boundingRect(contour)
            if min(cw, ch) < OBJECT_MIN_TRACK_SIDE_PX:
                continue

            candidate = [
                float(rx1+x), float(ry1+y),
                float(rx1+x+cw), float(ry1+y+ch)
            ]
            candidate_area = self._area(candidate)

            if candidate_area < 0.20 * self._area(predicted):
                continue
            if candidate_area > 2.5 * self._area(predicted):
                continue

            rectangularity = area / max(1.0, float(cw * ch))
            if rectangularity < SMALL_MIN_RECTANGULARITY:
                continue

            c = self._center(candidate)
            distance = float(np.linalg.norm(c-pred_center))
            color_fraction = float(
                cv2.countNonZero(
                    mask[int(candidate[1]):int(candidate[3]),
                         int(candidate[0]):int(candidate[2])]
                ) / candidate_area
            )
            if color_fraction < OBJECT_MIN_COLOR_FRACTION:
                continue

            # Prefer dense, rectangular color close to the predicted position.
            score = (
                color_fraction * 2.5
                + rectangularity
                - distance / max(bw, bh, 1.0)
            )
            if score > best_score:
                best_score = score
                best = candidate

        return best

    def update(self, frame, detections):
        predicted = self._predicted_box()

        if detections:
            if self.box is None:
                # Only YOLO detections can initialize a small-object track.
                best = max(detections, key=lambda x: x[0])
                self._accept(best[1], best[0], "yolo")
                return self.box

            old_center = self._center(self.box)
            scale = max(10.0, np.hypot(self.size[0], self.size[1]))
            valid = []

            for conf, box in detections:
                c = self._center(box)
                dist = float(np.linalg.norm(c-old_center))
                iou = self._iou(predicted, box)

                # Do not jump to a hand/skin detection elsewhere in the frame.
                if dist <= OBJECT_REDETECT_DISTANCE_FACTOR * scale or iou >= 0.15:
                    valid.append((conf, box, dist, iou))

            if valid:
                valid.sort(
                    key=lambda x: x[0] + 0.35*x[3] - 0.015*x[2]/scale,
                    reverse=True
                )
                self._accept(valid[0][1], valid[0][0], "yolo")
                return self.box

        recovered = self._color_recovery(frame, predicted)
        if recovered is not None:
            self._accept(
                recovered,
                min(0.5, max(0.30, self.last_conf)),
                "color"
            )
            return self.box

        if predicted is not None and self.missed < OBJECT_MAX_MISSED_FRAMES:
            self.box = predicted
            self.missed += 1
            self.source = "predicted"
            self.age += 1
            return self.box

        self.missed += 1
        if self.missed > OBJECT_MAX_MISSED_FRAMES:
            self.box = None
            self.source = "none"
        return self.box

    @property
    def visible(self):
        return self.box is not None and self.missed == 0

    @property
    def tracked(self):
        return self.box is not None

class MainBoxTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0],
            [0,0,1,0,0,0,1,0], [0,0,0,1,0,0,0,1],
            [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.initialized = False
        self.frames_since_measurement = 0

    def update(self, measured_box):
        if not self.initialized and measured_box is None:
            return
        self.kf.predict()
        if measured_box is not None:
            x1, y1, x2, y2 = measured_box
            cx, cy, bw, bh = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
            measurement = np.array([[cx], [cy], [bw], [bh]], dtype=np.float32)
            if not self.initialized:
                self.kf.statePost = np.array([[cx],[cy],[bw],[bh],[0],[0],[0],[0]], dtype=np.float32)
                self.initialized = True
            self.kf.correct(measurement)
            self.frames_since_measurement = 0
        else:
            self.frames_since_measurement += 1

    def get_box(self):
        if not self.initialized or self.frames_since_measurement > MAIN_TRACK_MAX_MISSED:
            return None
        cx, cy, bw, bh = self.kf.statePost[:4, 0]
        if bw <= 1 or bh <= 1:
            return None
        return [float(cx-bw/2), float(cy-bh/2), float(cx+bw/2), float(cy+bh/2)]


class StableRegion:
    def __init__(self, n=REGION_STABLE_FRAMES):
        self.samples = deque(maxlen=n)
        self.value = None

    def update(self, value):
        self.samples.append(value)
        if value is None:
            self.value = None
        elif len(self.samples) == self.samples.maxlen and all(v == value for v in self.samples):
            self.value = value
        return self.value


class LidGestureTracker:
    """Detect opening/closing of the main lid from hand/wrist motion.

    Step 7 is deliberately tolerant: a real operator may close the lid in a
    short movement, so the old requirement of TOUCH_SUSTAIN_FRAMES (15 frames)
    could miss the action.  We keep the normal strict gesture for opening, but
    use a dedicated close detector that looks for a clear downward movement
    while the hand is around the main box.
    """
    def __init__(self):
        self.samples = {"left": deque(), "right": deque()}
        self.touch_streak = {"left": 0, "right": 0}
        self.anchor_y = {"left": None, "right": None}
        self.qualified_until = {"left": 0.0, "right": 0.0}
        self.close_touch_streak = {"left": 0, "right": 0}
        self.close_anchor_y = {"left": None, "right": None}

    def reset(self):
        self.__init__()

    @staticmethod
    def _touching(point, main_box):
        if point is None or main_box is None:
            return False
        return AstrotrackEngine._point_to_box_distance(point, main_box) <= max(
            MAIN_TOUCH_MIN_PX,
            MAIN_TOUCH_DISTANCE_FACTOR * np.hypot(
                main_box[2]-main_box[0],
                main_box[3]-main_box[1]
            )
        )

    def _update_close(self, side, point, main_box, now):
        if point is None or main_box is None:
            self.close_touch_streak[side] = 0
            return False

        touching = self._touching(point, main_box)
        history = list(self.samples[side])
        cutoff = now - LID_CLOSE_WINDOW_S
        recent = [(t, y) for t, y in history if t >= cutoff]

        # Step 7 is only active when the expected action is closing the main
        # box.  Use a deliberately simple downward-motion detector here.  The
        # previous detector could miss the action because the wrist can move
        # outside the exact main-box boundary while the lid is being pushed
        # down.
        if touching:
            self.close_touch_streak[side] += 1
            if self.close_anchor_y[side] is None:
                self.close_anchor_y[side] = float(point[1])

        if self.close_anchor_y[side] is not None and recent:
            start_y = self.close_anchor_y[side]
            current_y = float(np.median([y for _, y in recent[-3:]]))
            travel = current_y - start_y
            main_h = max(1.0, main_box[3] - main_box[1])
            end_in_box_zone = current_y >= main_box[1] + LID_CLOSE_END_RATIO * main_h

            # Either the hand is still touching the box, or it has just moved
            # through the box zone. This handles the common case where the
            # wrist briefly leaves the detected box during lid closure.
            if travel >= LID_MIN_VERTICAL_TRAVEL_PX and (touching or end_in_box_zone):
                self.close_touch_streak[side] = 0
                self.close_anchor_y[side] = None
                return True
        else:
            # Do not erase a close anchor immediately: the wrist detector can
            # briefly leave the box during the final part of the lid motion.
            if self.close_anchor_y[side] is not None:
                history = list(self.samples[side])
                if not history or now - history[-1][0] > 0.45:
                    self.close_touch_streak[side] = 0
                    self.close_anchor_y[side] = None
            else:
                self.close_touch_streak[side] = 0
        return False

    def update(self, wrists, main_box, mode="open"):
        now = time.monotonic()
        for side in ("left", "right"):
            point = wrists.get(side)
            history = self.samples[side]

            if point is not None:
                history.append((now, float(point[1])))
                while history and now-history[0][0] > LID_QUALIFIED_WINDOW_S:
                    history.popleft()

            if mode == "close":
                if self._update_close(side, point, main_box, now):
                    self.reset()
                    return "down"
                continue

            touching = self._touching(point, main_box)
            if touching:
                self.touch_streak[side] += 1
                if self.touch_streak[side] >= TOUCH_SUSTAIN_FRAMES and self.anchor_y[side] is None:
                    recent_touch_y = [y for _, y in list(history)[-TOUCH_SUSTAIN_FRAMES:]]
                    self.anchor_y[side] = float(np.median(recent_touch_y))
                    self.qualified_until[side] = now + LID_QUALIFIED_WINDOW_S
            elif self.anchor_y[side] is None:
                self.touch_streak[side] = 0

            if self.anchor_y[side] is not None:
                if now > self.qualified_until[side]:
                    self.touch_streak[side] = 0
                    self.anchor_y[side] = None
                    history.clear()
                    continue
                recent = [y for _, y in list(history)[-3:]]
                if recent:
                    travel = float(np.median(recent)) - self.anchor_y[side]
                    if abs(travel) >= LID_MIN_VERTICAL_TRAVEL_PX:
                        self.reset()
                        return "down" if travel > 0 else "up"
        return None


def centroid_region(small_box, main_box):
    if small_box is None or main_box is None:
        return None
    cx = (small_box[0] + small_box[2]) / 2.0
    cy = (small_box[1] + small_box[3]) / 2.0
    x1, y1, x2, y2 = main_box
    pad_x, pad_y = 0.15*(x2-x1), 0.15*(y2-y1)
    inside_core = x1+pad_x <= cx <= x2-pad_x and y1+pad_y <= cy <= y2-pad_y
    return "loaded" if inside_core else "unloaded"


class AstrotrackEngine:
    def __init__(self, model_path=None, camera_index=0, compile_audio_files=False):
        self.voice = VoiceAnnouncer(dict(audio_library), compile_files=compile_audio_files)
        self.state = StateManager(experiment_steps)
        self.main_tracker = MainBoxTracker()
        self.lid_tracker = LidGestureTracker()
        self.regions = {CLASS_RED: StableRegion(), CLASS_YELLOW: StableRegion()}
        self.object_trackers = {
            CLASS_RED: SmallObjectTracker(CLASS_RED),
            CLASS_YELLOW: SmallObjectTracker(CLASS_YELLOW),
        }
        self._last_candidate_lists = {
            CLASS_MAIN: [], CLASS_RED: [], CLASS_YELLOW: []
        }
        self._main_measurement_source = "none"
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
        self.model = YOLO(model_path)
        names = self.model.names
        for class_id, expected_name in CLASS_NAMES.items():
            if names.get(class_id) != expected_name:
                raise ValueError(f"best.pt class {class_id} must be {expected_name!r}; model reports {names}")
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_pose = mp.solutions.pose
        self.camera_index = camera_index
        self.cap = None
        self._started = False
        self.touch_counts = {CLASS_MAIN: 0, CLASS_RED: 0, CLASS_YELLOW: 0}
        self.anomaly_latched = False
        self.no_touch_frames = 0
        self._active_step_index = 0
        self._step_start_region = None
        self._interaction_frame_count = 0
        self._last_confirmed_transition = None
        self._load_anchors = {CLASS_RED: None, CLASS_YELLOW: None}
        self._object_phase = {CLASS_RED: None, CLASS_YELLOW: None}
        self._step_anchor = None
        self._step_motion_count = 0
        self._object_interaction_seen = False
        self._hand_anchor = None
        self._hand_motion_frames = 0
        self._last_hand_center = None

    def start(self):
        if not self._started:
            self.voice.say("startup")
            self._started = True

    @staticmethod
    def _select_candidate(candidates, previous_box):
        if not candidates:
            return None
        if previous_box is None:
            return max(candidates, key=lambda item: item[0])[1]
        old_cx = (previous_box[0] + previous_box[2]) / 2
        old_cy = (previous_box[1] + previous_box[3]) / 2
        def score(item):
            conf, box = item
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            scale = max(1.0, np.hypot(previous_box[2]-previous_box[0], previous_box[3]-previous_box[1]))
            return conf - min(0.20, np.hypot(cx-old_cx, cy-old_cy) / scale * 0.025)
        return max(candidates, key=score)[1]

    def _detect(self, frame, tracked_main_box=None):
        """Detect the main box and payload boxes.

        Important fixes:
        1. YOLO inference is not forced to 0.85, so the main box can actually
           reach the class-specific MAIN_CONFIDENCE check.
        2. Red/yellow boxes are accepted only near the main box.
        3. Color segmentation is NOT used to create new red/yellow detections.
           This prevents skin, shirts and background colors becoming boxes.
        4. A geometry fallback finds the large rectangular main box when YOLO
           temporarily labels it as yellow/red or misses it.
        """
        result = self.model.predict(
            frame,
            conf=YOLO_INFER_CONFIDENCE,
            iou=NMS_IOU,
            max_det=MAX_DETECTIONS,
            verbose=False
        )[0]

        candidates = {CLASS_MAIN: [], CLASS_RED: [], CLASS_YELLOW: []}

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = (
            cv2.inRange(hsv, (0, 120, 55), (8, 255, 255)) |
            cv2.inRange(hsv, (172, 120, 55), (180, 255, 255))
        )
        yellow_mask = cv2.inRange(hsv, (20, 120, 70), (38, 255, 255))
        color_masks = {CLASS_RED: red_mask, CLASS_YELLOW: yellow_mask}

        def clipped_box(box):
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(frame.shape[1]-1, x1))
            y1 = max(0, min(frame.shape[0]-1, y1))
            x2 = max(0, min(frame.shape[1], x2))
            y2 = max(0, min(frame.shape[0], y2))
            return x1, y1, x2, y2

        def color_fraction(box, class_id):
            x1, y1, x2, y2 = clipped_box(box)
            if x2 <= x1 or y2 <= y1:
                return 0.0
            roi = color_masks[class_id][y1:y2, x1:x2]
            return cv2.countNonZero(roi) / float(max(1, (x2-x1)*(y2-y1)))

        def rectangularity(box, class_id):
            x1, y1, x2, y2 = clipped_box(box)
            if x2 <= x1 or y2 <= y1:
                return 0.0
            roi = color_masks[class_id][y1:y2, x1:x2]
            contours, _ = cv2.findContours(
                roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return 0.0
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            return area / float(max(1, (x2-x1)*(y2-y1)))

        # --------------------------------------------------------------
        # YOLO proposals
        # --------------------------------------------------------------
        if result.boxes is not None and len(result.boxes):
            coords = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.cls.int().cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for box, class_id, score in zip(coords, ids, scores):
                class_id = int(class_id)
                if class_id not in candidates:
                    continue

                x1, y1, x2, y2 = map(float, box)
                width = x2 - x1
                height = y2 - y1
                area = width * height

                if min(width, height) < MIN_DETECTION_SIDE_PX:
                    continue

                if class_id == CLASS_MAIN:
                    if score < MAIN_CONFIDENCE:
                        continue
                    # Main box is expected to be substantial.
                    if area < 0.012 * frame.shape[0] * frame.shape[1]:
                        continue
                else:
                    if score < SMALL_CONFIDENCE:
                        continue
                    aspect = width / max(1.0, height)
                    if not 0.40 <= aspect <= 2.50:
                        continue
                    if width > frame.shape[1] * 0.32 or height > frame.shape[0] * 0.32:
                        continue

                    # A payload must contain its class color densely.
                    cf = color_fraction(box, class_id)
                    rect = rectangularity(box, class_id)
                    if cf < SMALL_MIN_COLOR_FRACTION:
                        continue
                    if rect < SMALL_MIN_RECTANGULARITY:
                        continue

                candidates[class_id].append(
                    (float(score), [x1, y1, x2, y2])
                )

        previous = getattr(
            self, "_last_boxes",
            {c: None for c in candidates}
        )

        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_h * frame_w)

        # --------------------------------------------------------------
        # Geometry fallback for the large main box.
        # --------------------------------------------------------------
        def find_geometry_main(reference=None):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

            # Ignore the very top area where faces/arms are likely to appear.
            roi_y = int(frame_h * 0.28)
            roi = edges[roi_y:, :]
            contours, _ = cv2.findContours(
                roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            best = None
            best_score = -1e9

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 0.012 * frame_area or area > 0.55 * frame_area:
                    continue

                peri = cv2.arcLength(contour, True)
                if peri <= 0:
                    continue

                approx = cv2.approxPolyDP(contour, 0.035 * peri, True)
                x, y, bw, bh = cv2.boundingRect(contour)
                y += roi_y

                if bw < frame_w * 0.18 or bh < frame_h * 0.08:
                    continue
                if bw > frame_w * 0.90 or bh > frame_h * 0.65:
                    continue

                cx = x + bw / 2.0
                cy = y + bh / 2.0
                if not (frame_w * 0.12 <= cx <= frame_w * 0.88):
                    continue
                if cy < frame_h * 0.38:
                    continue

                rect = area / float(max(1, bw * bh))
                if rect < MAIN_FALLBACK_MIN_RECTANGULARITY:
                    continue

                aspect = bw / max(1.0, bh)
                if not 1.0 <= aspect <= 4.5:
                    continue

                center_score = 1.0 - min(
                    1.0, abs(cx-frame_w/2) / (frame_w*0.45)
                )
                lower_score = min(
                    1.0, max(0.0, (cy-frame_h*0.38)/(frame_h*0.50))
                )
                area_score = min(1.0, area/(0.10*frame_area))
                ref_score = 0.0

                if reference is not None:
                    rcx = (reference[0] + reference[2]) / 2.0
                    rcy = (reference[1] + reference[3]) / 2.0
                    rdiag = max(
                        20.0,
                        np.hypot(reference[2]-reference[0],
                                 reference[3]-reference[1])
                    )
                    dist = np.hypot(cx-rcx, cy-rcy)
                    ref_score = max(0.0, 1.0-dist/(2.0*rdiag))

                # A quadrilateral is a useful extra signal, but not mandatory
                # because perspective can make the box non-rectangular.
                quad_bonus = 0.15 if len(approx) == 4 else 0.0

                score = (
                    2.2*center_score +
                    1.6*lower_score +
                    1.8*area_score +
                    1.8*rect +
                    2.5*ref_score +
                    quad_bonus
                )

                if score > best_score:
                    best_score = score
                    best = [float(x), float(y),
                            float(x+bw), float(y+bh)]

            return best

        geometry_main = find_geometry_main(tracked_main_box)

        # Pick a valid YOLO main. If it is implausibly small or far from the
        # previous/geometry box, prefer the geometry measurement.
        main_yolo = None
        if candidates[CLASS_MAIN]:
            candidates[CLASS_MAIN].sort(key=lambda x: x[0], reverse=True)
            for conf, box in candidates[CLASS_MAIN]:
                area = (box[2]-box[0]) * (box[3]-box[1])
                cy = (box[1]+box[3]) / 2.0
                if area < 0.012*frame_area or cy < 0.28*frame_h:
                    continue

                if tracked_main_box is not None:
                    old_cx = (tracked_main_box[0]+tracked_main_box[2])/2
                    old_cy = (tracked_main_box[1]+tracked_main_box[3])/2
                    new_cx = (box[0]+box[2])/2
                    new_cy = (box[1]+box[3])/2
                    scale = max(
                        30.0,
                        np.hypot(
                            tracked_main_box[2]-tracked_main_box[0],
                            tracked_main_box[3]-tracked_main_box[1]
                        )
                    )
                    if np.hypot(new_cx-old_cx, new_cy-old_cy) > 2.0*scale:
                        continue

                main_yolo = box
                break

        # If the model says the large box is yellow/red, geometry is used as
        # the main box instead of allowing the small-class detection to own it.
        if main_yolo is not None:
            if geometry_main is not None:
                ga = (geometry_main[2]-geometry_main[0])*(geometry_main[3]-geometry_main[1])
                ya = (main_yolo[2]-main_yolo[0])*(main_yolo[3]-main_yolo[1])
                gcx = (geometry_main[0]+geometry_main[2])/2
                gcy = (geometry_main[1]+geometry_main[3])/2
                ycx = (main_yolo[0]+main_yolo[2])/2
                ycy = (main_yolo[1]+main_yolo[3])/2
                if ga > 1.35*ya or np.hypot(gcx-ycx, gcy-ycy) < 0.45*np.sqrt(max(ga, 1)):
                    main_box = geometry_main
                    self._main_measurement_source = "geometry"
                else:
                    main_box = main_yolo
                    self._main_measurement_source = "yolo"
            else:
                main_box = main_yolo
                self._main_measurement_source = "yolo"
        elif geometry_main is not None:
            main_box = geometry_main
            self._main_measurement_source = "geometry"
        elif previous[CLASS_MAIN] is not None:
            main_box = previous[CLASS_MAIN]
            self._main_measurement_source = "tracked"
        else:
            main_box = tracked_main_box
            self._main_measurement_source = "none" if main_box is None else "tracked"

        candidates[CLASS_MAIN] = (
            [(0.50, main_box)] if main_box is not None else []
        )

        # --------------------------------------------------------------
        # Small-object filtering: only inside/near the actual main box.
        # --------------------------------------------------------------
        if main_box is not None:
            mx1, my1, mx2, my2 = main_box
            main_area = max(1.0, (mx2-mx1)*(my2-my1))
            pad_x = 0.40*(mx2-mx1)
            pad_y = 0.55*(my2-my1)

            vicinity = (
                max(0, mx1-pad_x), max(0, my1-pad_y),
                min(frame_w, mx2+pad_x), min(frame_h, my2+pad_y)
            )

            for class_id in (CLASS_RED, CLASS_YELLOW):
                filtered = []
                for score, box in candidates[class_id]:
                    area = (box[2]-box[0])*(box[3]-box[1])
                    cx = (box[0]+box[2])/2
                    cy = (box[1]+box[3])/2

                    # Payload must be genuinely smaller than the main box.
                    if area > SMALL_MAX_MAIN_AREA_RATIO * main_area:
                        continue
                    if (box[2]-box[0]) > 0.65*(mx2-mx1):
                        continue
                    if (box[3]-box[1]) > 0.65*(my2-my1):
                        continue

                    # It must be spatially related to the box.
                    if not (
                        vicinity[0] <= cx <= vicinity[2] and
                        vicinity[1] <= cy <= vicinity[3]
                    ):
                        continue

                    filtered.append((score, box))

                candidates[class_id] = filtered
        else:
            # Without a trustworthy main box, do not invent payloads.
            candidates[CLASS_RED] = []
            candidates[CLASS_YELLOW] = []

        self._last_candidate_lists = {
            c: list(candidates[c]) for c in candidates
        }

        selected = {
            c: self._select_candidate(candidates[c], previous[c])
            for c in candidates
        }
        self._last_boxes = selected
        return selected

    def _pose_points(self, frame):
        h, w = frame.shape[:2]
        result = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        points = []
        wrists = {"left": None, "right": None}
        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            for side, landmark_id in (("left", self.mp_pose.PoseLandmark.LEFT_WRIST),
                                      ("right", self.mp_pose.PoseLandmark.RIGHT_WRIST)):
                point = lm[landmark_id]
                if point.visibility >= 0.20 and 0 <= point.x <= 1 and 0 <= point.y <= 1:
                    wrists[side] = (int(point.x*w), int(point.y*h))
            landmarks = (self.mp_pose.PoseLandmark.LEFT_WRIST,
                         self.mp_pose.PoseLandmark.RIGHT_WRIST,
                         self.mp_pose.PoseLandmark.LEFT_INDEX,
                         self.mp_pose.PoseLandmark.RIGHT_INDEX,
                         self.mp_pose.PoseLandmark.LEFT_THUMB,
                         self.mp_pose.PoseLandmark.RIGHT_THUMB)
            for landmark_id in landmarks:
                point = lm[landmark_id]
                if point.visibility >= 0.15 and 0 <= point.x <= 1 and 0 <= point.y <= 1:
                    points.append((int(point.x*w), int(point.y*h)))
        return points, wrists, result

    @staticmethod
    def _point_in_box(point, box, margin=TOUCH_MARGIN_PX):
        if box is None:
            return False
        x, y = point
        return (box[0]-margin) <= x <= (box[2]+margin) and (box[1]-margin) <= y <= (box[3]+margin)

    @staticmethod
    def _point_to_box_distance(point, box):
        if point is None or box is None:
            return float("inf")
        x, y = point
        dx = max(box[0] - x, 0.0, x - box[2])
        dy = max(box[1] - y, 0.0, y - box[3])
        return float(np.hypot(dx, dy))

    @classmethod
    def _hand_near_object(cls, point, box):
        if point is None or box is None:
            return False
        bw = max(1.0, box[2]-box[0])
        bh = max(1.0, box[3]-box[1])
        threshold = max(OBJECT_HAND_DISTANCE_MIN_PX,
                        OBJECT_HAND_DISTANCE_FACTOR*np.hypot(bw, bh))
        return cls._point_to_box_distance(point, box) <= threshold

    def _step_geometry(self, small_boxes, main_box):
        stable = {}
        active = self.state.current_step()
        for class_id in (CLASS_RED, CLASS_YELLOW):
            raw = centroid_region(small_boxes.get(class_id), main_box)
            stable[class_id] = self.regions[class_id].update(raw)
            if self._object_phase[class_id] is None and stable[class_id] in ("loaded", "unloaded"):
                self._object_phase[class_id] = stable[class_id]
            is_active_unload = (active is not None and active["class_id"] == class_id
                                and active["rule"] == "unload")
            if stable[class_id] == "loaded" and small_boxes.get(class_id) is not None and not is_active_unload:
                b = small_boxes[class_id]
                self._load_anchors[class_id] = ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)
        return stable

    def _on_step_change(self, index, stable_regions):
        if index == self._active_step_index:
            return
        self._active_step_index = index
        step = experiment_steps[index] if index < len(experiment_steps) else None
        self._step_start_region = stable_regions.get(step["class_id"]) if step and step["rule"] in ("load", "unload") else None
        self._interaction_frame_count = 0
        self._last_confirmed_transition = None
        self._step_motion_count = 0
        self._object_interaction_seen = False
        self._hand_anchor = None
        self._hand_motion_frames = 0
        self._last_hand_center = None
        if step and step["rule"] == "unload":
            self._step_anchor = self._load_anchors[step["class_id"]]
            if self._step_anchor is None:
                track = self.object_trackers.get(step["class_id"])
                if track is not None and track.box is not None:
                    c = SmallObjectTracker._center(track.box)
                    self._step_anchor = (float(c[0]), float(c[1]))
        else:
            self._step_anchor = None
        self.touch_counts = {CLASS_MAIN: 0, CLASS_RED: 0, CLASS_YELLOW: 0}
        self.lid_tracker = LidGestureTracker()
        self.anomaly_latched = False
        self.no_touch_frames = 0

    def process_frame(self, frame):
        if frame is None or frame.ndim != 3:
            raise ValueError("process_frame expects a valid BGR image")

        self.start()

        raw_boxes = self._detect(frame, self.main_tracker.get_box())
        measured_main = raw_boxes[CLASS_MAIN]

        self.main_tracker.update(measured_main)
        main_box = self.main_tracker.get_box()
        geometry_main = measured_main if measured_main is not None else main_box

        small_boxes = {}
        small_sources = {}
        for class_id in (CLASS_RED, CLASS_YELLOW):
            track = self.object_trackers[class_id]
            small_boxes[class_id] = track.update(
                frame,
                list(self._last_candidate_lists.get(class_id, []))
            )
            small_sources[class_id] = track.source

        stable_regions = self._step_geometry(small_boxes, geometry_main)

        hand_points, wrists, pose_result = self._pose_points(frame)
        annotated = frame.copy()

        if pose_result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated, pose_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

        draw_boxes = {
            CLASS_MAIN: main_box,
            CLASS_RED: (
                small_boxes[CLASS_RED]
                if self.object_trackers[CLASS_RED].visible else None
            ),
            CLASS_YELLOW: (
                small_boxes[CLASS_YELLOW]
                if self.object_trackers[CLASS_YELLOW].visible else None
            ),
        }
        
        colors = {
            CLASS_MAIN: (255, 200, 0),   
            CLASS_RED: (0, 0, 255),      
            CLASS_YELLOW: (0, 255, 255), 
        }

        for class_id, box in draw_boxes.items():
            if box is None:
                continue
            x1, y1, x2, y2 = map(int, box)
            
            if class_id != CLASS_MAIN:
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                new_width = width * 0.80
                new_height = height * 0.80
                
                x1 = int(center_x - (new_width / 2))
                x2 = int(center_x + (new_width / 2))
                y1 = int(center_y - (new_height / 2))
                y2 = int(center_y + (new_height / 2))
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colors[class_id], 4)
            
            if class_id == CLASS_MAIN:
                suffix = f" [{self._main_measurement_source}]"
            else:
                suffix = f" [{small_sources[class_id]}]"
            label_text = f"{CLASS_NAMES[class_id]}{suffix}"
            
            label_y1 = max(0, y1 - 25)
            label_y2 = max(25, y1)
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w, label_y2), colors[class_id], -1)
            
            cv2.putText(
                annotated,
                label_text,
                (x1, label_y2 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

        current_index, _, done = self.state.snapshot_for_hud()
        self._on_step_change(current_index, stable_regions)
        current = self.state.current_step()
        now = time.monotonic()

        interaction_flags = {
            CLASS_MAIN: False,
            CLASS_RED: False,
            CLASS_YELLOW: False,
        }

        if current is not None:
            expected = current["class_id"]
            rule = current["rule"]

            if main_box is not None:
                main_threshold = max(
                    MAIN_TOUCH_MIN_PX,
                    MAIN_TOUCH_DISTANCE_FACTOR * np.hypot(
                        main_box[2]-main_box[0],
                        main_box[3]-main_box[1]
                    )
                )
                interaction_flags[CLASS_MAIN] = any(
                    self._point_to_box_distance(p, main_box) <= main_threshold
                    for p in hand_points
                )

            for class_id in (CLASS_RED, CLASS_YELLOW):
                box = small_boxes[class_id]
                interaction_flags[class_id] = any(
                    self._hand_near_object(p, box) for p in hand_points
                )

            any_touch = any(interaction_flags.values())
            if any_touch:
                self.no_touch_frames = 0
            else:
                self.no_touch_frames += 1
                if self.no_touch_frames >= ANOMALY_RELEASE_FRAMES:
                    self.anomaly_latched = False

            for class_id in self.touch_counts:
                self.touch_counts[class_id] = (
                    self.touch_counts[class_id] + 1
                    if interaction_flags[class_id] else 0
                )

            # Main-box contact is intentionally ignored for wrong-action
            # anomaly detection. During unloading/loading, the operator's hand
            # naturally goes inside/over the main box while handling a payload.
            # Red/yellow objects are still checked for genuine wrong-object use.
            wrong_held = [
                c for c, count in self.touch_counts.items()
                if c != CLASS_MAIN
                and c != expected
                and count >= TOUCH_SUSTAIN_FRAMES
            ]
            if wrong_held and not self.anomaly_latched:
                expected_action = current["action"]
                wrong_names = ", ".join(CLASS_NAMES[c] for c in wrong_held)
                anomaly_message = (
                    f"Anomaly detected. Step {current_index + 1} requires {expected_action}. "
                    f"Detected interaction with {wrong_names}. Please perform the current step correctly."
                )
                self.state.flag_anomaly(anomaly_message)
                self.anomaly_latched = True

            expected_touch = (
                self.touch_counts[expected] >= TOUCH_SUSTAIN_FRAMES
            )

            wrist_points = [p for p in wrists.values() if p is not None]
            hand_center = None
            if wrist_points:
                hand_center = np.mean(np.asarray(wrist_points, dtype=np.float32), axis=0)

            if rule in ("load", "unload") and expected_touch and hand_center is not None:
                if self._hand_anchor is None:
                    self._hand_anchor = hand_center.copy()
                self._last_hand_center = hand_center.copy()

            if rule in ("load", "unload") and self._hand_anchor is not None and hand_center is not None:
                hand_motion = float(np.linalg.norm(hand_center-self._hand_anchor))
                if hand_motion >= 25.0:
                    self._hand_motion_frames += 1
                else:
                    self._hand_motion_frames = 0

            success = False

            if rule == "touch":
                success = expected_touch

            elif rule in ("lid_open", "lid_close"):
                if rule == "lid_close":
                    # Closing gets its own tolerant detector so a quick real
                    # lid-closing motion is not missed.
                    gesture = self.lid_tracker.update(wrists, main_box, mode="close")
                    success = gesture == "down"
                else:
                    gesture = self.lid_tracker.update(wrists, main_box, mode="open")
                    success = gesture == "up"

            elif rule in ("load", "unload"):
                if expected_touch:
                    self._object_interaction_seen = True

                cls_state = stable_regions.get(expected)
                small = small_boxes.get(expected)

                if rule == "unload":
                    if self._step_anchor is None:
                        anchor = self._load_anchors.get(expected)
                        if anchor is not None:
                            self._step_anchor = anchor

                    moved_from_loaded_spot = False
                    displacement = 0.0

                    if self._step_anchor is not None and small is not None:
                        cx = (small[0] + small[2]) * 0.5
                        cy = (small[1] + small[3]) * 0.5
                        displacement = float(np.hypot(
                            cx - self._step_anchor[0],
                            cy - self._step_anchor[1]
                        ))
                        object_diag = np.hypot(
                            small[2] - small[0],
                            small[3] - small[1]
                        )
                        move_threshold = max(
                            UNLOAD_MIN_DISPLACEMENT_PX,
                            UNLOAD_MIN_DISPLACEMENT_FACTOR * object_diag
                        )
                        moved_from_loaded_spot = displacement >= move_threshold

                    object_evidence = (
                        cls_state == "unloaded" or
                        moved_from_loaded_spot
                    )
                    hand_evidence = (
                        self._hand_motion_frames >= UNLOAD_STABLE_FRAMES
                    )

                    if object_evidence or hand_evidence:
                        self._step_motion_count += 1
                    else:
                        self._step_motion_count = 0

                    if self._step_motion_count >= UNLOAD_STABLE_FRAMES:
                        self._last_confirmed_transition = "unloaded"

                    success = (
                        self._object_interaction_seen
                        and self._last_confirmed_transition == "unloaded"
                    )

                    if success:
                        self._object_phase[expected] = "unloaded"

                else:  # load
                    if (
                        self._object_phase[expected] == "unloaded"
                        and cls_state == "loaded"
                    ):
                        self._last_confirmed_transition = "loaded"

                    if self._last_confirmed_transition == "loaded":
                        self._step_motion_count += 1
                    else:
                        self._step_motion_count = 0

                    success = (
                        self._object_interaction_seen
                        and self._step_motion_count >= LOAD_STABLE_FRAMES
                    )

                    if success:
                        self._object_phase[expected] = "loaded"

            if success:
                self.state.advance()

        yellow_track = self.object_trackers[CLASS_YELLOW]
        red_track = self.object_trackers[CLASS_RED]

        def track_debug(track, box, cls):
            displacement = None
            anchor = self._step_anchor if (
                current is not None and current["class_id"] == cls
                and current["rule"] == "unload"
            ) else self._load_anchors.get(cls)
            if anchor is not None and box is not None:
                c = SmallObjectTracker._center(box)
                displacement = round(
                    float(np.linalg.norm(c - np.asarray(anchor))),
                    1
                )
            return {
                "source": track.source,
                "tracked": track.tracked,
                "visible": track.visible,
                "missed_frames": track.missed,
                "confidence": round(track.last_conf, 3),
                "displacement_px": displacement,
                "region": stable_regions.get(cls),
                "interaction_frames": self.touch_counts[cls],
            }

        audio_item = self.state.pop_audio()
        if audio_item:
            if isinstance(audio_item, tuple) and audio_item[0] == "__text__":
                self.voice.say_text(audio_item[1])
            else:
                self.voice.say(audio_item)

        event = self.state.pop_event()
        step_index, step_text, done = self.state.snapshot_for_hud()
        label = "EXPERIMENT COMPLETE" if done else step_text

        cv2.putText(
            annotated,
            f"STEP {min(step_index+1, len(experiment_steps))}/7: {label}",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 255), 2
        )

        completed = self.state.completed_history()
        if completed:
            done_text = "DONE: " + ", ".join(str(e["step_index"] + 1) for e in completed)
            cv2.putText(
                annotated, done_text, (20, 116),
                cv2.FONT_HERSHEY_SIMPLEX, .42, (80, 255, 80), 1
            )

        main_status = (
            f"MAIN: {self._main_measurement_source} | "
            f"{'tracked' if main_box is not None else 'NONE'} | "
            f"touch={self.touch_counts[CLASS_MAIN]}"
        )
        cv2.putText(
            annotated, main_status, (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX, .42, colors[CLASS_MAIN], 1
        )

        y = 80
        for name, cls, track in (
            ("YELLOW", CLASS_YELLOW, yellow_track),
            ("RED", CLASS_RED, red_track),
        ):
            status = (
                f"{name}: {track.source} | miss={track.missed} | "
                f"region={stable_regions.get(cls)} | touch={self.touch_counts[cls]}"
            )
            cv2.putText(
                annotated, status, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, .42, colors[cls], 1
            )
            y += 18

        hud = {
            "step_index": step_index,
            "step_text": step_text,
            "done": done,
            "touch_frames": dict(self.touch_counts),
            "regions": dict(stable_regions),
            "objects": {
                "yellow": track_debug(yellow_track, small_boxes[CLASS_YELLOW],
                                      CLASS_YELLOW),
                "red": track_debug(red_track, small_boxes[CLASS_RED],
                                   CLASS_RED),
            },
            "interaction_seen": self._object_interaction_seen,
            "step_motion_frames": self._step_motion_count,
            "confirmed_transition": self._last_confirmed_transition,
            "main": {
                "source": self._main_measurement_source,
                "tracked": main_box is not None,
                "touch_frames": self.touch_counts[CLASS_MAIN],
            },
            "hand_motion_frames": self._hand_motion_frames,
            "main_interaction_anomaly_disabled": True,
        }
        return annotated, hud

    def close(self):
        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            try:
                self.pose.close()
            except Exception:
                pass
            try:
                self.voice.stop()
            except Exception:
                pass

    def run_cli(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")
        try:
            while self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok:
                    break
                annotated, _ = self.process_frame(frame)
                
                cv2.namedWindow("ISRO-Astrotrack", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("ISRO-Astrotrack", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("ISRO-Astrotrack", annotated)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.close()
            cv2.destroyAllWindows()

def main():
    engine = AstrotrackEngine(camera_index=0)
    engine.run_cli()

if __name__ == "__main__":
    main()