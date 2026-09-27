"""ISRO-Astrotrack: conservative, GUI-ready vision/state pipeline.

Dependencies: opencv-python, numpy, ultralytics, mediapipe, pyttsx3 (Windows).
Place best.pt beside this file or pass its path to AstrotrackEngine.

process_frame(frame) is the GUI integration point. It returns (annotated_frame,
HUD_dict) and does not open a window or block the caller's event loop.
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

# YOLO performs class-aware NMS internally. Keep the semantic threshold high
# for the large main box, but inspect low-score small-box proposals only when
# their pixels independently match the expected box color.
MAIN_CONFIDENCE = 0.20
MAIN_FALLBACK_MIN_AREA_RATIO = 0.025
MAIN_FALLBACK_MAX_AREA_RATIO = 0.65
MAIN_FALLBACK_MIN_RECTANGULARITY = 0.45
MAIN_FALLBACK_CENTER_X_MIN = 0.15
MAIN_FALLBACK_CENTER_X_MAX = 0.85
MAIN_FALLBACK_CENTER_Y_MIN = 0.35
MAIN_TOUCH_DISTANCE_FACTOR = 0.32
MAIN_TOUCH_MIN_PX = 28
SMALL_PROPOSAL_CONFIDENCE = 0.10
NMS_IOU = 0.45
MAX_DETECTIONS = 40
MIN_DETECTION_SIDE_PX = 8

TOUCH_MARGIN_PX = 15
TOUCH_SUSTAIN_FRAMES = 5
ANOMALY_RELEASE_FRAMES = 5
REGION_STABLE_FRAMES = 6

# Lid gesture tuning. The wrist must contact the main box for a short period
# before vertical travel is evaluated.
LID_CONTACT_WINDOW_S = 1.2
LID_MIN_VERTICAL_TRAVEL_PX = 18
LID_QUALIFIED_WINDOW_S = 2.8

# Main-box tracker.
MAIN_TRACK_MAX_MISSED = 45

# Small-object detector constraints.
SMALL_MAX_MAIN_AREA_RATIO = 0.12

# Small-object temporal tracking. These values are intentionally configurable.
OBJECT_MAX_MISSED_FRAMES = 18
OBJECT_REDETECT_DISTANCE_FACTOR = 1.35
OBJECT_COLOR_SEARCH_FACTOR = 2.2
OBJECT_MIN_COLOR_FRACTION = 0.08
OBJECT_MAX_VELOCITY_PX = 90.0
OBJECT_VELOCITY_SMOOTHING = 0.55
OBJECT_MIN_TRACK_SIDE_PX = 8

# Human/object interaction.
OBJECT_HAND_DISTANCE_FACTOR = 0.90
OBJECT_HAND_DISTANCE_MIN_PX = 22

# Unload/load event confirmation.
UNLOAD_MIN_DISPLACEMENT_FACTOR = 0.50
UNLOAD_MIN_DISPLACEMENT_PX = 24
UNLOAD_STABLE_FRAMES = 4
LOAD_STABLE_FRAMES = 4

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
    """Reliable, serialized voice output for the GUI/vision loop.

    The vision loop never waits for speech. Messages are placed in a FIFO queue
    and a dedicated worker speaks them one at a time. WAV playback is preferred
    because it is deterministic on Windows; pyttsx3 is used as a fallback.
    """
    def __init__(self, library: dict, compile_files=True):
        self.texts = dict(library)
        self.files = {}
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._engine = None

        if compile_files:
            self.files = self._compile_files()

        self._worker.start()

    def _compile_files(self):
        compiled = {}
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astrotrack_audio")
            os.makedirs(folder, exist_ok=True)
            for key, text in self.texts.items():
                filepath = os.path.join(folder, f"{key}.wav")
                engine.save_to_file(text, filepath)
                compiled[key] = filepath
            engine.runAndWait()
        except Exception:
            # The worker will fall back to live pyttsx3 speech.
            compiled = {}
        return compiled

    def say(self, key):
        if key and key in self.texts:
            self._queue.put((key, self.texts[key]))

    def say_text(self, text):
        if text:
            self._queue.put((None, str(text)))

    def _run(self):
        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 170)
        except Exception:
            self._engine = None

        while not self._stop.is_set():
            try:
                key, text = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                played = False
                filepath = self.files.get(key) if key else None
                if filepath and os.path.exists(filepath):
                    try:
                        winsound.PlaySound(filepath, winsound.SND_FILENAME)
                        played = True
                    except Exception:
                        played = False
                if not played and self._engine is not None:
                    try:
                        self._engine.say(text)
                        self._engine.runAndWait()
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
    """Temporal tracker for the red/yellow experiment objects.

    YOLO is treated as a measurement source, not as the object's identity.
    When a hand occludes the object and YOLO misses it, the tracker predicts
    the last motion for a limited number of frames.  A color measurement is
    also searched near the predicted position.  A later YOLO detection is
    accepted only when it is spatially consistent with the existing track.

    This is deliberately lightweight so the prototype keeps its existing
    dependencies and GUI API.
    """
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
            # Smooth velocity rather than replacing it with detector jitter.
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
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if self.class_id == CLASS_YELLOW:
            # Yellow object can look pale under indoor lighting.
            return cv2.inRange(hsv, (15, 42, 50), (45, 255, 255))
        return (
            cv2.inRange(hsv, (0, 70, 45), (15, 255, 255)) |
            cv2.inRange(hsv, (165, 70, 45), (180, 255, 255))
        )

    def _color_recovery(self, frame, predicted):
        if predicted is None:
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
            if area < max(20.0, 0.015 * bw * bh):
                continue
            x, y, cw, ch = cv2.boundingRect(contour)
            if min(cw, ch) < OBJECT_MIN_TRACK_SIDE_PX:
                continue
            candidate = [
                float(rx1+x), float(ry1+y),
                float(rx1+x+cw), float(ry1+y+ch)
            ]
            candidate_area = self._area(candidate)
            # Reject tiny colored fragments and enormous unrelated regions.
            if candidate_area < 0.08 * self._area(predicted):
                continue
            if candidate_area > 4.0 * self._area(predicted):
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
            score = color_fraction * 2.0 - distance / max(bw, bh, 1.0)
            if score > best_score:
                best_score = score
                best = candidate
        return best

    def update(self, frame, detections):
        """Update from YOLO detections, otherwise predict/recover by color."""
        predicted = self._predicted_box()

        if detections:
            if self.box is None:
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
                # During a normal frame, stay close to the established identity.
                if dist <= OBJECT_REDETECT_DISTANCE_FACTOR * scale or iou >= 0.05:
                    valid.append((conf, box, dist, iou))
            if valid:
                valid.sort(key=lambda x: x[0] + 0.25*x[3] - 0.01*x[2]/scale,
                           reverse=True)
                self._accept(valid[0][1], valid[0][0], "yolo")
                return self.box

        recovered = self._color_recovery(frame, predicted)
        if recovered is not None:
            self._accept(recovered, min(0.5, max(0.18, self.last_conf)),
                         "color")
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
    """8-state constant-velocity Kalman filter for main-box [x1,y1,x2,y2]."""
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
    """Debounce loaded/unloaded labels; missing detections break the run."""
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
    """Track each wrist separately; latch a 10-frame touch before judging travel.

    The hand may leave the box rectangle during a real opening/closing motion.
    Once contact has been sustained, that wrist is followed briefly outside the
    rectangle so the gesture is not lost at the exact moment of lid movement.
    """
    def __init__(self):
        self.samples = {"left": deque(), "right": deque()}
        self.touch_streak = {"left": 0, "right": 0}
        self.anchor_y = {"left": None, "right": None}
        self.qualified_until = {"left": 0.0, "right": 0.0}

    def reset(self):
        self.__init__()

    def update(self, wrists, main_box):
        now = time.monotonic()
        for side in ("left", "right"):
            point = wrists.get(side)
            history = self.samples[side]
            touching = (
                point is not None and main_box is not None and
                AstrotrackEngine._point_to_box_distance(point, main_box)
                <= max(
                    MAIN_TOUCH_MIN_PX,
                    MAIN_TOUCH_DISTANCE_FACTOR * np.hypot(
                        main_box[2]-main_box[0],
                        main_box[3]-main_box[1]
                    )
                )
            )
            if point is not None:
                history.append((now, float(point[1])))
                while history and now-history[0][0] > LID_QUALIFIED_WINDOW_S:
                    history.popleft()
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
                # A short trailing median rejects landmark jitter without
                # averaging away a quick, intentional lid movement.
                recent = [y for _, y in list(history)[-3:]]
                if recent:
                    travel = float(np.median(recent)) - self.anchor_y[side]
                    if abs(travel) >= LID_MIN_VERTICAL_TRAVEL_PX:
                        self.reset()
                        return "down" if travel > 0 else "up"
        return None


def centroid_region(small_box, main_box):
    """Classify against an inset loading zone rather than the whole main bbox.

    A box resting in front of the container can overlap the outer detector bbox.
    It is still "unloaded" unless its center is in the inset loading interior.
    """
    if small_box is None or main_box is None:
        return None
    cx = (small_box[0] + small_box[2]) / 2.0
    cy = (small_box[1] + small_box[3]) / 2.0
    x1, y1, x2, y2 = main_box
    pad_x, pad_y = 0.15*(x2-x1), 0.15*(y2-y1)
    inside_core = x1+pad_x <= cx <= x2-pad_x and y1+pad_y <= cy <= y2-pad_y
    return "loaded" if inside_core else "unloaded"


class AstrotrackEngine:
    """GUI-ready engine. process_frame has no imshow/waitKey calls."""
    def __init__(self, model_path=None, camera_index=0, compile_audio_files=True):
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
        # Preserve identity through competing same-class detections while
        # still allowing a real object move (confidence remains a strong term).
        def score(item):
            conf, box = item
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            scale = max(1.0, np.hypot(previous_box[2]-previous_box[0], previous_box[3]-previous_box[1]))
            return conf - min(0.20, np.hypot(cx-old_cx, cy-old_cy) / scale * 0.025)
        return max(candidates, key=score)[1]

    @staticmethod
    def _fallback_main_box(frame, previous_box=None):
        """Find the large physical container when YOLO misses class 0.

        This is intentionally constrained to the lower/central workspace. It is
        a recovery measurement, not a replacement for YOLO.
        """
        h, w = frame.shape[:2]
        rx1, rx2 = int(MAIN_FALLBACK_CENTER_X_MIN*w), int(MAIN_FALLBACK_CENTER_X_MAX*w)
        ry1, ry2 = int(MAIN_FALLBACK_CENTER_Y_MIN*h), h
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 35, 110)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # The prototype box is a large light/neutral rectangle. Add a second
        # recovery path based on brightness in the lower-central workspace.
        # This is especially useful when the trained model misses the box
        # because the lid/person overlaps it.
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = cv2.threshold(gray_full, 105, 255, cv2.THRESH_BINARY)[1]
        bright[:int(0.50*h), :] = 0
        bright[:, :int(0.10*w)] = 0
        bright[:, int(0.90*w):] = 0
        bright = cv2.morphologyEx(
            bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2
        )
        bright = cv2.morphologyEx(
            bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
        )
        # Convert the brightness mask to the same local ROI coordinates as
        # the edge contours so bounding boxes are transformed exactly once.
        bright_roi = bright[ry1:ry2, rx1:rx2]
        bright_contours, _ = cv2.findContours(
            bright_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        frame_area = float(w*h)
        best = None
        best_score = -1e9

        all_contours = list(contours)
        for contour in bright_contours:
            area = cv2.contourArea(contour)
            if area >= MAIN_FALLBACK_MIN_AREA_RATIO*frame_area:
                all_contours.append(contour)

        for contour in all_contours:
            area = float(cv2.contourArea(contour))
            if area < MAIN_FALLBACK_MIN_AREA_RATIO*frame_area:
                continue
            if area > MAIN_FALLBACK_MAX_AREA_RATIO*frame_area:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 40 or bh < 30:
                continue
            box_area = float(bw*bh)
            rectangularity = area / max(1.0, box_area)
            if rectangularity < 0.30:
                continue

            gx1, gy1, gx2, gy2 = rx1+x, ry1+y, rx1+x+bw, ry1+y+bh
            cx, cy = (gx1+gx2)/2.0, (gy1+gy2)/2.0
            if cy < max(MAIN_FALLBACK_CENTER_Y_MIN, 0.42)*h:
                continue

            # Prefer a large, central, lower object.
            area_score = min(1.0, box_area/(0.25*frame_area))
            center_score = 1.0 - min(1.0, abs(cx-w/2)/(0.5*w))
            lower_score = min(1.0, max(0.0, (cy/h-0.45)/0.50))

            continuity = 0.0
            if previous_box is not None:
                pcx = (previous_box[0]+previous_box[2])/2
                pcy = (previous_box[1]+previous_box[3])/2
                scale = max(40.0, np.hypot(
                    previous_box[2]-previous_box[0],
                    previous_box[3]-previous_box[1]))
                dist = np.hypot(cx-pcx, cy-pcy)
                continuity = max(0.0, 1.0-dist/(2.0*scale))

            score = (
                2.2*area_score +
                1.2*rectangularity +
                0.8*center_score +
                0.7*lower_score +
                1.5*continuity
            )
            if score > best_score:
                best_score = score
                best = [float(gx1), float(gy1), float(gx2), float(gy2)]

        return best

    def _detect(self, frame, tracked_main_box=None):
        # Small objects in the supplied recording score below .60 or are assigned
        # the wrong class. Gather candidates, then validate them with pixel color.
        # The main container still requires .60 semantic confidence.
        result = self.model.predict(frame, conf=SMALL_PROPOSAL_CONFIDENCE, iou=NMS_IOU,
                                    max_det=MAX_DETECTIONS, verbose=False)[0]
        candidates = {CLASS_MAIN: [], CLASS_RED: [], CLASS_YELLOW: []}
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_masks = {
            CLASS_RED: (cv2.inRange(hsv, (0, 85, 45), (12, 255, 255)) |
                        cv2.inRange(hsv, (168, 85, 45), (180, 255, 255))),
            CLASS_YELLOW: cv2.inRange(hsv, (18, 85, 70), (40, 255, 255)),
        }

        def color_fraction(box, class_id):
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                return 0.0
            return cv2.countNonZero(color_masks[class_id][y1:y2, x1:x2]) / ((x2-x1)*(y2-y1))

        if result.boxes is not None and len(result.boxes):
            coords = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.cls.int().cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for box, class_id, score in zip(coords, ids, scores):
                class_id = int(class_id)
                if class_id not in candidates:
                    continue
                x1, y1, x2, y2 = map(float, box)
                if x2-x1 < MIN_DETECTION_SIDE_PX or y2-y1 < MIN_DETECTION_SIDE_PX:
                    continue
                if class_id == CLASS_MAIN:
                    if score < MAIN_CONFIDENCE:
                        continue
                elif score < 0.10 or color_fraction(box, class_id) < 0.10:
                    # Reject the high-count red hallucinations unless the actual
                    # pixels in the candidate contain a meaningful red region.
                    continue
                candidates[class_id].append((float(score), [x1,y1,x2,y2]))

        # Main-box recovery: if YOLO has no usable class-0 measurement,
        # search for the large physical container in the constrained workspace.
        previous = getattr(self, "_last_boxes", {c: None for c in candidates})
        frame_area = float(frame.shape[0] * frame.shape[1])

        # Reject a class-0 YOLO box that is obviously too small or too high in
        # the image to be the physical main container.
        main_yolo_valid = False
        if candidates[CLASS_MAIN]:
            best_main = max(candidates[CLASS_MAIN], key=lambda x: x[0])[1]
            main_area = (best_main[2]-best_main[0]) * (best_main[3]-best_main[1])
            main_cy = (best_main[1]+best_main[3]) * 0.5
            main_yolo_valid = (
                main_area >= 0.02 * frame_area and
                main_cy >= 0.35 * frame.shape[0]
            )

        fallback_main = self._fallback_main_box(
            frame,
            previous.get(CLASS_MAIN)
        )

        if not main_yolo_valid:
            if fallback_main is not None:
                candidates[CLASS_MAIN] = [(0.19, fallback_main)]
                self._main_measurement_source = "geometry"
            else:
                candidates[CLASS_MAIN] = []
                self._main_measurement_source = "none"
        else:
            self._main_measurement_source = "yolo"

        # Color segmentation recovers a real red/yellow object when YOLO misses
        # it or confuses the class. Restrict proposals to the main-box vicinity,
        # with plausible object dimensions, to exclude unrelated room pixels.
        kernel = np.ones((3, 3), np.uint8)
        main_hint = (candidates[CLASS_MAIN][0][1] if candidates[CLASS_MAIN]
                     else previous[CLASS_MAIN] if previous[CLASS_MAIN] is not None
                     else tracked_main_box)
        if main_hint is not None:
            mx1, my1, mx2, my2 = main_hint
            pad_x, pad_y = 0.85*(mx2-mx1), 0.85*(my2-my1)
            vicinity = (max(0, int(mx1-pad_x)), max(0, int(my1-pad_y)),
                        min(frame.shape[1], int(mx2+pad_x)), min(frame.shape[0], int(my2+pad_y)))
            main_area = max(1.0, (mx2-mx1)*(my2-my1))
            # Person-sized false positives were being called small boxes. Limit
            # semantic small-object candidates relative to the container size.
            for class_id in (CLASS_RED, CLASS_YELLOW):
                candidates[class_id] = [
                    (score, box) for score, box in candidates[class_id]
                    if ((box[2]-box[0])*(box[3]-box[1]) <= SMALL_MAX_MAIN_AREA_RATIO*main_area
                        and (box[2]-box[0]) <= 0.65*(mx2-mx1)
                        and (box[3]-box[1]) <= 0.65*(my2-my1))
                ]
            min_area = max(35, frame.shape[0]*frame.shape[1]*0.00020)
            for class_id, mask in color_masks.items():
                cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < min_area or area > SMALL_MAX_MAIN_AREA_RATIO*main_area:
                        continue
                    x, y, bw, bh = cv2.boundingRect(contour)
                    if min(bw, bh) < MIN_DETECTION_SIDE_PX or not (0.35 <= bw/max(1,bh) <= 2.8):
                        continue
                    if x+bw < vicinity[0] or x > vicinity[2] or y+bh < vicinity[1] or y > vicinity[3]:
                        continue
                    box = [float(x), float(y), float(x+bw), float(y+bh)]
                    if color_fraction(box, class_id) >= 0.14:
                        # Color candidates rank below validated YOLO proposals,
                        # but remain eligible when they are the only detection.
                        candidates[class_id].append((0.18, box))
        else:
            # Before the main-box tracker initializes, retain only genuinely
            # small candidates; this prevents torso-sized class hallucinations.
            frame_area = float(frame.shape[0]*frame.shape[1])
            for class_id in (CLASS_RED, CLASS_YELLOW):
                candidates[class_id] = [
                    (score, box) for score, box in candidates[class_id]
                    if ((box[2]-box[0])*(box[3]-box[1]) <= 0.02*frame_area
                        and (box[2]-box[0]) <= 0.35*frame.shape[1]
                        and (box[3]-box[1]) <= 0.35*frame.shape[0])
                ]
        # Keep all validated proposals for the temporal object trackers.
        # _last_boxes remains the single best semantic box for compatibility,
        # while _last_candidate_lists lets a tracker recover from an occlusion
        # without being forced to accept a bad single-frame winner.
        self._last_candidate_lists = {
            c: list(candidates[c]) for c in candidates
        }
        selected = {c: self._select_candidate(candidates[c], previous[c]) for c in candidates}
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
            # Keep the last known in-container center as the unloading reference.
            # Freeze it during that object's unload step so moving it forward
            # cannot drag the reference along with it.
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
        """Process one BGR frame; returns (annotated_frame, HUD_dict).

        The GUI contract is unchanged.  Small-object state is based on a
        temporal tracker rather than requiring YOLO to succeed on every frame.
        """
        if frame is None or frame.ndim != 3:
            raise ValueError("process_frame expects a valid BGR image")

        self.start()

        # 1) YOLO provides semantic measurements.
        raw_boxes = self._detect(frame, self.main_tracker.get_box())
        measured_main = raw_boxes[CLASS_MAIN]

        # 2) Main container gets its existing Kalman bridge.
        self.main_tracker.update(measured_main)
        main_box = self.main_tracker.get_box()
        geometry_main = measured_main if measured_main is not None else main_box

        # 3) Red/yellow get their own temporal tracks.
        small_boxes = {}
        small_sources = {}
        for class_id in (CLASS_RED, CLASS_YELLOW):
            track = self.object_trackers[class_id]
            small_boxes[class_id] = track.update(
                frame,
                list(self._last_candidate_lists.get(class_id, []))
            )
            small_sources[class_id] = track.source

        # Geometry must use tracked boxes, including predicted frames.
        stable_regions = self._step_geometry(small_boxes, geometry_main)

        hand_points, wrists, pose_result = self._pose_points(frame)
        annotated = frame.copy()

        if pose_result.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                annotated, pose_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )

        # Draw main + tracked objects.
        draw_boxes = {
            CLASS_MAIN: main_box,
            CLASS_RED: small_boxes[CLASS_RED],
            CLASS_YELLOW: small_boxes[CLASS_YELLOW],
        }
        
        # Ultra-High Contrast Colors (BGR Format)
        colors = {
            CLASS_MAIN: (255, 200, 0),   # Bright Cyan/Blue
            CLASS_RED: (0, 0, 255),      # Pure Red
            CLASS_YELLOW: (0, 255, 255), # Pure Yellow
        }

        for class_id, box in draw_boxes.items():
            if box is None:
                continue
            x1, y1, x2, y2 = map(int, box)
            
            # 1. Draw a thick, highly visible border (Thickness = 4)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colors[class_id], 4)
            
            # 2. Determine label text
            if class_id == CLASS_MAIN:
                suffix = f" [{self._main_measurement_source}]"
            else:
                suffix = f" [{small_sources[class_id]}]"
            label_text = f"{CLASS_NAMES[class_id]}{suffix}"
            
            # 3. Draw a solid background box for the text to make it readable
            # Ensure the label doesn't draw off the top of the screen
            label_y1 = max(0, y1 - 25)
            label_y2 = max(25, y1)
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w, label_y2), colors[class_id], -1)
            
            # 4. Draw black text over the solid color block for maximum contrast
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

            # Main-box interaction uses distance to the physical box, not only
            # strict point-in-rectangle containment. This is important when the
            # hand touches the top/front edge and the YOLO rectangle is imperfect.
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

            # For red/yellow use distance-to-box, so a hand touching an edge
            # still counts even when the detector box is slightly imperfect.
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

            # Consecutive interaction frames.
            for class_id in self.touch_counts:
                self.touch_counts[class_id] = (
                    self.touch_counts[class_id] + 1
                    if interaction_flags[class_id] else 0
                )

            wrong_held = [
                c for c, count in self.touch_counts.items()
                if c != expected and count >= TOUCH_SUSTAIN_FRAMES
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

            # Track the hand's motion after interaction. This becomes a second
            # source of evidence when the object is temporarily hidden by the hand.
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
                gesture = self.lid_tracker.update(wrists, main_box)
                wanted = "up" if rule == "lid_open" else "down"
                success = gesture == wanted

            elif rule in ("load", "unload"):
                # A hand/object interaction starts the event, but is NOT itself
                # sufficient to complete it.
                if expected_touch:
                    self._object_interaction_seen = True

                cls_state = stable_regions.get(expected)
                small = small_boxes.get(expected)

                if rule == "unload":
                    # Freeze the original loaded position. Never move the anchor
                    # along with the object during unloading.
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

                    # A real unload can temporarily hide the object completely.
                    # Therefore object displacement, zone exit, OR sustained hand
                    # motion away from the loaded position can contribute evidence.
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
                    # Loading is confirmed by the object returning to the
                    # interior loaded zone after interaction.
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

        # Debug HUD: this is deliberately visible so tuning can be done from
        # the actual recording instead of guessing.
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

        # Compact on-frame debug status.
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
        }
        return annotated, hud

    def close(self):
        """Release camera/pose resources and stop the voice worker."""
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
        """Optional standalone debug loop; GUI callers should call process_frame."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")
        try:
            while self.cap.isOpened():
                ok, frame = self.cap.read()
                if not ok:
                    break
                annotated, _ = self.process_frame(frame)
                
                # Native OpenCV fullscreen execution
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