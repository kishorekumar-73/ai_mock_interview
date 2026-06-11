import cv2
import numpy as np
import threading
import time
from collections import deque

# ── MediaPipe new Tasks API (mediapipe >= 0.10) ──────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# FaceLandmarker is used instead of the deprecated mp.solutions.face_mesh
_FL_OPTIONS = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(
        model_asset_path=None,      # will be downloaded on first use
        delegate=mp_python.BaseOptions.Delegate.CPU,
    ),
    output_face_blendshapes=True,   # gives us expression scores
    num_faces=1,
)

# Lazy-initialise the landmarker (heavyweight object) once per process
_landmarker = None
_landmarker_lock = threading.Lock()


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        with _landmarker_lock:
            if _landmarker is None:
                # Download the model bundle automatically
                import urllib.request, os, tempfile
                model_url = (
                    "https://storage.googleapis.com/mediapipe-models/"
                    "face_landmarker/face_landmarker/float16/1/"
                    "face_landmarker.task"
                )
                model_path = os.path.join(
                    tempfile.gettempdir(), "face_landmarker.task")
                if not os.path.exists(model_path):
                    urllib.request.urlretrieve(model_url, model_path)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=model_path,
                        delegate=mp_python.BaseOptions.Delegate.CPU,
                    ),
                    output_face_blendshapes=True,
                    num_faces=1,
                )
                _landmarker = mp_vision.FaceLandmarker.create_from_options(
                    options)
    return _landmarker


# ── Blendshape → pseudo-emotion mapping ──────────────────────────────────────
# MediaPipe blendshape names that correlate with each emotion bucket
_BLENDSHAPE_MAP = {
    "happy":    ["mouthSmileLeft", "mouthSmileRight", "cheekSquintLeft",
                 "cheekSquintRight"],
    "sad":      ["mouthFrownLeft", "mouthFrownRight", "browInnerUp"],
    "angry":    ["browDownLeft", "browDownRight", "noseSneerLeft",
                 "noseSneerRight"],
    "surprise": ["browOuterUpLeft", "browOuterUpRight",
                 "eyeWideLeft", "eyeWideRight", "jawOpen"],
    "fear":     ["eyeWideLeft", "eyeWideRight", "browInnerUp",
                 "mouthOpen"],
    "disgust":  ["noseSneerLeft", "noseSneerRight", "mouthLowerDownLeft",
                 "mouthLowerDownRight"],
    "neutral":  [],   # filled in by exclusion
}


def _blendshapes_to_emotions(blendshapes) -> dict:
    """Convert a list of FaceBlendshape objects to an emotion dict (0-100)."""
    bs = {b.category_name: b.score for b in blendshapes}

    raw = {}
    for emotion, keys in _BLENDSHAPE_MAP.items():
        if emotion == "neutral":
            continue
        scores = [bs.get(k, 0.0) for k in keys]
        raw[emotion] = (sum(scores) / len(scores) * 100) if scores else 0.0

    total_non_neutral = sum(raw.values())
    raw["neutral"] = max(0.0, 100.0 - total_non_neutral)

    # Normalise to 100
    grand_total = sum(raw.values())
    if grand_total > 0:
        raw = {k: round(v / grand_total * 100, 1) for k, v in raw.items()}

    return raw


def _analyze_frame(frame) -> dict:
    """Run face-landmark inference on a BGR frame; return emotion dict or {}."""
    try:
        landmarker = _get_landmarker()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if not result.face_blendshapes:
            return {}
        return _blendshapes_to_emotions(result.face_blendshapes[0])
    except Exception:
        return {}


def _calculate_confidence(emotions: dict) -> float:
    if not emotions:
        return 0.0
    if emotions.get("neutral", 0) >= 98:
        return 0.0
    positive = emotions.get("happy", 0) + emotions.get("neutral", 0)
    negative = (emotions.get("sad", 0) + emotions.get("angry", 0) +
                emotions.get("fear", 0) + emotions.get("disgust", 0))
    if positive + negative == 0:
        return 0.0
    return round(min(10, max(1, (positive / (positive + negative)) * 10)), 1)


def get_face_feedback(confidence: float, dominant_emotion: str) -> str:
    if confidence == 0.0:
        return ("Face not detected. "
                "Please ensure camera is on and face is visible.")
    feedback = []
    if confidence >= 7:
        feedback.append("Good facial confidence!")
    elif confidence >= 5:
        feedback.append("Try to appear more confident.")
    else:
        feedback.append("Practice smiling and maintaining composure.")

    emotion_map = {
        "happy":    "Positive expressions detected!",
        "neutral":  "Professional neutral expression maintained.",
        "fear":     "Try to relax and stay calm.",
        "sad":      "Try to relax and stay calm.",
        "angry":    "Watch your facial expressions.",
        "surprise": "Try to maintain a calm expression.",
        "disgust":  "Watch your facial expressions.",
    }
    feedback.append(emotion_map.get(dominant_emotion, "Expression looks okay."))
    feedback.append("Tip: Maintain eye contact and smile naturally.")
    return " | ".join(feedback)


# ── Public helper used by app.py for single-frame analysis ───────────────────
def analyze_frame_emotions(frame) -> dict:
    emotions = _analyze_frame(frame)
    if not emotions:
        return {"happy": 0, "neutral": 50, "sad": 0,
                "angry": 0, "surprise": 0, "fear": 0, "disgust": 0}
    return emotions


def calculate_confidence_from_emotions(emotions: dict) -> float:
    positive = emotions.get("happy", 0) + emotions.get("neutral", 0)
    negative = (emotions.get("sad", 0) + emotions.get("angry", 0) +
                emotions.get("fear", 0) + emotions.get("disgust", 0))
    if positive + negative == 0:
        return 7.0
    return round(min(10, max(1, (positive / (positive + negative)) * 10)), 1)


def analyze_video_file(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    emotion_totals = {k: 0.0 for k in
                      ["happy", "neutral", "sad", "angry",
                       "surprise", "fear", "disgust"]}
    frame_count = 0
    analyzed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 == 0:
            emotions = analyze_frame_emotions(frame)
            for emotion, value in emotions.items():
                if emotion in emotion_totals:
                    emotion_totals[emotion] += value
            analyzed_frames += 1
    cap.release()

    avg_emotions = ({k: v / analyzed_frames for k, v in emotion_totals.items()}
                    if analyzed_frames > 0 else emotion_totals)
    confidence_score = calculate_confidence_from_emotions(avg_emotions)
    dominant_emotion = max(avg_emotions, key=avg_emotions.get)
    return {
        "confidence_score": confidence_score,
        "dominant_emotion": dominant_emotion,
        "emotions": {k: round(v, 1) for k, v in avg_emotions.items()},
        "feedback": get_face_feedback(confidence_score, dominant_emotion),
    }


# ── Background monitor class ──────────────────────────────────────────────────
class ContinuousPostureMonitor:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.emotion_history = deque(maxlen=200)
        self.confidence_history = deque(maxlen=200)
        self.lock = threading.Lock()
        self._thread = None
        self.latest_frame = None

    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not accessible")
        self.is_running = True
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.is_running = False
        if self.cap:
            self.cap.release()

    def _monitor_loop(self):
        frame_count = 0
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            frame_count += 1
            with self.lock:
                self.latest_frame = frame.copy()
            if frame_count % 30 == 0:
                emotions = _analyze_frame(frame)
                confidence = _calculate_confidence(emotions)
                with self.lock:
                    self.emotion_history.append(emotions)
                    self.confidence_history.append(confidence)
            time.sleep(0.001)

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    def get_snapshot(self, question_number: int) -> dict:
        with self.lock:
            if not self.confidence_history:
                return {
                    "question": question_number,
                    "confidence_score": 0.0,
                    "dominant_emotion": "not detected",
                    "emotions": {},
                    "feedback": "No face detected during this question.",
                }
            valid_conf = [c for c in self.confidence_history if c > 0]
            if not valid_conf:
                return {
                    "question": question_number,
                    "confidence_score": 0.0,
                    "dominant_emotion": "not detected",
                    "emotions": {},
                    "feedback": ("Face not visible. "
                                 "Please sit in front of camera."),
                }
            avg_confidence = round(sum(valid_conf) / len(valid_conf), 1)

            avg_emotions = {}
            for emotion in ["happy", "neutral", "sad",
                            "angry", "surprise", "fear", "disgust"]:
                values = [e.get(emotion, 0)
                          for e in self.emotion_history if e]
                avg_emotions[emotion] = (round(sum(values) / len(values), 1)
                                         if values else 0.0)
            dominant = max(avg_emotions, key=avg_emotions.get)

            return {
                "question": question_number,
                "confidence_score": avg_confidence,
                "dominant_emotion": dominant,
                "emotions": avg_emotions,
                "feedback": get_face_feedback(avg_confidence, dominant),
            }

    def reset_history(self):
        with self.lock:
            self.emotion_history.clear()
            self.confidence_history.clear()
