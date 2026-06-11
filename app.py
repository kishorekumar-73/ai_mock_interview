import cv2
import numpy as np
import threading
import time
from collections import deque

# MediaPipe imports
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# ── Landmark indices for eye/head analysis ──
# Left eye: 33, 160, 158, 133, 153, 144
# Right eye: 362, 385, 387, 263, 373, 380
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Nose tip and chin for head pose
NOSE_TIP = 1
CHIN = 152
LEFT_TEMPLE = 234
RIGHT_TEMPLE = 454


def _eye_aspect_ratio(landmarks, eye_indices, w, h):
    """Calculate EAR to detect if eyes are open."""
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
           for i in eye_indices]
    # Vertical distances
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # Horizontal distance
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def _head_pose_score(landmarks, w, h):
    """
    Returns a 0-10 score based on how centred and upright
    the head is. Looks at horizontal tilt and vertical nod.
    """
    nose = landmarks[NOSE_TIP]
    chin = landmarks[CHIN]
    left = landmarks[LEFT_TEMPLE]
    right = landmarks[RIGHT_TEMPLE]

    # Horizontal centre ratio (0 = far left, 1 = far right)
    centre_x = nose.x  # should be ~0.5
    h_offset = abs(centre_x - 0.5)  # 0 = perfect, 0.5 = edge

    # Vertical: nose should be between chin and top of face
    v_pos = nose.y  # should be ~0.4-0.6
    v_offset = abs(v_pos - 0.45)

    # Combined penalty
    penalty = (h_offset * 6) + (v_offset * 4)
    score = max(1.0, min(10.0, 10.0 - penalty * 10))
    return round(score, 1)


def _analyse_landmarks(frame):
    """
    Run MediaPipe FaceMesh on a single frame.
    Returns a dict with:
      - face_detected: bool
      - eye_contact: float 0-10
      - head_pose: float 0-10
      - ear_left / ear_right: float (eye openness)
    """
    h, w = frame.shape[:2]
    result_data = {
        "face_detected": False,
        "eye_contact": 0.0,
        "head_pose": 0.0,
        "ear_left": 0.0,
        "ear_right": 0.0,
    }
    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                return result_data

            lm = results.multi_face_landmarks[0].landmark
            result_data["face_detected"] = True

            ear_l = _eye_aspect_ratio(lm, LEFT_EYE, w, h)
            ear_r = _eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            result_data["ear_left"] = ear_l
            result_data["ear_right"] = ear_r

            # Eye contact: higher EAR = eyes more open = looking at screen
            avg_ear = (ear_l + ear_r) / 2
            # EAR typically 0.15-0.35; map to 0-10
            eye_score = min(10.0, max(0.0, avg_ear * 35))
            result_data["eye_contact"] = round(eye_score, 1)

            result_data["head_pose"] = _head_pose_score(lm, w, h)

    except Exception:
        pass

    return result_data


def _calculate_confidence(analysis):
    """
    Derive a confidence score 0-10 from mediapipe analysis.
    Combines eye contact + head pose.
    """
    if not analysis["face_detected"]:
        return 0.0
    eye = analysis["eye_contact"]
    pose = analysis["head_pose"]
    score = (eye * 0.5) + (pose * 0.5)
    return round(min(10.0, max(1.0, score)), 1)


def _infer_emotion(analysis):
    """
    MediaPipe doesn't do emotion detection, so we infer a
    simple label from posture signals instead.
    """
    if not analysis["face_detected"]:
        return "not detected"
    eye = analysis["eye_contact"]
    pose = analysis["head_pose"]
    if eye >= 7 and pose >= 7:
        return "confident"
    elif eye >= 5 and pose >= 5:
        return "neutral"
    elif pose < 4:
        return "distracted"
    elif eye < 3:
        return "nervous"
    else:
        return "neutral"


# ──────────────────────────────────────────────────────────
# Public helper — drop-in replacement for DeepFace version
# ──────────────────────────────────────────────────────────

def get_face_feedback(confidence, dominant_emotion):
    feedback = []
    if confidence == 0.0:
        return ("Face not detected. "
                "Please ensure camera is on and face is visible.")
    if confidence >= 7:
        feedback.append("Good facial confidence!")
    elif confidence >= 5:
        feedback.append("Try to appear more confident.")
    else:
        feedback.append("Practice maintaining composure.")

    emotion_map = {
        "confident": "Great confident posture!",
        "neutral": "Professional neutral expression maintained.",
        "distracted": "Keep your head straight and look at the camera.",
        "nervous": "Try to relax — take a deep breath.",
        "not detected": "Face not visible.",
    }
    feedback.append(
        emotion_map.get(dominant_emotion, "Expression looks okay."))
    feedback.append("Tip: Maintain eye contact and sit upright.")
    return " | ".join(feedback)


def analyze_frame_emotions(frame):
    """
    Drop-in replacement for the DeepFace version.
    Returns a dict that mimics the old emotions dict shape
    so the rest of the code stays unchanged.
    """
    analysis = _analyse_landmarks(frame)
    confidence = _calculate_confidence(analysis)
    emotion = _infer_emotion(analysis)

    # Return in the same shape the old code expected
    # (dominant_emotion key used elsewhere)
    return {
        "face_detected": analysis["face_detected"],
        "eye_contact": analysis["eye_contact"],
        "head_pose": analysis["head_pose"],
        "confidence_score": confidence,
        "dominant_emotion": emotion,
    }


def calculate_confidence_from_emotions(emotions):
    return emotions.get("confidence_score", 7.0)


# ──────────────────────────────────────────────────────────
# ContinuousPostureMonitor — same public API as before
# ──────────────────────────────────────────────────────────

class ContinuousPostureMonitor:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.analysis_history = deque(maxlen=200)
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
            # Analyse every 30 frames (~1 s at 30 fps)
            if frame_count % 30 == 0:
                analysis = _analyse_landmarks(frame)
                confidence = _calculate_confidence(analysis)
                with self.lock:
                    self.analysis_history.append(analysis)
                    self.confidence_history.append(confidence)
            time.sleep(0.001)

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    def get_snapshot(self, question_number):
        with self.lock:
            if not self.confidence_history:
                return {
                    "question": question_number,
                    "confidence_score": 0.0,
                    "dominant_emotion": "not detected",
                    "emotions": {},
                    "feedback": "No face detected during this question."
                }

            valid_conf = [c for c in self.confidence_history if c > 0]
            if not valid_conf:
                return {
                    "question": question_number,
                    "confidence_score": 0.0,
                    "dominant_emotion": "not detected",
                    "emotions": {},
                    "feedback": (
                        "Face not visible. "
                        "Please sit in front of camera.")
                }

            avg_confidence = round(
                sum(valid_conf) / len(valid_conf), 1)

            # Build average metrics from history
            if self.analysis_history:
                eye_vals = [
                    a["eye_contact"]
                    for a in self.analysis_history
                    if a["face_detected"]]
                pose_vals = [
                    a["head_pose"]
                    for a in self.analysis_history
                    if a["face_detected"]]
                avg_eye = (round(sum(eye_vals)/len(eye_vals), 1)
                           if eye_vals else 0.0)
                avg_pose = (round(sum(pose_vals)/len(pose_vals), 1)
                            if pose_vals else 0.0)
                # Infer dominant label from averages
                dummy = {
                    "face_detected": bool(eye_vals),
                    "eye_contact": avg_eye,
                    "head_pose": avg_pose,
                }
                dominant = _infer_emotion(dummy)
                avg_emotions = {
                    "eye_contact": avg_eye,
                    "head_pose": avg_pose,
                }
            else:
                avg_emotions = {}
                dominant = "neutral"

            return {
                "question": question_number,
                "confidence_score": avg_confidence,
                "dominant_emotion": dominant,
                "emotions": avg_emotions,
                "feedback": get_face_feedback(
                    avg_confidence, dominant)
            }

    def reset_history(self):
        with self.lock:
            self.analysis_history.clear()
            self.confidence_history.clear()


# ──────────────────────────────────────────────────────────
# analyze_video_file — same public API as before
# ──────────────────────────────────────────────────────────

def analyze_video_file(video_path):
    cap = cv2.VideoCapture(video_path)
    confidence_scores = []
    emotions = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 5 == 0:
            analysis = _analyse_landmarks(frame)
            conf = _calculate_confidence(analysis)
            emotion = _infer_emotion(analysis)
            if analysis["face_detected"]:
                confidence_scores.append(conf)
                emotions.append(emotion)

    cap.release()

    if confidence_scores:
        avg_confidence = round(
            sum(confidence_scores) / len(confidence_scores), 1)
        dominant = max(set(emotions), key=emotions.count)
    else:
        avg_confidence = 0.0
        dominant = "not detected"

    return {
        "confidence_score": avg_confidence,
        "dominant_emotion": dominant,
        "emotions": {"confidence": avg_confidence},
        "feedback": get_face_feedback(avg_confidence, dominant)
    }
