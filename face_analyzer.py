import cv2
import numpy as np
import threading
import time
from collections import deque

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
                emotions = self._analyze_frame(frame)
                confidence = self._calculate_confidence(emotions)
                with self.lock:
                    self.emotion_history.append(emotions)
                    self.confidence_history.append(confidence)
            time.sleep(0.001)

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
        return None

    def _analyze_frame(self, frame):
        try:
            from deepface import DeepFace
            result = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )
            if isinstance(result, list):
                result = result[0]
            emotions = result.get('emotion', {})
            # Check if face actually detected
            # DeepFace returns all zeros or 100% neutral when no face
            total = sum(emotions.values())
            if total == 0:
                return {}
            return emotions
        except:
            return {}

    def _calculate_confidence(self, emotions):
        if not emotions:
            return 0.0
        # If neutral is 100% likely no face detected
        if emotions.get('neutral', 0) >= 98:
            return 0.0
        positive = (emotions.get('happy', 0) +
                   emotions.get('neutral', 0))
        negative = (emotions.get('sad', 0) +
                   emotions.get('angry', 0) +
                   emotions.get('fear', 0) +
                   emotions.get('disgust', 0))
        if positive + negative == 0:
            return 0.0
        return round(
            min(10, max(1,
                (positive / (positive + negative)) * 10)), 1)

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
            # Filter out zero confidence (no face)
            valid_conf = [
                c for c in self.confidence_history if c > 0]
            if not valid_conf:
                return {
                    "question": question_number,
                    "confidence_score": 0.0,
                    "dominant_emotion": "not detected",
                    "emotions": {},
                    "feedback": "Face not visible. Please sit in front of camera."
                }
            avg_confidence = round(
                sum(valid_conf) / len(valid_conf), 1)
            if self.emotion_history:
                avg_emotions = {}
                for emotion in ['happy', 'neutral', 'sad',
                               'angry', 'surprise',
                               'fear', 'disgust']:
                    values = [
                        e.get(emotion, 0)
                        for e in self.emotion_history
                        if e]
                    if values:
                        avg_emotions[emotion] = round(
                            sum(values) / len(values), 1)
                    else:
                        avg_emotions[emotion] = 0.0
                dominant = max(
                    avg_emotions, key=avg_emotions.get)
            else:
                avg_emotions = {}
                dominant = 'neutral'
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
            self.emotion_history.clear()
            self.confidence_history.clear()


def get_face_feedback(confidence, dominant_emotion):
    feedback = []
    if confidence == 0.0:
        return "Face not detected. Please ensure camera is on and face is visible."
    if confidence >= 7:
        feedback.append("Good facial confidence!")
    elif confidence >= 5:
        feedback.append("Try to appear more confident.")
    else:
        feedback.append("Practice smiling and maintaining composure.")

    emotion_map = {
        'happy': "Positive expressions detected!",
        'neutral': "Professional neutral expression maintained.",
        'fear': "Try to relax and stay calm.",
        'sad': "Try to relax and stay calm.",
        'angry': "Watch your facial expressions.",
        'surprise': "Try to maintain a calm expression.",
        'disgust': "Watch your facial expressions."
    }
    feedback.append(
        emotion_map.get(dominant_emotion,
                       "Expression looks okay."))
    feedback.append(
        "Tip: Maintain eye contact and smile naturally.")
    return " | ".join(feedback)


def analyze_frame_emotions(frame):
    try:
        from deepface import DeepFace
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )
        if isinstance(result, list):
            result = result[0]
        return result.get('emotion', {})
    except:
        return {
            'happy': 0, 'neutral': 50, 'sad': 0,
            'angry': 0, 'surprise': 0,
            'fear': 0, 'disgust': 0
        }


def calculate_confidence_from_emotions(emotions):
    positive = (emotions.get('happy', 0) +
               emotions.get('neutral', 0))
    negative = (emotions.get('sad', 0) +
               emotions.get('angry', 0) +
               emotions.get('fear', 0) +
               emotions.get('disgust', 0))
    if positive + negative == 0:
        return 7.0
    return round(
        min(10, max(1,
            (positive / (positive + negative)) * 10)), 1)


def analyze_video_file(video_path):
    cap = cv2.VideoCapture(video_path)
    emotion_totals = {
        'happy': 0, 'neutral': 0, 'sad': 0,
        'angry': 0, 'surprise': 0,
        'fear': 0, 'disgust': 0
    }
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
    if analyzed_frames > 0:
        avg_emotions = {
            k: v/analyzed_frames
            for k, v in emotion_totals.items()
        }
    else:
        avg_emotions = emotion_totals
    confidence_score = calculate_confidence_from_emotions(
        avg_emotions)
    dominant_emotion = max(avg_emotions, key=avg_emotions.get)
    return {
        "confidence_score": confidence_score,
        "dominant_emotion": dominant_emotion,
        "emotions": {
            k: round(v, 1) for k, v in avg_emotions.items()},
        "feedback": get_face_feedback(
            confidence_score, dominant_emotion)
    }