import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import librosa
import tempfile
import os

def record_audio(duration=30, sample_rate=44100):
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    temp_file = tempfile.mktemp(suffix='.wav')
    wav.write(temp_file, sample_rate, audio)
    return temp_file

def analyze_voice(audio_path):
    try:
        y, sr = librosa.load(audio_path)

        # Check if audio has content
        if len(y) == 0:
            raise ValueError("Empty audio")

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        energy = np.mean(librosa.feature.rms(y=y))
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        silence_threshold = 0.01
        silence_ratio = np.sum(
            np.abs(y) < silence_threshold) / len(y)

        confidence_score = min(10, max(1, float(energy) * 1000))
        clarity_score = min(
            10, max(1, 10 - (silence_ratio * 10)))
        pace_score = min(
            10, max(1, 10 - abs(float(tempo) - 120) / 20))

        return {
            "confidence_score": round(confidence_score, 1),
            "clarity_score": round(clarity_score, 1),
            "pace_score": round(pace_score, 1),
            "overall_voice_score": round(
                (confidence_score +
                 clarity_score +
                 pace_score) / 3, 1),
            "tempo": round(float(tempo), 1),
            "silence_ratio": round(float(silence_ratio * 100), 1),
            "feedback": get_voice_feedback(
                confidence_score, clarity_score,
                pace_score, silence_ratio)
        }
    except Exception as e:
        return {
            "confidence_score": 0.0,
            "clarity_score": 0.0,
            "pace_score": 0.0,
            "overall_voice_score": 0.0,
            "tempo": 0.0,
            "silence_ratio": 0.0,
            "feedback": "Voice analysis unavailable"
        }

def get_voice_feedback(confidence, clarity, pace, silence):
    feedback = []
    if confidence < 5:
        feedback.append("Speak louder and with more confidence.")
    else:
        feedback.append("Good voice confidence.")
    if clarity < 5:
        feedback.append("Try to speak more clearly.")
    else:
        feedback.append("Clear speech detected.")
    if pace < 5:
        feedback.append("Adjust your speaking pace.")
    else:
        feedback.append("Good speaking pace.")
    if silence > 0.4:
        feedback.append("Too many pauses - practice fluency.")
    else:
        feedback.append("Good fluency.")
    return " | ".join(feedback)