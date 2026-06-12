import gradio as gr
import cv2
import numpy as np
import tempfile
import os
import time
import threading
from collections import deque

from evaluator import (extract_resume_text, extract_candidate_name,
                       generate_questions, evaluate_answer, extract_score)
from report import generate_report
from pdf_report import generate_pdf_report

# ── Haar cascades (built into opencv-python-headless) ──
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ──────────────────────────────────────────────────────────
# Posture analysis helpers
# ──────────────────────────────────────────────────────────

def analyse_frame(frame_bgr):
    """Analyse a single BGR frame. Returns metrics dict."""
    if frame_bgr is None:
        return {"face_detected": False, "confidence_score": 0.0,
                "dominant_emotion": "not detected",
                "eye_contact": 0.0, "head_pose": 0.0}

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        return {"face_detected": False, "confidence_score": 0.0,
                "dominant_emotion": "not detected",
                "eye_contact": 0.0, "head_pose": 0.0}

    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    face_roi = gray[fy:fy + fh, fx:fx + fw]
    eyes = _eye_cascade.detectMultiScale(
        face_roi, scaleFactor=1.1, minNeighbors=5)
    eye_count = min(len(eyes), 2)

    face_cx = fx + fw / 2
    face_cy = fy + fh / 2
    h_off = abs(face_cx / w - 0.5) * 2
    v_off = abs(face_cy / h - 0.5) * 2
    centre_off = (h_off + v_off) / 2
    size_ratio = (fw * fh) / (w * h)

    eye_score = min(10.0, eye_count * 4.0 + 2.0)
    size_score = min(10.0, size_ratio * 80)
    head_score = max(1.0, min(10.0, size_score - centre_off * 5 + 5))
    conf = round((eye_score * 0.5 + head_score * 0.5), 1)

    if eye_score >= 7 and head_score >= 7:
        emotion = "confident"
    elif eye_score >= 5 and head_score >= 5:
        emotion = "neutral"
    elif head_score < 4:
        emotion = "distracted"
    else:
        emotion = "nervous"

    return {
        "face_detected": True,
        "confidence_score": conf,
        "dominant_emotion": emotion,
        "eye_contact": round(eye_score, 1),
        "head_pose": round(head_score, 1),
    }


def draw_overlay(frame_bgr, metrics):
    """Draw bounding box + live HUD on frame. Returns RGB."""
    frame = frame_bgr.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (fx, fy, fw, fh) in faces:
        conf = metrics.get("confidence_score", 0)
        color = (0, 220, 80) if conf >= 6 else (255, 200, 0) if conf >= 4 else (220, 50, 50)
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)
        cv2.putText(frame, f"Conf: {conf}/10",
                    (fx, fy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2)

    # HUD bar at top
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (30, 30, 30), -1)
    emotion = metrics.get("dominant_emotion", "---")
    eye = metrics.get("eye_contact", 0)
    hud = f"Eye: {eye}/10  |  Posture: {metrics.get('head_pose', 0)}/10  |  State: {emotion}"
    cv2.putText(frame, hud, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ──────────────────────────────────────────────────────────
# Global interview state
# ──────────────────────────────────────────────────────────

state = {
    "questions": [], "answers": [], "feedbacks": [], "scores": [],
    "resume_text": "", "name": "", "q_index": 0, "active": False,
    "posture_snapshots": [],   # list of metrics dicts, one per question
    "latest_metrics": {},      # from last camera frame
}


# ──────────────────────────────────────────────────────────
# Interview logic
# ──────────────────────────────────────────────────────────

def start_interview(resume_file):
    if resume_file is None:
        return ("Please upload a resume!",
                gr.update(visible=False), "", gr.update(visible=False))
    try:
        with open(resume_file.name, 'rb') as f:
            resume_text = extract_resume_text(f)
        name = extract_candidate_name(resume_text)
        questions = generate_questions(resume_text)

        state.update({
            "resume_text": resume_text, "name": name,
            "questions": questions, "answers": [], "feedbacks": [],
            "scores": [], "q_index": 0, "active": True,
            "posture_snapshots": [], "latest_metrics": {},
        })

        first_q = questions[0] if questions else "No questions generated"
        status = f"✅ Welcome {name}! {len(questions)} questions ready."
        progress = f"Question 1 of {len(questions)}"
        return (status,
                gr.update(visible=True),
                f"**{progress}**\n\n{first_q}",
                gr.update(visible=False))
    except Exception as e:
        return (f"Error: {str(e)}",
                gr.update(visible=False), "", gr.update(visible=False))


def _process_answer(answer_text, metrics_snapshot):
    """Core: evaluate answer, store posture, advance question."""
    if not state["active"]:
        return "Please start interview first!", "", gr.update(visible=False)

    idx = state["q_index"]
    questions = state["questions"]
    if idx >= len(questions):
        return "Interview complete!", "", gr.update(visible=False)

    current_q = questions[idx]
    feedback = evaluate_answer(current_q, answer_text, state["resume_text"])
    score = extract_score(feedback)

    state["answers"].append(answer_text)
    state["feedbacks"].append(feedback)
    state["scores"].append(score)
    state["posture_snapshots"].append(metrics_snapshot or {})
    state["q_index"] += 1

    next_idx = state["q_index"]

    if next_idx >= len(questions):
        state["active"] = False
        _, average, grade, _ = generate_report(
            state["name"], questions,
            state["answers"], state["feedbacks"], state["scores"])
        score_emoji = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"
        fb_display = (
            f"{score_emoji} **Score: {score}/10**\n\n{feedback}\n\n"
            f"---\n🏁 **Interview Complete!**\n"
            f"Average: **{average:.1f}/10** | Grade: **{grade}**"
        )
        return (fb_display,
                "✅ Interview finished! Go to Download Report tab.",
                gr.update(visible=True))

    next_q = questions[next_idx]
    progress = f"Question {next_idx+1} of {len(questions)}"
    score_emoji = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"
    fb_display = (
        f"{score_emoji} **Score: {score}/10**\n\n{feedback}"
    )
    return (fb_display,
            f"**{progress}**\n\n{next_q}",
            gr.update(visible=False))


def submit_text_answer(answer_text):
    metrics = state.get("latest_metrics", {})
    return _process_answer(answer_text, metrics)


def submit_voice_answer(audio_file):
    if audio_file is None:
        return "No audio recorded!", "", gr.update(visible=False)
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
        fb, prog, dl = _process_answer(text, state.get("latest_metrics", {}))
        return f"🎙️ You said: *{text}*\n\n{fb}", prog, dl
    except Exception as e:
        return f"Could not process audio: {str(e)}", "", gr.update(visible=False)


# ──────────────────────────────────────────────────────────
# Continuous camera feed (Gradio streaming)
# ──────────────────────────────────────────────────────────

def process_camera_frame(frame_rgb):
    """
    Called repeatedly by gr.Image with streaming=True.
    frame_rgb: numpy array H×W×3 (RGB) from webcam.
    Returns: annotated RGB frame + status string.
    """
    if frame_rgb is None:
        return None, "Camera not active"

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    metrics = analyse_frame(frame_bgr)
    state["latest_metrics"] = metrics   # save for answer submission

    annotated = draw_overlay(frame_bgr, metrics)

    if not metrics["face_detected"]:
        status = "⚠️ No face detected — please sit in front of camera"
    else:
        conf = metrics["confidence_score"]
        emotion = metrics["dominant_emotion"]
        eye = metrics["eye_contact"]
        status = (
            f"✅ Face detected | "
            f"Confidence: {conf}/10 | "
            f"Eye contact: {eye}/10 | "
            f"State: {emotion}"
        )
    return annotated, status


# ──────────────────────────────────────────────────────────
# PDF download
# ──────────────────────────────────────────────────────────

def download_pdf():
    if not state["scores"]:
        return None

    # Aggregate posture snapshots into one summary
    snapshots = [s for s in state["posture_snapshots"] if s.get("face_detected")]
    if snapshots:
        avg_conf = round(
            sum(s["confidence_score"] for s in snapshots) / len(snapshots), 1)
        emotions = [s["dominant_emotion"] for s in snapshots]
        dominant = max(set(emotions), key=emotions.count)
        face_summary = {
            "confidence_score": avg_conf,
            "dominant_emotion": dominant,
            "eye_contact": round(
                sum(s.get("eye_contact", 0) for s in snapshots) / len(snapshots), 1),
            "head_pose": round(
                sum(s.get("head_pose", 0) for s in snapshots) / len(snapshots), 1),
        }
    else:
        face_summary = None

    try:
        pdf_bytes = generate_pdf_report(
            state["name"],
            state["questions"],
            state["answers"],
            state["feedbacks"],
            state["scores"],
            voice_data=None,
            face_data=face_summary,
        )
        tmp = tempfile.mktemp(suffix='.pdf')
        with open(tmp, 'wb') as f:
            f.write(pdf_bytes)
        return tmp
    except Exception as e:
        print(f"PDF error: {e}")
        return None


# ──────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────

CSS = """
body { font-family: 'Segoe UI', sans-serif; }
.title-block { text-align:center; padding: 10px 0 4px 0; }
.title-block h1 { font-size: 2.2em; font-weight: 900;
    background: linear-gradient(90deg,#6C63FF,#ff6584);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.title-block p { color: #888; margin-top: 0; }
.question-box textarea { font-size: 1.1em !important;
    font-weight: 600 !important; color: #fff !important;
    background: #1a1a2e !important; border-radius: 10px !important; }
.feedback-box textarea { font-size: 0.97em !important;
    background: #111 !important; color: #eee !important; }
.status-bar { background:#1e1e2e; border-radius:8px;
    padding:8px 14px; color:#a0a0ff; font-size:0.9em; }
.gr-button-primary { background: linear-gradient(90deg,#6C63FF,#8b5cf6) !important;
    border: none !important; font-weight: 700 !important; }
.kk { text-align:right; color:#6C63FF; font-style:italic;
    font-weight:bold; padding: 8px 16px; }
footer { display: none !important; }
"""

THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="pink",
    neutral_hue="slate",
).set(
    body_background_fill="#0d0d1a",
    block_background_fill="#141428",
    block_border_color="#2a2a4a",
    input_background_fill="#1a1a2e",
    body_text_color="#e0e0f0",
    block_title_text_color="#a0a0ff",
)

with gr.Blocks(theme=THEME, css=CSS, title="AI Mock Interview") as demo:

    gr.HTML("""
    <div class="title-block">
      <h1>🤖 AI Mock Interview System</h1>
      <p>Upload your resume → get personalised questions →
         answer & get instant AI feedback → download PDF report</p>
    </div>
    """)

    # ── TAB 1: Interview ──────────────────────────────────
    with gr.Tab("📄 Interview"):
        with gr.Row():
            # Left column: resume + start
            with gr.Column(scale=1, min_width=260):
                resume_input = gr.File(
                    label="📎 Upload Resume (PDF)",
                    file_types=[".pdf"])
                start_btn = gr.Button(
                    "🚀 Start Interview", variant="primary", size="lg")
                status_box = gr.Textbox(
                    label="Status", interactive=False,
                    elem_classes=["status-bar"])

            # Right column: question + answer
            with gr.Column(scale=2):
                question_md = gr.Markdown(
                    "Upload your resume and click **Start Interview**",
                    elem_classes=["question-box"])

                with gr.Tab("✍️ Type Answer"):
                    text_answer = gr.Textbox(
                        label="Your Answer", lines=5,
                        placeholder="Type your answer here…")
                    submit_text_btn = gr.Button(
                        "Submit Answer ➡️", variant="primary")

                with gr.Tab("🎙️ Voice Answer"):
                    voice_input = gr.Audio(
                        label="Record Your Answer",
                        sources=["microphone"], type="filepath")
                    submit_voice_btn = gr.Button(
                        "Submit Voice Answer 🎙️", variant="primary")

                feedback_md = gr.Markdown("", elem_classes=["feedback-box"])

                download_done = gr.Markdown("", visible=False)

    # ── TAB 2: Camera Monitor ─────────────────────────────
    with gr.Tab("🎥 Live Camera Monitor"):
        gr.Markdown(
            "### Continuous Posture & Confidence Monitor\n"
            "Keep this tab open during your interview. "
            "The camera analyses your posture every second and saves "
            "a snapshot when you submit each answer.")
        with gr.Row():
            with gr.Column(scale=3):
                camera_feed = gr.Image(
                    label="Live Feed",
                    sources=["webcam"],
                    streaming=True,          # continuous stream
                    type="numpy",
                    mirror_webcam=True,
                )
            with gr.Column(scale=1):
                cam_status = gr.Textbox(
                    label="Live Analysis",
                    interactive=False,
                    lines=6,
                    elem_classes=["status-bar"])
                gr.Markdown("""
**What is monitored:**
- 👁️ Eye contact score
- 🧍 Head position & posture
- 😐 Confidence state
- 📸 Auto-snapshot per answer
                """)

        camera_feed.stream(
            fn=process_camera_frame,
            inputs=[camera_feed],
            outputs=[camera_feed, cam_status],
        )

    # ── TAB 3: Download ───────────────────────────────────
    with gr.Tab("📥 Download Report"):
        gr.Markdown("### Download your full interview report as PDF")
        gr.Markdown(
            "The PDF includes:\n"
            "- All questions, answers, and AI feedback\n"
            "- Score charts\n"
            "- **Posture & body language analysis** (5–8 lines per metric)\n"
            "- Recommendations")
        dl_btn = gr.Button(
            "📥 Generate PDF Report", variant="primary", size="lg")
        pdf_output = gr.File(label="Your Report")
        dl_btn.click(fn=download_pdf, inputs=[], outputs=[pdf_output])

    gr.HTML('<div class="kk">✦ Created by KK ✦</div>')

    # ── Wiring ────────────────────────────────────────────
    def _start(rf):
        status, _, question, _ = start_interview(rf)
        return status, question

    start_btn.click(
        fn=_start,
        inputs=[resume_input],
        outputs=[status_box, question_md]
    )

    submit_text_btn.click(
        fn=submit_text_answer,
        inputs=[text_answer],
        outputs=[feedback_md, question_md, download_done])

    submit_voice_btn.click(
        fn=submit_voice_answer,
        inputs=[voice_input],
        outputs=[feedback_md, question_md, download_done])


if __name__ == "__main__":
    demo.launch(share=True)
