import streamlit as st
import speech_recognition as sr
import cv2
import tempfile
import os
import time
import threading
from evaluator import (extract_resume_text, extract_candidate_name,
                       generate_questions, evaluate_answer,
                       extract_score)
from report import generate_report, create_score_chart, create_pie_chart
from pdf_report import generate_pdf_report
from voice_analyzer import analyze_voice, record_audio
from face_analyzer import ContinuousPostureMonitor

st.set_page_config(
    page_title="AI Mock Interview",
    page_icon="🤖",
    layout="centered"
)

# ── FIX 1: CSS was raw text outside st.markdown — moved inside the call ──
st.markdown("""
<style>
.kk-brand {
    position: fixed;
    bottom: 15px;
    right: 20px;
    background: linear-gradient(135deg, #6C63FF, #FF6B35);
    color: white !important;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.85em;
    font-weight: 700;
    letter-spacing: 2px;
    box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4);
    z-index: 999;
    font-style: italic;
}
.big-title {
    font-size: 3em;
    font-weight: 800;
    text-align: center;
    color: #6C63FF;
}
.subtitle {
    font-size: 1.2em;
    text-align: center;
    color: #aaa;
    margin-bottom: 30px;
}
.question-box {
    background: #1E2130;
    padding: 20px;
    border-radius: 12px;
    border-left: 4px solid #6C63FF;
    font-size: 1.1em;
    margin-bottom: 20px;
    color: white !important;

}
.feedback-box {
    background: #1E2130;
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
}
.analysis-box {
    background: #1E2130;
    padding: 15px;
    border-radius: 12px;
    border-left: 4px solid #00C853;
    margin: 10px 0;
}
.live-box {
    background: #1E2130;
    padding: 10px;
    border-radius: 8px;
    border-left: 4px solid #FF6B35;
    margin: 5px 0;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "stage": "home",
        "resume_text": "",
        "candidate_name": "",
        "questions": [],
        "answers": [],
        "feedbacks": [],
        "scores": [],
        "q_index": 0,
        "voice_results": [],
        "face_results": [],
        "enable_video": False,
        "enable_voice": False,
        "monitor": None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── FIX 2: Typo "iinit_state()" → "init_state()" ──
init_state()

# KK Brand watermark — shows on every page
st.markdown(
    '<div class="kk-brand">✦ Created by KK ✦</div>',
    unsafe_allow_html=True)

# ─────────────────────────────
# HOME
# ─────────────────────────────
if st.session_state.stage == "home":
    st.markdown(
        '<div class="big-title">🤖 AI Mock Interview</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Upload Resume → 12 Questions'
        ' → Live Monitoring → PDF Report</div>',
        unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 📄\n**Resume**\nAI reads it")
    with col2:
        st.markdown("### ❓\n**12 Questions**\nPersonalized")
    with col3:
        st.markdown("### 🎥\n**Live Monitor**\nFace + Voice")
    with col4:
        st.markdown("### 📊\n**PDF Report**\nWith charts")

    st.markdown("---")
    st.markdown("### ⚙️ Interview Settings")
    col1, col2 = st.columns(2)
    with col1:
        enable_video = st.toggle(
            "🎥 Enable Camera Monitoring", value=False)
    with col2:
        enable_voice = st.toggle(
            "🗣️ Enable Voice Analysis", value=False)

    st.session_state.enable_video = enable_video
    st.session_state.enable_voice = enable_voice

    if enable_video:
        st.info(
            "Camera runs throughout ALL questions "
            "automatically in background!")
    if enable_voice:
        st.info(
            "Speak your answer — voice analyzed "
            "after you submit!")

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📄 Upload Your Resume (PDF only)", type=["pdf"])

    if uploaded_file:
        if st.button("🚀 Start My Interview",
                     use_container_width=True):
            with st.spinner("🧠 Analyzing your resume..."):
                resume_text = extract_resume_text(uploaded_file)
                name = extract_candidate_name(resume_text)
                questions = generate_questions(resume_text)

            st.session_state.resume_text = resume_text
            st.session_state.candidate_name = name
            st.session_state.questions = questions

            if enable_video:
                try:
                    monitor = ContinuousPostureMonitor()
                    monitor.start()
                    st.session_state.monitor = monitor
                    time.sleep(2)
                except Exception as e:
                    st.warning(f"Camera error: {e}")
                    st.session_state.monitor = None

            st.session_state.stage = "interview"
            st.rerun()

# ─────────────────────────────
# INTERVIEW
# ─────────────────────────────
elif st.session_state.stage == "interview":
    name = st.session_state.candidate_name
    questions = st.session_state.questions
    index = st.session_state.q_index
    total = len(questions)
    monitor = st.session_state.monitor

    st.markdown(f"### 👋 Hello, {name}!")

    # Show live camera feed
    if monitor and monitor.is_running:
        st.markdown(
            '<div class="live-box">🔴 LIVE — '
            'Camera monitoring your face</div>',
            unsafe_allow_html=True)
        frame = monitor.get_latest_frame()
        if frame is not None:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="Live Camera", width=280)

    # ── FIX 3: Guard against division by zero when total == 0 ──
    progress = (index / total) if total > 0 else 0.0
    st.progress(progress,
                text=f"Q{index + 1} of {total} — "
                     f"{int(progress * 100)}% Complete")

    if st.session_state.scores:
        avg = (sum(st.session_state.scores) /
               len(st.session_state.scores))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average", f"{avg:.1f}/10")
        with col2:
            st.metric("Done", len(st.session_state.scores))
        with col3:
            st.metric("Left", total - index)

    st.markdown("---")

    if index < total:
        current_q = questions[index]
        st.markdown(
            f'<div class="question-box">❓ '
            f'<b>Q{index + 1}:</b> {current_q}</div>',
            unsafe_allow_html=True)

        mode = st.radio(
            "Answer via:",
            ["✍️ Type", "🎙️ Speak"],
            horizontal=True)

        if mode == "✍️ Type":
            user_answer = st.text_area(
                "Your Answer", height=150,
                placeholder="Type detailed answer here...")

            col1, col2 = st.columns(2)
            with col1:
                submit = st.button(
                    "Submit ➡️", use_container_width=True)
            with col2:
                skip = st.button(
                    "Skip ⏭️", use_container_width=True)

            if submit and user_answer.strip():
                with st.spinner("🧠 Evaluating..."):
                    feedback = evaluate_answer(
                        current_q, user_answer,
                        st.session_state.resume_text)
                    score = extract_score(feedback)

                # Voice analysis
                voice_result = None
                if st.session_state.enable_voice:
                    with st.spinner("🎙️ Analyzing voice..."):
                        try:
                            audio_path = record_audio(duration=5)
                            voice_result = analyze_voice(audio_path)
                            if os.path.exists(audio_path):
                                os.remove(audio_path)
                        except Exception:
                            voice_result = None

                # Posture snapshot
                face_result = None
                if monitor and monitor.is_running:
                    face_result = monitor.get_snapshot(index + 1)
                    monitor.reset_history()

                st.session_state.answers.append(user_answer)
                st.session_state.feedbacks.append(feedback)
                st.session_state.scores.append(score)
                st.session_state.voice_results.append(voice_result)
                st.session_state.face_results.append(face_result)

                dot = ("🟢" if score >= 7 else
                       "🟡" if score >= 5 else "🔴")
                st.markdown(f"### {dot} Score: {score}/10")
                st.markdown(
                    f'<div class="feedback-box">'
                    f'{feedback}</div>',
                    unsafe_allow_html=True)

                if (voice_result and
                        voice_result.get('overall_voice_score', 0) > 0):
                    st.markdown(
                        f'<div class="analysis-box">'
                        f'🎙️ <b>Voice:</b> '
                        f'{voice_result["overall_voice_score"]}/10 | '
                        f'Confidence: {voice_result["confidence_score"]} | '
                        f'Clarity: {voice_result["clarity_score"]}'
                        f'</div>',
                        unsafe_allow_html=True)

                if (face_result and
                        face_result.get('confidence_score', 0) > 0):
                    st.markdown(
                        f'<div class="analysis-box">'
                        f'🎥 <b>Posture:</b> '
                        f'{face_result["confidence_score"]}/10 | '
                        f'Emotion: {face_result["dominant_emotion"]}'
                        f'<br>{face_result["feedback"]}'
                        f'</div>',
                        unsafe_allow_html=True)
                elif face_result:
                    st.warning(
                        "Face not detected. "
                        "Please sit in front of camera.")

                st.session_state.q_index += 1
                if st.session_state.q_index >= total:
                    if monitor:
                        monitor.stop()
                    st.session_state.stage = "report"
                st.rerun()

            elif skip:
                st.session_state.answers.append("Skipped")
                st.session_state.feedbacks.append("Skipped.")
                st.session_state.scores.append(0)
                st.session_state.voice_results.append(None)
                st.session_state.face_results.append(None)
                if monitor:
                    monitor.reset_history()
                st.session_state.q_index += 1
                if st.session_state.q_index >= total:
                    if monitor:
                        monitor.stop()
                    st.session_state.stage = "report"
                st.rerun()

        else:  # Speak mode
            st.info(
                "Click Record → speak your full answer "
                "→ stop speaking when done")
            if st.button("🎙️ Record Answer",
                         use_container_width=True):
                r = sr.Recognizer()
                r.energy_threshold = 300
                r.dynamic_energy_threshold = True
                try:
                    with sr.Microphone() as source:
                        r.adjust_for_ambient_noise(
                            source, duration=1)
                        st.info(
                            "🎙️ Listening... "
                            "Speak your full answer!")
                        audio = r.listen(
                            source,
                            timeout=30,
                            phrase_time_limit=120)
                    user_answer = r.recognize_google(audio)
                    st.success(f"You said: {user_answer}")

                    with st.spinner("🧠 Evaluating..."):
                        feedback = evaluate_answer(
                            current_q, user_answer,
                            st.session_state.resume_text)
                        score = extract_score(feedback)

                    # Posture snapshot
                    face_result = None
                    if monitor and monitor.is_running:
                        face_result = monitor.get_snapshot(
                            index + 1)
                        monitor.reset_history()

                    st.session_state.answers.append(user_answer)
                    st.session_state.feedbacks.append(feedback)
                    st.session_state.scores.append(score)
                    # ── FIX 4: Speak mode never appended voice_results
                    #    so list lengths became mismatched with answers ──
                    st.session_state.voice_results.append(None)
                    st.session_state.face_results.append(face_result)

                    dot = ("🟢" if score >= 7 else
                           "🟡" if score >= 5 else "🔴")
                    st.markdown(f"### {dot} Score: {score}/10")
                    st.markdown(
                        f'<div class="feedback-box">'
                        f'{feedback}</div>',
                        unsafe_allow_html=True)

                    if (face_result and
                            face_result.get('confidence_score', 0) > 0):
                        st.markdown(
                            f'<div class="analysis-box">'
                            f'🎥 <b>Posture:</b> '
                            f'{face_result["confidence_score"]}/10 | '
                            f'{face_result["dominant_emotion"]}'
                            f'<br>{face_result["feedback"]}'
                            f'</div>',
                            unsafe_allow_html=True)

                    st.session_state.q_index += 1
                    if st.session_state.q_index >= total:
                        if monitor:
                            monitor.stop()
                        st.session_state.stage = "report"
                    st.rerun()

                except sr.WaitTimeoutError:
                    st.error("No speech detected. Try again.")
                except sr.UnknownValueError:
                    st.error("Could not understand. Try again.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── FIX 5: Missing else-branch — if index >= total we should
    #    redirect to report instead of silently showing nothing ──
    else:
        if monitor:
            monitor.stop()
        st.session_state.stage = "report"
        st.rerun()

# ─────────────────────────────
# REPORT
# ─────────────────────────────
elif st.session_state.stage == "report":
    monitor = st.session_state.monitor
    if monitor and monitor.is_running:
        monitor.stop()

    name = st.session_state.candidate_name
    questions = st.session_state.questions
    answers = st.session_state.answers
    feedbacks = st.session_state.feedbacks
    scores = st.session_state.scores
    voice_results = st.session_state.voice_results
    face_results = st.session_state.face_results

    report_text, average, grade, color = generate_report(
        name, questions, answers, feedbacks, scores)

    st.balloons()
    st.markdown(f"## 🏆 Interview Complete, {name}!")
    st.markdown("---")

    # ── FIX 6: max(scores) crashes on empty list — guard it ──
    best_score = max(scores) if scores else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Score", f"{average:.1f}/10")
    with col2:
        st.metric("Questions", len(questions))
    with col3:
        st.metric("Grade", grade.split()[1] if len(grade.split()) > 1 else grade)
    with col4:
        st.metric("Best Score", f"{best_score}/10")

    st.markdown("---")

    st.markdown("### 📊 Performance Charts")
    chart_b64 = create_score_chart(questions, scores)
    st.markdown(
        f'<img src="data:image/png;base64,{chart_b64}" '
        f'width="100%">',
        unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pie_b64 = create_pie_chart(scores)
        st.markdown(
            f'<img src="data:image/png;base64,{pie_b64}" '
            f'width="100%">',
            unsafe_allow_html=True)
    with col2:
        st.markdown("### 📈 Scores")
        for i, (q, s) in enumerate(zip(questions, scores)):
            dot = "🟢" if s >= 7 else "🟡" if s >= 5 else "🔴"
            st.markdown(
                f"{dot} **Q{i + 1}:** {q[:40]}... **{s}/10**")

    # Voice summary
    valid_voice = [
        v for v in voice_results
        if v is not None and v.get('overall_voice_score', 0) > 0]
    if valid_voice:
        st.markdown("---")
        st.markdown("### 🎙️ Voice Analysis")
        avg_voice = (sum(
            v['overall_voice_score'] for v in valid_voice
        ) / len(valid_voice))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Voice Score", f"{avg_voice:.1f}/10")
        with col2:
            avg_conf = (sum(
                v['confidence_score'] for v in valid_voice
            ) / len(valid_voice))
            st.metric("Confidence", f"{avg_conf:.1f}/10")
        with col3:
            avg_clarity = (sum(
                v['clarity_score'] for v in valid_voice
            ) / len(valid_voice))
            st.metric("Clarity", f"{avg_clarity:.1f}/10")

    # Face summary
    valid_face = [f for f in face_results if f is not None]
    if valid_face:
        st.markdown("---")
        st.markdown("### 🎥 Posture Summary")
        avg_face = (sum(
            f['confidence_score'] for f in valid_face
        ) / len(valid_face))
        emotions_list = [f['dominant_emotion'] for f in valid_face]
        most_common = max(set(emotions_list), key=emotions_list.count)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Posture Score", f"{avg_face:.1f}/10")
        with col2:
            st.metric("Dominant Emotion", most_common.title())
        with col3:
            positive = sum(
                1 for e in emotions_list
                if e in ['happy', 'neutral'])
            st.metric("Positive",
                      f"{positive}/{len(emotions_list)}")

    st.markdown("---")

    # Detailed feedback
    st.markdown("### 📝 Detailed Feedback")
    for i, (q, a, f, s) in enumerate(
            zip(questions, answers, feedbacks, scores)):
        dot = "🟢" if s >= 7 else "🟡" if s >= 5 else "🔴"
        with st.expander(
                f"{dot} Q{i + 1}: {q[:50]}... Score:{s}/10"):
            st.markdown(f"**Answer:** {a}")
            st.markdown("**Feedback:**")
            st.markdown(
                f'<div class="feedback-box">{f}</div>',
                unsafe_allow_html=True)
            if (i < len(voice_results) and
                    voice_results[i] is not None and
                    voice_results[i].get(
                        'overall_voice_score', 0) > 0):
                v = voice_results[i]
                st.markdown(
                    f'<div class="analysis-box">'
                    f'Voice: {v["overall_voice_score"]}/10 | '
                    f'Confidence: {v["confidence_score"]} | '
                    f'Clarity: {v["clarity_score"]}'
                    f'</div>',
                    unsafe_allow_html=True)
            if (i < len(face_results) and
                    face_results[i] is not None and
                    face_results[i].get(
                        'confidence_score', 0) > 0):
                fr = face_results[i]
                st.markdown(
                    f'<div class="analysis-box">'
                    f'Posture: {fr["confidence_score"]}/10 | '
                    f'Emotion: {fr["dominant_emotion"]}'
                    f'<br>{fr["feedback"]}'
                    f'</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # PDF generation
    with st.spinner("📄 Generating PDF..."):
        voice_summary = None
        if valid_voice:
            voice_summary = {
                'confidence_score': round(sum(
                    v['confidence_score']
                    for v in valid_voice) / len(valid_voice), 1),
                'clarity_score': round(sum(
                    v['clarity_score']
                    for v in valid_voice) / len(valid_voice), 1),
                'pace_score': round(sum(
                    v['pace_score']
                    for v in valid_voice) / len(valid_voice), 1),
                'overall_voice_score': round(sum(
                    v['overall_voice_score']
                    for v in valid_voice) / len(valid_voice), 1),
                'feedback': valid_voice[-1].get('feedback', '')
            }

        face_summary = None
        if valid_face:
            avg_face_score = round(sum(
                f['confidence_score']
                for f in valid_face) / len(valid_face), 1)
            emotions_list = [
                f['dominant_emotion'] for f in valid_face]
            most_common_emotion = max(
                set(emotions_list), key=emotions_list.count)
            fb_lines = valid_face[-1].get('feedback', '').split(' | ')
            unique_fb = list(dict.fromkeys(fb_lines))
            face_summary = {
                'confidence_score': avg_face_score,
                'dominant_emotion': most_common_emotion,
                'feedback': ' | '.join(unique_fb[:3])
            }

        pdf_bytes = generate_pdf_report(
            name, questions, answers, feedbacks, scores,
            voice_summary, face_summary)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download PDF",
            data=pdf_bytes,
            file_name=f"{name}_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    with col2:
        st.download_button(
            label="📄 Download MD",
            data=report_text,
            file_name=f"{name}_report.md",
            mime="text/markdown",
            use_container_width=True
        )

    if st.button("🔄 New Interview", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
