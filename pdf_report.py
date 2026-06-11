from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tempfile
import os

class InterviewReportPDF(FPDF):
    def header(self):
        self.set_fill_color(108, 99, 255)
        self.rect(0, 0, 210, 35, 'F')
        self.set_font('Helvetica', 'B', 22)
        self.set_text_color(255, 255, 255)
        self.cell(0, 20, 'AI Mock Interview Report',
                 align='C', ln=True)
        self.set_font('Helvetica', '', 11)
        self.cell(0, 10, 'Powered by AI Interview System',
                 align='C', ln=True)
        self.set_text_color(0, 0, 0)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    replacements = {
        '✅': '[OK]', '⚠️': '[WARN]', '❌': '[BAD]',
        '💡': '[TIP]', '🟢': '[GOOD]', '🟡': '[AVG]',
        '🔴': '[POOR]', '📋': '', '🎙️': '', '🎥': '',
        '🏆': '', '👋': '', '❓': '', '📄': '',
        '📊': '', '🤖': '', '*': '', '#': '',
        '→': '->', '•': '-'
    }
    for emoji, replacement in replacements.items():
        text = text.replace(emoji, replacement)
    cleaned = ''
    for char in text:
        try:
            char.encode('latin-1')
            cleaned += char
        except:
            cleaned += ' '
    return cleaned.strip()

def generate_posture_analysis(face_data, face_results=None):
    """Generate detailed 8-10 line posture analysis"""
    if not face_data:
        return "No posture data available for this session."

    conf = face_data.get('confidence_score', 0)
    emotion = face_data.get('dominant_emotion', 'neutral')
    lines = []

    # Overall assessment
    if conf >= 8:
        lines.append(
            "OVERALL: Excellent body language maintained "
            "throughout the interview.")
    elif conf >= 6:
        lines.append(
            "OVERALL: Good body language with minor areas "
            "for improvement.")
    elif conf >= 4:
        lines.append(
            "OVERALL: Average body language detected. "
            "Significant improvement needed.")
    else:
        lines.append(
            "OVERALL: Poor body language observed. "
            "Practice is strongly recommended.")

    # Confidence detail
    lines.append(
        f"CONFIDENCE SCORE: {conf}/10 - "
        + ("Strong confident presence detected."
           if conf >= 7 else
           "Moderate confidence shown, try to sit upright."
           if conf >= 5 else
           "Low confidence visible, practice power posing."))

    # Emotion analysis
    emotion_analysis = {
        'happy': "EXPRESSION: Positive and engaging expressions "
                 "were observed, which creates a great impression.",
        'neutral': "EXPRESSION: Neutral professional expression "
                   "maintained, appropriate for interviews.",
        'sad': "EXPRESSION: Sad or low energy expressions "
               "detected. Try to appear more enthusiastic.",
        'fear': "EXPRESSION: Nervousness visible in expressions. "
                "Deep breathing before answering helps.",
        'angry': "EXPRESSION: Tense expressions observed. "
                 "Relax your facial muscles consciously.",
        'surprise': "EXPRESSION: Surprised expressions noted. "
                    "Try to maintain composure when hearing questions.",
        'disgust': "EXPRESSION: Uncomfortable expressions detected. "
                   "Practice neutral expressions in mirror."
    }
    lines.append(emotion_analysis.get(emotion,
        "EXPRESSION: Expression was generally appropriate."))

    # Eye contact
    if conf >= 7:
        lines.append(
            "EYE CONTACT: Good eye contact with camera "
            "maintained, showing confidence and engagement.")
    elif conf >= 5:
        lines.append(
            "EYE CONTACT: Inconsistent eye contact observed. "
            "Look directly at camera when speaking.")
    else:
        lines.append(
            "EYE CONTACT: Poor eye contact detected. "
            "Practice speaking while maintaining camera focus.")

    # Posture
    if conf >= 7:
        lines.append(
            "POSTURE: Upright and attentive posture observed "
            "indicating high engagement and professionalism.")
    else:
        lines.append(
            "POSTURE: Slouching or unstable posture detected. "
            "Sit straight with shoulders back for better impression.")

    # Facial movement
    lines.append(
        "FACIAL ACTIVITY: Camera continuously monitored "
        f"expressions. Dominant mood was {emotion}. "
        "Consistent professional expression is recommended.")

    # Stress indicators
    if emotion in ['fear', 'sad', 'angry']:
        lines.append(
            "STRESS LEVEL: Signs of stress or discomfort "
            "detected during interview. Practice mock interviews "
            "regularly to build comfort.")
    else:
        lines.append(
            "STRESS LEVEL: Relatively calm demeanor observed "
            "throughout the session. Keep maintaining this composure.")

    # Recommendations
    if conf >= 7:
        lines.append(
            "RECOMMENDATION: Strong non-verbal communication. "
            "Continue practicing to maintain this standard in "
            "actual interviews.")
    elif conf >= 5:
        lines.append(
            "RECOMMENDATION: Practice in front of mirror daily "
            "for 10 minutes. Record yourself and review "
            "expressions and posture.")
    else:
        lines.append(
            "RECOMMENDATION: Daily practice strongly advised. "
            "Watch interview tips videos, practice power poses "
            "and work on confidence building exercises.")

    # Final tip
    lines.append(
        "GENERAL TIP: Smile naturally, maintain eye contact, "
        "sit straight, and speak with confidence. "
        "First impressions matter significantly in interviews.")

    return "\n".join(lines)

def create_bar_chart_image(questions, scores):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['#00C853' if s >= 7 else '#FFD600' if s >= 5
              else '#FF1744' for s in scores]
    q_labels = [f"Q{i+1}" for i in range(len(questions))]
    bars = ax.bar(q_labels, scores, color=colors,
                  edgecolor='white', linewidth=0.5)
    ax.set_ylim(0, 10)
    ax.set_title('Score Per Question',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2.,
               bar.get_height() + 0.1,
               f'{score}', ha='center',
               va='bottom', fontweight='bold')
    plt.tight_layout()
    temp = tempfile.mktemp(suffix='.png')
    plt.savefig(temp, dpi=150, bbox_inches='tight')
    plt.close()
    return temp

def create_pie_chart_image(scores):
    excellent = sum(1 for s in scores if s >= 8)
    good = sum(1 for s in scores if 6 <= s < 8)
    average = sum(1 for s in scores if 4 <= s < 6)
    needs_work = sum(1 for s in scores if s < 4)

    labels, sizes, colors = [], [], []
    if excellent > 0:
        labels.append(f'Excellent ({excellent})')
        sizes.append(excellent)
        colors.append('#00C853')
    if good > 0:
        labels.append(f'Good ({good})')
        sizes.append(good)
        colors.append('#6C63FF')
    if average > 0:
        labels.append(f'Average ({average})')
        sizes.append(average)
        colors.append('#FFD600')
    if needs_work > 0:
        labels.append(f'Needs Work ({needs_work})')
        sizes.append(needs_work)
        colors.append('#FF1744')
    if not sizes:
        sizes = [1]
        labels = ['No Data']
        colors = ['#888888']

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(sizes, labels=labels, colors=colors,
           autopct='%1.1f%%', startangle=90)
    ax.set_title('Answer Quality Distribution',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    temp = tempfile.mktemp(suffix='.png')
    plt.savefig(temp, dpi=150, bbox_inches='tight')
    plt.close()
    return temp

def generate_pdf_report(name, questions, answers,
                        feedbacks, scores,
                        voice_data=None, face_data=None):
    average = sum(scores) / len(scores) if scores else 0
    if average >= 8:
        grade = "Excellent"
    elif average >= 6:
        grade = "Good"
    elif average >= 4:
        grade = "Average"
    else:
        grade = "Needs Work"

    pdf = InterviewReportPDF()
    pdf.set_margins(15, 45, 15)
    pdf.add_page()

    # Candidate info
    pdf.set_fill_color(245, 245, 255)
    pdf.rect(10, 40, 190, 40, 'F')
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_xy(15, 43)
    pdf.cell(0, 8, clean_text(f'Candidate: {name}'), ln=True)
    pdf.set_font('Helvetica', '', 12)
    pdf.set_x(15)
    pdf.cell(60, 8, f'Total Questions: {len(questions)}')
    pdf.cell(60, 8, f'Average Score: {average:.1f}/10')
    pdf.cell(60, 8, f'Grade: {grade}', ln=True)
    pdf.ln(15)

    # Summary boxes
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 8, 'Performance Summary', ln=True)
    pdf.ln(3)

    box_colors = [
        (0, 200, 83), (108, 99, 255),
        (255, 214, 0), (255, 23, 68)
    ]
    labels_count = [
        ('Excellent (8-10)', sum(1 for s in scores if s >= 8)),
        ('Good (6-8)', sum(1 for s in scores if 6 <= s < 8)),
        ('Average (4-6)', sum(1 for s in scores if 4 <= s < 6)),
        ('Needs Work (<4)', sum(1 for s in scores if s < 4))
    ]

    x_start = 15
    y_before = pdf.get_y()
    for i, ((label, count), color) in enumerate(
            zip(labels_count, box_colors)):
        pdf.set_fill_color(*color)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(x_start + i*47, y_before)
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(44, 12, str(count),
                align='C', fill=True, ln=False)

    pdf.ln(14)
    pdf.set_text_color(0, 0, 0)
    y_label = pdf.get_y()
    for i, (label, count) in enumerate(labels_count):
        pdf.set_xy(x_start + i*47, y_label)
        pdf.set_font('Helvetica', '', 8)
        pdf.cell(44, 6, label, align='C', ln=False)
    pdf.ln(12)

    # Charts
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_x(15)
    pdf.cell(0, 8, 'Performance Charts', ln=True)
    pdf.ln(3)

    bar_chart = create_bar_chart_image(questions, scores)
    pie_chart = create_pie_chart_image(scores)
    chart_y = pdf.get_y()
    pdf.image(bar_chart, x=10, y=chart_y, w=125)
    pdf.image(pie_chart, x=138, y=chart_y, w=62)
    pdf.ln(70)

    # Voice analysis
    if voice_data and voice_data.get(
            'overall_voice_score', 0) > 0:
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_x(15)
        pdf.cell(0, 8, 'Voice Analysis', ln=True)
        pdf.ln(3)
        pdf.set_fill_color(245, 245, 255)
        pdf.rect(10, pdf.get_y(), 190, 50, 'F')
        pdf.set_font('Helvetica', '', 11)
        pdf.set_xy(15, pdf.get_y() + 3)
        pdf.cell(90, 8,
                f"Confidence: "
                f"{voice_data['confidence_score']}/10")
        pdf.cell(90, 8,
                f"Clarity: "
                f"{voice_data['clarity_score']}/10",
                ln=True)
        pdf.set_x(15)
        pdf.cell(90, 8,
                f"Pace: {voice_data['pace_score']}/10")
        pdf.cell(90, 8,
                f"Overall: "
                f"{voice_data['overall_voice_score']}/10",
                ln=True)
        pdf.set_x(15)
        fb = clean_text(voice_data.get('feedback', ''))
        pdf.multi_cell(180, 8, f"Feedback: {fb}")
        pdf.ln(5)

    # Face/posture analysis — DETAILED
    if face_data:
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_x(15)

        # Section header with purple background
        pdf.set_fill_color(108, 99, 255)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(180, 10,
                'Body Language & Posture Analysis',
                fill=True, ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # Score boxes
        conf = face_data.get('confidence_score', 0)
        emotion = face_data.get('dominant_emotion', 'N/A')

        pdf.set_fill_color(108, 99, 255)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 20)
        pdf.set_x(15)
        pdf.cell(88, 15, f"{conf}/10",
                align='C', fill=True, ln=False)
        pdf.set_fill_color(0, 200, 83)
        pdf.cell(88, 15,
                clean_text(emotion.upper()),
                align='C', fill=True, ln=True)

        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_x(15)
        pdf.cell(88, 6, 'Confidence Score',
                align='C', ln=False)
        pdf.cell(88, 6, 'Dominant Emotion',
                align='C', ln=True)
        pdf.ln(8)

        # Detailed analysis lines
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_x(15)
        pdf.cell(0, 8, 'Detailed Posture Report:', ln=True)
        pdf.ln(2)

        analysis = generate_posture_analysis(face_data)
        lines = analysis.split('\n')

        for line in lines:
            if pdf.get_y() > 260:
                pdf.add_page()

            # Style each line based on prefix
            if line.startswith('OVERALL:'):
                pdf.set_fill_color(230, 230, 255)
                pdf.set_font('Helvetica', 'B', 10)
            elif line.startswith('RECOMMENDATION:'):
                pdf.set_fill_color(255, 245, 200)
                pdf.set_font('Helvetica', 'B', 10)
            elif line.startswith('GENERAL TIP:'):
                pdf.set_fill_color(200, 255, 220)
                pdf.set_font('Helvetica', 'B', 10)
            else:
                pdf.set_fill_color(248, 248, 248)
                pdf.set_font('Helvetica', '', 10)

            pdf.set_x(15)
            pdf.multi_cell(180, 7,
                          clean_text(line),
                          fill=True)
            pdf.ln(1)

        pdf.ln(5)

    # Detailed Q&A
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_x(15)
    pdf.cell(0, 10, 'Detailed Question Analysis', ln=True)
    pdf.ln(3)

    for i, (q, a, f, s) in enumerate(
            zip(questions, answers, feedbacks, scores)):
        if pdf.get_y() > 240:
            pdf.add_page()

        if s >= 7:
            pdf.set_fill_color(0, 200, 83)
        elif s >= 5:
            pdf.set_fill_color(255, 214, 0)
        else:
            pdf.set_fill_color(255, 23, 68)

        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_x(15)
        pdf.cell(180, 8,
                clean_text(f'Q{i+1}: Score {s}/10'),
                fill=True, ln=True)
        pdf.set_text_color(0, 0, 0)

        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_fill_color(245, 245, 255)
        pdf.set_x(15)
        pdf.multi_cell(180, 7,
                      clean_text(f'Question: {q}'),
                      fill=True)

        pdf.set_font('Helvetica', '', 10)
        pdf.set_x(15)
        pdf.multi_cell(180, 7,
                      clean_text(
                          f'Your Answer: {str(a)[:1500]}'))

        pdf.set_font('Helvetica', 'I', 10)
        clean_fb = clean_text(str(f))[:1500]
        pdf.set_x(15)
        pdf.multi_cell(180, 7, f'Feedback: {clean_fb}')
        pdf.ln(4)

    # Save
    temp_pdf = tempfile.mktemp(suffix='.pdf')
    pdf.output(temp_pdf)
    try:
        os.remove(bar_chart)
        os.remove(pie_chart)
    except:
        pass
    with open(temp_pdf, 'rb') as f:
        pdf_bytes = f.read()
    try:
        os.remove(temp_pdf)
    except:
        pass
    return pdf_bytes