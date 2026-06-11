import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
import os

def create_score_chart(questions, scores):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0E1117')

    # Bar chart
    colors = ['#00C853' if s >= 7 else '#FFD600' if s >= 5 else '#FF1744' for s in scores]
    q_labels = [f"Q{i+1}" for i in range(len(questions))]
    bars = axes[0].bar(q_labels, scores, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_facecolor('#1E2130')
    axes[0].set_ylim(0, 10)
    axes[0].set_title('Score Per Question', color='white', fontsize=14, fontweight='bold')
    axes[0].tick_params(colors='white')
    axes[0].spines['bottom'].set_color('white')
    axes[0].spines['left'].set_color('white')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    for bar, score in zip(bars, scores):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{score}', ha='center', va='bottom', color='white', fontweight='bold')

    # Radar/Line chart
    axes[1].plot(q_labels, scores, color='#6C63FF', linewidth=2.5, marker='o', markersize=8)
    axes[1].fill_between(range(len(scores)), scores, alpha=0.3, color='#6C63FF')
    axes[1].set_facecolor('#1E2130')
    axes[1].set_ylim(0, 10)
    axes[1].set_title('Performance Trend', color='white', fontsize=14, fontweight='bold')
    axes[1].tick_params(colors='white')
    axes[1].set_xticks(range(len(q_labels)))
    axes[1].set_xticklabels(q_labels, color='white')
    axes[1].spines['bottom'].set_color('white')
    axes[1].spines['left'].set_color('white')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].axhline(y=7, color='#00C853', linestyle='--', alpha=0.5, label='Good threshold')
    axes[1].legend(facecolor='#1E2130', labelcolor='white')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#0E1117')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_base64

def create_pie_chart(scores):
    excellent = sum(1 for s in scores if s >= 8)
    good = sum(1 for s in scores if 6 <= s < 8)
    average = sum(1 for s in scores if 4 <= s < 6)
    needs_work = sum(1 for s in scores if s < 4)

    labels = []
    sizes = []
    colors_pie = []

    if excellent > 0:
        labels.append(f'Excellent ({excellent})')
        sizes.append(excellent)
        colors_pie.append('#00C853')
    if good > 0:
        labels.append(f'Good ({good})')
        sizes.append(good)
        colors_pie.append('#6C63FF')
    if average > 0:
        labels.append(f'Average ({average})')
        sizes.append(average)
        colors_pie.append('#FFD600')
    if needs_work > 0:
        labels.append(f'Needs Work ({needs_work})')
        sizes.append(needs_work)
        colors_pie.append('#FF1744')

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors_pie,
        autopct='%1.1f%%', startangle=90,
        textprops={'color': 'white'}
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('Answer Quality Distribution',
                color='white', fontsize=14, fontweight='bold')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#0E1117')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_base64

def generate_report(name, questions, answers, feedbacks, scores):
    total = sum(scores)
    average = total / len(scores) if scores else 0

    if average >= 8:
        grade = "🏆 Excellent"
        message = "You are interview ready! Apply with full confidence."
        color = "#00C853"
    elif average >= 6:
        grade = "✅ Good"
        message = "You are on the right track. Practice a bit more."
        color = "#6C63FF"
    elif average >= 4:
        grade = "⚠️ Average"
        message = "Need more preparation. Focus on weak areas."
        color = "#FFD600"
    else:
        grade = "❌ Needs Work"
        message = "Keep practicing daily. You will get there!"
        color = "#FF1744"

    report = f"""# 📊 Interview Performance Report
**Candidate:** {name}
**Total Questions:** {len(questions)}
**Average Score:** {average:.1f} / 10
**Grade:** {grade}
**Message:** {message}
---
"""
    for i, (q, a, f, s) in enumerate(zip(questions, answers, feedbacks, scores)):
        report += f"""
## Question {i+1}
**Q:** {q}
**Your Answer:** {a}
**Score:** {s}/10
**Feedback:**
{f}
---
"""
    return report, average, grade, color