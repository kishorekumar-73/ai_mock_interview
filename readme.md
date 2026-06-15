# 🤖 AI Mock Interview

An AI-powered mock interview platform built with Streamlit. Upload your resume and get a personalized, fully evaluated mock interview — complete with live face monitoring, voice analysis, performance charts, and a downloadable PDF report.

---

## ✨ Features

- **Resume-Driven Questions** — Parses your PDF resume and generates 10–12 personalized interview questions mixing HR and technical topics
- **AI Answer Evaluation** — Each answer is scored out of 10 with structured feedback (strengths, areas to improve, and a sample better answer)
- **Live Face Monitoring** — Optional webcam analysis using DeepFace to track emotions and confidence throughout the interview
- **Voice Analysis** — Optional microphone recording per question, analyzed for confidence, clarity, pace, and fluency using Librosa
- **Performance Charts** — Bar chart, trend line, and pie chart showing score distribution per question
- **PDF Report** — Downloadable, professionally formatted report including scores, feedback, voice summary, and posture analysis
- **Markdown Report** — Lightweight `.md` version of the full report also available for download

---

## 🗂️ Project Structure

```
├── app.py              # Main Streamlit app — UI, session state, interview flow
├── evaluator.py        # Resume parsing, question generation, answer evaluation via AI
├── face_analyzer.py    # Webcam emotion detection using DeepFace (runs in background thread)
├── voice_analyzer.py   # Audio recording and voice quality analysis using Librosa
├── report.py           # In-app chart generation (Matplotlib)
├── pdf_report.py       # PDF report generation using FPDF
└── requirements.txt    # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai-mock-interview.git
cd ai-mock-interview
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

You will also need to install a few additional packages not listed in `requirements.txt`:

```bash
pip install openai fpdf2 matplotlib librosa sounddevice scipy opencv-python deepface
```

> **Note:** `pyaudio` can be tricky to install. On macOS use `brew install portaudio` first. On Windows, download a prebuilt wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

### 3. Configure your API key

In `evaluator.py`, replace the placeholder API key with your own [OpenRouter](https://openrouter.ai) key:

```python
client = OpenAI(
    api_key="your-openrouter-api-key",
    base_url="https://openrouter.ai/api/v1"
)
```

### 4. Run the app

```bash
streamlit run app.py
```

---

## 🖥️ Usage

1. Open the app in your browser (default: `http://localhost:8501`)
2. Toggle **Camera Monitoring** and/or **Voice Analysis** as desired
3. Upload your resume as a **PDF**
4. Click **Start My Interview**
5. Answer each question by typing or speaking
6. View your full report at the end and download it as PDF or Markdown

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `openai` | OpenRouter API client |
| `PyPDF2` | Resume PDF text extraction |
| `deepface` | Facial emotion recognition |
| `opencv-python` | Webcam frame capture |
| `librosa` | Audio feature extraction for voice analysis |
| `sounddevice` / `scipy` | Audio recording |
| `matplotlib` | Performance charts |
| `fpdf2` | PDF report generation |
| `SpeechRecognition` | Speech-to-text for voice answers |

---

## ⚙️ Optional Features

Both face and voice analysis are **opt-in** via toggles on the home screen. The interview works fully without a camera or microphone — you can type all your answers.

---

## 📄 Report Output

At the end of each interview session you receive:

- A **score per question** with colour-coded bar and trend charts
- A **pie chart** showing answer quality distribution (Excellent / Good / Average / Needs Work)
- **Detailed AI feedback** for every answer
- Optional **voice metrics**: confidence, clarity, pace, fluency
- Optional **posture report**: emotion timeline, eye contact, stress level, recommendations
- A downloadable **PDF** and **Markdown** report

---

## 🔑 API

This project uses [OpenRouter](https://openrouter.ai) to route requests to models like `openrouter/auto` and `mistralai/mistral-small-3.1-24b-instruct:free`. You can swap in any OpenRouter-compatible model in `evaluator.py`.

---

## 🙏 Credits

Created by **KISHORE KUMAR**
