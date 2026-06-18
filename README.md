# AI Mock Interview System

An AI-powered mock interview platform built with Python and Gradio that generates personalized interview questions from a candidate's resume, evaluates responses, analyzes communication skills, and generates detailed performance reports.

## Features

### Resume-Based Question Generation

* Upload a PDF resume
* Extracts skills, projects, and experience
* Generates 10–12 personalized HR and technical interview questions

### Answer Evaluation

* Scores each answer out of 10
* Provides strengths and improvement areas
* Suggests better answer formulations

### Facial Expression Analysis (Optional)

* Uses DeepFace and OpenCV
* Monitors confidence, engagement, and emotional patterns during the interview

### Voice Analysis (Optional)

* Records spoken responses
* Analyzes fluency, speaking pace, and clarity using Librosa
* Converts speech to text for evaluation

### Performance Analytics

* Question-wise score tracking
* Score trend visualization
* Answer quality distribution charts

### Report Generation

* Downloadable PDF report
* Markdown report export
* Complete interview summary with feedback and analytics

## Project Structure

```text
├── app.py              # Gradio interface and interview workflow
├── evaluator.py        # Resume parsing, question generation, scoring
├── face_analyzer.py    # Facial expression analysis
├── voice_analyzer.py   # Voice recording and analysis
├── report.py           # Chart generation
├── pdf_report.py       # PDF report generation
└── requirements.txt
```

## Installation

Clone the repository:

```bash
https://github.com/kishorekumar-73/ai_mock_interview
cd ai-mock-interview
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Additional packages:

```bash
pip install openai fpdf2 matplotlib librosa sounddevice scipy opencv-python deepface gradio
```

## Configuration

Add your OpenRouter API key in `evaluator.py`:

```python
client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://openrouter.ai/api/v1"
)
```

## Run the Application

```bash
python app.py
```

The Gradio interface will launch locally and provide a URL in the terminal.

## Technologies Used

* Python
* Gradio
* OpenRouter API
* DeepFace
* OpenCV
* Librosa
* SpeechRecognition
* Matplotlib
* FPDF2
* PyPDF2

## Future Improvements

* Multi-round interview simulation
* Domain-specific interview modes
* Advanced speech sentiment analysis
* Resume improvement suggestions
* Interview history tracking

## Built by

**Kishore Kumar T**
B.E. Electronics and Communication Engineering (ECE)
