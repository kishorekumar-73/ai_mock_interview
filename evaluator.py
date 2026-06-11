from openai import OpenAI
import PyPDF2
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def extract_resume_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def call_ai(prompt):
    try:
        response = client.chat.completions.create(
            model="openrouter/auto",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        content = response.choices[0].message.content
        if content is None:
            return "Unable to generate response. Please try again."
        return content.strip()
    except Exception as e:
        # Try backup model
        try:
            response = client.chat.completions.create(
                model="mistralai/mistral-small-3.1-24b-instruct:free",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            content = response.choices[0].message.content
            if content is None:
                return "Unable to generate response."
            return content.strip()
        except:
            return "Unable to generate response. Please try again."
def extract_candidate_name(resume_text):
    prompt = f"""
    Extract only the candidate's full name from this resume.
    Return ONLY the name, nothing else.
    Resume: {resume_text[:500]}
    """
    return call_ai(prompt)

def generate_questions(resume_text):
    prompt = f"""
    You are a senior HR interviewer at a top IT company.
    Based on this resume generate exactly 12 interview questions.
    Rules:
    - Mix HR and technical questions
    - Reference actual projects, skills, technologies from resume
    - Make questions specific not generic
    - Vary difficulty from easy to hard
    Resume: {resume_text}
    Return ONLY a numbered list like:
    1. Question here
    2. Question here
    """
    response = call_ai(prompt)
    lines = response.strip().split("\n")
    questions = []
    for l in lines:
        l = l.strip()
        if l and l[0].isdigit():
            parts = l.split(". ", 1)
            if len(parts) == 2:
                questions.append(parts[1])
    return questions[:10]

def evaluate_answer(question, answer, resume_text):
    prompt = f"""
    You are a senior HR interviewer.
    Candidate Resume Summary: {resume_text[:300]}
    Question Asked: {question}
    Candidate Answer: {answer}
    Evaluate and return in this exact format:
    SCORE: X/10
    GOOD:
    - Point 1
    - Point 2
    IMPROVE:
    - Point 1
    - Point 2
    BETTER ANSWER:
    Write a sample better answer in 3-4 sentences.
    """
    return call_ai(prompt)

def extract_score(feedback_text):
    try:
        for line in feedback_text.split("\n"):
            if "SCORE:" in line:
                score_str = line.replace("SCORE:", "").strip()
                score = float(score_str.split("/")[0].strip())
                return score
    except:
        pass
    return 7.0