from dotenv import load_dotenv
load_dotenv()

import os
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

#  Load API key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env file")

#  Setup FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Define input format
class TranscriptRequest(BaseModel):
    transcript: str

#  Get context from past meetings
def get_rag_context(new_transcript: str, folder_path="past_meetings"):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Missing folder: {folder_path}")
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, filename))
            data = loader.load()
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs.extend(splitter.split_documents(data))
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    results = db.similarity_search(new_transcript, k=2)
    return [doc.page_content for doc in results]

#  Query GROQ LLM
def query_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "You are a meeting assistant. Summarize the meeting and extract action items."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

#  Parse the LLM output
def parse_response(result):
    lines = result.strip().split("\n")
    summary_lines = []
    tasks = []
    current_task = {}
    parsing_tasks = False

    for line in lines:
        if line.strip().lower().startswith("tasks"):
            parsing_tasks = True
            continue
        if not parsing_tasks:
            summary_lines.append(line)
        else:
            if "Task:" in line:
                if current_task:
                    tasks.append(current_task)
                current_task = {"task": line.split("Task:")[1].strip()}
            elif "Owner:" in line:
                current_task["owner"] = line.split("Owner:")[1].strip()
            elif "Deadline:" in line:
                current_task["deadline"] = line.split("Deadline:")[1].strip()
    if current_task:
        tasks.append(current_task)

    return "\n".join(summary_lines), tasks

#  FastAPI endpoint
@app.post("/process_meeting")
def process_transcript(request: TranscriptRequest):
    transcript = request.transcript
    context = get_rag_context(transcript)
    context_text = "\n\n".join(context)

    prompt = f"""
You are a project manager bot. Analyze this meeting transcript and related past notes.

Transcript:
{transcript}

Related Past Meetings:
{context_text}

Output Format:

Summary:
- Bullet points...

Tasks:
- Task: ...
  Owner: ...
  Deadline: ...
"""

    result = query_groq(prompt)
    summary, tasks = parse_response(result)

    return {
        "summary": summary,
        "tasks": tasks,
        "related_meetings": context
    }
