from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from typing import List

from ai_integration import process_basic_knowledge

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


def retrieve_knowledge(query):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="knowledge_base")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    query_embedding = model.encode(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)

    if results and "documents" in results and results["documents"] and results["documents"][0]:
        return results["documents"][0][0]
    else:
        return None


chat_history: List[dict] = []


@app.get("/")
async def get_chat(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request, "messages": chat_history})


@app.post("/")
async def post_message(request: Request, message: str = Form(...)):
    response = process_message(message)
    chat_history.append({"user": message, "bot": response})
    return templates.TemplateResponse("chat.html", {"request": request, "messages": chat_history})


def process_message(message: str) -> str:
    retrieved_text = retrieve_knowledge(message)
    if retrieved_text:
        response = ollama.chat(
            model="deepseek-r1:7b",
            messages=[{"role": "user", "content": f"{message}. Here is some info {retrieved_text}"}]
        )
        return response["message"]["content"]
    else:
        return f"Not a relevant question"
