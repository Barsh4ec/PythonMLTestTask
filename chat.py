from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

from typing import List
import re
import markdown

from ai_integration import process_basic_knowledge, add_user_knowledge


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

process_basic_knowledge("./basic_knowledge/final_cocktails.csv")

chat_history: List[dict] = []


def retrieve_knowledge(query: str) -> str | None:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="knowledge_base")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    query_embedding = model.encode(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)

    if results and "documents" in results and results["documents"] and results["documents"][0]:
        return results["documents"][0][0]
    else:
        return None


def clean_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


@app.get("/")
async def get_chat(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("chat.html", {"request": request, "messages": chat_history})


@app.post("/")
async def post_message(request: Request, message: str = Form(...)) -> HTMLResponse:
    add_user_knowledge(message)
    response = clean_text(process_message(message))
    html_text = markdown.markdown(response)
    chat_history.append({"user": message, "bot": html_text})
    return templates.TemplateResponse("chat.html", {"request": request, "messages": chat_history})


def process_message(message: str) -> str:
    retrieved_text = retrieve_knowledge(message)
    response = ollama.chat(
        model="deepseek-r1:7b",
        messages=[
            {
                "role": "system",
                "content": "You are AI bartender that helps user to choose cocktails."
            },
            {
                "role": "user",
                "content": f"{message}. Here is some info {retrieved_text}"
            }
        ]
    )
    return response["message"]["content"]
