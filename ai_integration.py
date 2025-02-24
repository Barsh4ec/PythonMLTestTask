import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

import uuid


client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="knowledge_base")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def process_basic_knowledge(csv_path):
    df = pd.read_csv(csv_path)
    ids = []
    cocktails = []
    embeddings = []
    metadatas = []

    for _, row in df.iterrows():
        ids.append(f"{row['name'].replace(' ', '_').lower()}")
        cocktails.append(convert_cocktail(row.to_dict()))
        embedding = model.encode(convert_cocktail(row.to_dict()))
        embeddings.append(embedding)
        metadata = {
            "id": row["id"],
            "name": row["name"],
            "instructions": row["instructions"],
            "ingredients": row["ingredients"],
            "alcoholic": row["alcoholic"].strip().lower() == "alcoholic",
        }
        metadatas.append(metadata)

    collection.add(
        ids=ids,
        documents=cocktails,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print(f"Added {csv_path} to ChromaDB")


def add_user_knowledge(text):
    unique_id = str(uuid.uuid4())
    embedding = model.encode(text).tolist()
    collection.add(
        ids=[unique_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"source": "user_input"}]
    )
    print(f"Added knowledge to ChromaDB")


def convert_cocktail(row):
    text = f"Name: {row['name']} (Alcoholic: {row['alcoholic']}): {row['ingredients']}"
    return text

