import json
import pandas as pd
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "Documents/pdf jsons"

db_location = Path(__file__).resolve().parent.parent / "Database"
chroma_data_file = db_location / "chroma.sqlite3"
add_documents = not chroma_data_file.exists()
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
# chunkuje jedan fajl
def build_documents(paper):
    title = paper["metadata"]["title"]
    paper_id = paper["paper_id"]

    paragraphs = [p["text"] for p in paper["body_text"] if p["text"].strip()]

    documents = []
    current_chunk = ""
    chunk_size = 2000  # char-based proxy
    chunk_index = 0

    for p in paragraphs:
        if len(current_chunk) + len(p) < chunk_size:
            current_chunk += " " + p
        else:
            documents.append(
                Document(
                    page_content=f"Title: {title}\n\n{current_chunk}",
                    metadata={
                        "paper_id": paper_id,
                        "title": title,
                        "chunk_index": chunk_index
                    }
                )
            )
            current_chunk = p
            chunk_index += 1

    if current_chunk:
        documents.append(
            Document(
                page_content=f"Title: {title}\n\n{current_chunk}",
                metadata={
                    "paper_id": paper_id,
                    "title": title,
                    "chunk_index": chunk_index
                }
            )
        )

    return documents

all_docs = []
i=0
for file_name in os.listdir(data_path):
    i+=1
    print(i)
    with open(os.path.join(data_path, file_name), "r", encoding="utf-8") as f:
        paper = json.load(f)
    docs = build_documents(paper)
    docs = splitter.split_documents(docs)
    all_docs.extend(docs)

print('docs len', len(all_docs))
vector_store=Chroma.from_documents(
    documents=all_docs,
    collection_name="CORD-19",
    persist_directory = db_location,
    embedding=embeddings
)


