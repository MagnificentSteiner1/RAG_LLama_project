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

db_location = "/app/Database"
chroma_data_file = Path(db_location) / "chroma.sqlite3"
add_documents = True
embeddings = OllamaEmbeddings(model="mxbai-embed-large"
                              ,base_url="http://ollama:11434")
#Jer se skripta pokrece direktno iz containera, mora se dodati base url na embeddings i modelu, da bi se pokrenuli iz llama containera, a ne lokalno
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
# chunkuje jedan fajl
def build_documents(paper, file_name):
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
                        "chunk_index": chunk_index,
                        "file_name": file_name
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
                    "chunk_index": chunk_index,
                    "file_name": file_name
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
    docs = build_documents(paper, file_name)
    docs = splitter.split_documents(docs)
    all_docs.extend(docs)

print('docs len', len(all_docs))
vector_store = Chroma(
    collection_name="CORD-19",
    persist_directory=str(db_location),
    embedding_function=embeddings
)

BATCH_SIZE = 2000

for i in range(0, len(all_docs), BATCH_SIZE):
    print(i)
    batch = all_docs[i:i + BATCH_SIZE]

    print(f"Inserting batch {i} - {i + len(batch)}")

    vector_store.add_documents(batch)
