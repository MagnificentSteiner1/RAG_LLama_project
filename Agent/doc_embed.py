import pandas as pd
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

BASE_DIR = Path(__file__).resolve().parent.parent
data_path = BASE_DIR / "Documents"/"Quality_of_Life.csv"
df = pd.read_csv(data_path)
print(df)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = Path(__file__).resolve().parent / "Database"
chroma_data_file = db_location / "chroma.sqlite3"
add_documents = not chroma_data_file.exists()

if add_documents:
    documents = []
    ids = []
    for index, row in df.iterrows():
        content = " | ".join([f"{col}: {row[col]}" for col in df.columns])

        document = Document(
            page_content=content,
            metadata={"source": "Quality_of_Life.csv", "row_index": index},
            id=str(index)
        )
        ids.append(str(index))
        documents.append(document)
vector_store=Chroma(
    collection_name="quality_of_life_data",
    persist_directory = db_location,
    embedding_function=embeddings
)
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever=vector_store.as_retriever(
    search_kwargs={"k":20}
)


