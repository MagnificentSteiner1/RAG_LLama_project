import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Agent.core_agent import vector_store

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
def build_documents(paper):

    docs = []

    title = paper["metadata"].get("title", "")

    for idx, section in enumerate(
        paper.get("body_text", [])
    ):

        text = section.get("text", "")

        if not text.strip():
            continue

        docs.append(
            Document(
                page_content=f"Title: {title}\n\n{text}",
                metadata={
                    "paper_id": paper.get("paper_id"),
                    "chunk_index": idx
                }
            )
        )

    return docs

def ingest_document(paper: dict):

    docs = build_documents(paper)

    split_docs = splitter.split_documents(docs)

    vector_store.add_documents(split_docs)

    return {
        "status": "success",
        "chunks_added": len(split_docs)
    }