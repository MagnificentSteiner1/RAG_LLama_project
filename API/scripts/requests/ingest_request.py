from fastapi import APIRouter
from API.scripts.schemes import IngestResponse
import json
import os
from Agent.ingest_agent import ingest_document
from fastapi import UploadFile
router = APIRouter(prefix="/API", tags=["Requests"])
DATA_DIR = "/app/Documents"
@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile):
    content = await file.read()
    file_path = os.path.join(DATA_DIR, file.filename)
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(content)

    paper = json.loads(content)
    result = ingest_document(paper=paper, source=file.filename)

    return result