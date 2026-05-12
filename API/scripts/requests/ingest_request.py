from fastapi import APIRouter
from API.scripts.schemes import IngestResponse
import json
from Agent.ingest_agent import ingest_document
from fastapi import UploadFile
router = APIRouter(prefix="/API", tags=["Requests"])

@router.post("/ingest", response_model = IngestResponse)
async def ingest(file: UploadFile):
    content = await file.read()

    paper = json.loads(content)

    result = ingest_document(paper)

    return result