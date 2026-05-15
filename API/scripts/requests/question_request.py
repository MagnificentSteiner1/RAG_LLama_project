from fastapi import APIRouter
from API.scripts.schemes import AnswerResponse, QuestionRequest
from Agent.response_agent import AskQuestion
from Agent.core_agent import  vector_store
router = APIRouter(prefix="/API", tags=["Requests"])

@router.get("/debug/vector-count")
def vector_count():
    return {
        "count": vector_store._collection.count()
    }

@router.post("/question", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    return AskQuestion(request.question)
