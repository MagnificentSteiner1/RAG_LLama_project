from fastapi import APIRouter
from API.scripts.schemes import AnswerResponse, QuestionRequest
from Agent.response_agent import AskQuestion

router = APIRouter(prefix="/API", tags=["Requests"])
@router.post("/question", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    return AskQuestion(request.question)
