from dataclasses import Field
from enum import Enum

from fastapi import FastAPI
from pydantic import BaseModel, Field
app = FastAPI()


class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    context: list[str]
class IngestResponse(BaseModel):
    status: str
    chunks_added: int