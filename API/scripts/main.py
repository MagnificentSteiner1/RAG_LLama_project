from fastapi import FastAPI
from API.scripts.requests.question_request import router as question_router
from API.scripts.requests.ingest_request import router as ingest_router


#Skripta za predvidjanje ban rate-a, ako se unese ime championa i pozicija
app = FastAPI()
app.include_router(question_router)
app.include_router(ingest_router)