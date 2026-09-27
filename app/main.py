
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import SIMILARITY_THRESHOLD
from app.generator import generate_answer, UNAVAILABLE_MESSAGE
from app.retriever import retrieve

app = FastAPI(
    title="Lightweight RAG Document Q&A API",
    description="Ask questions grounded strictly in an ingested document.",
    version="1.0.0",
)


class Question(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask about the document.")


class SourceSnippet(BaseModel):
    text: str
    score: float


class AnswerResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[SourceSnippet]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(payload: Question):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    retrieved_chunks = retrieve(question)

    relevant_chunks = [c for c in retrieved_chunks if c["score"] >= SIMILARITY_THRESHOLD]

    if not relevant_chunks:
        return AnswerResponse(
            answer=UNAVAILABLE_MESSAGE,
            grounded=False,
            sources=[],
        )

    context_texts = [c["text"] for c in relevant_chunks]

    try:
        answer = generate_answer(question, context_texts)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    grounded = UNAVAILABLE_MESSAGE.lower() not in answer.lower()

    return AnswerResponse(
        answer=answer,
        grounded=grounded,
        sources=[SourceSnippet(text=c["text"], score=c["score"]) for c in relevant_chunks],
    )
