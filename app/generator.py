from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL

UNAVAILABLE_MESSAGE = "I don't have enough information in the document to answer that."

SYSTEM_PROMPT = (
    "You are a precise Q&A assistant. You must answer ONLY using the information "
    "in the provided context. Do not use outside knowledge, do not guess, and do not "
    f"speculate. If the context does not contain the answer, respond with exactly: "
    f'"{UNAVAILABLE_MESSAGE}" and nothing else. Keep answers concise.'
)


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above:"
    )


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Call Groq to produce a grounded answer from the given context chunks."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

    client = Groq(api_key=GROQ_API_KEY)
    user_prompt = build_prompt(question, context_chunks)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=400,
    )

    return response.choices[0].message.content.strip()
