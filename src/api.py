import pathlib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import CodeLLM
from .retriever import CodeRetriever

app = FastAPI(title="Local Code‑Assist Agent")

# Initialise once at import time
PROJECT_ROOT = pathlib.Path.cwd()
llm = CodeLLM()
retriever = CodeRetriever(project_root=PROJECT_ROOT)

class CompletionRequest(BaseModel):
    # The snippet the user is editing (may be incomplete)
    code: str
    # Cursor offset in characters from the beginning of `code`
    cursor: int
    # Optional file path (used for better retrieval)
    file_path: str | None = None

@app.post("/complete")
def complete(req: CompletionRequest):
    # 1️⃣ Build retrieval query – we use the code + file name
    query = f"File: {req.file_path or 'unknown'}\\nSnippet: {req.code[:req.cursor]}"
    context = retriever.retrieve(query, top_k=5)

    # 2️⃣ Construct prompt (few‑shot style)
    prompt = f"""You are an AI coding assistant. Use the provided context to generate the most likely continuation for the user's code.
### Context
{context}
### User code (cursor at ⬇️)
{req.code[:req.cursor]}⬇️
### Completion (only raw code, no explanations)"""
    # 3️⃣ Call the model
    try:
        completion = llm.complete(prompt, max_new_tokens=256, temperature=0.1,
                                 stop=["```", "<|endoftext|>"])
        return {"completion": completion}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))