"""
OCD RAG Support – FastAPI backend
Designed to run within ~1.5 GB RAM on Railway (free/hobby tier).

Retrofit sends:
  POST /chat     → { session_id, messages: [{role, content}], severity? }
  POST /summary  → { session_id, messages: [{role, content}] }
"""

import os
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

# pypdf – lightweight, no torch/tesseract needed (~2 MB)
from pypdf import PdfReader

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN        = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN", "")
HF_LLM_REPO_ID  = os.getenv("HF_LLM_REPO_ID",  "meta-llama/Llama-3.1-8B-Instruct")
HF_EMBED_MODEL  = os.getenv("HF_EMBED_MODEL",   "sentence-transformers/all-MiniLM-L6-v2")
MAX_INPUT_CHARS = 3500
SEVERITY_LEVELS = {"LOW", "MILD", "HIGH"}

# ── Pydantic models ───────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str       # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    session_id: str
    messages: List[Message]          # last 10-15 messages from Kotlin
    severity: Optional[str] = None   # optional override from Kotlin classifier

class SummaryRequest(BaseModel):
    session_id: str
    messages: List[Message]          # full or partial history

class ChatResponse(BaseModel):
    session_id: str
    ai_response: str
    severity: str
    timestamp: str

class SummaryResponse(BaseModel):
    session_id: str
    generated_at: str
    summary_text: str
    message_count: int

# ── HuggingFace helpers (pure httpx, no LangChain) ────────────────────────────

HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

async def _hf_chat(system: str, user: str, max_new_tokens: int = 512) -> str:
    """Call HF Inference API chat endpoint."""
    url = f"https://api-inference.huggingface.co/models/{HF_LLM_REPO_ID}/v1/chat/completions"
    payload = {
        "model": HF_LLM_REPO_ID,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.4,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, headers=HF_HEADERS, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


async def _hf_embed(texts: List[str]) -> List[List[float]]:
    """Call HF feature-extraction endpoint for embeddings."""
    url = f"https://api-inference.huggingface.co/models/{HF_EMBED_MODEL}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=HF_HEADERS, json={"inputs": texts})
        resp.raise_for_status()
    return resp.json()  # list of float vectors

# ── Tiny in-process vector store (replaces FAISS, ~0 overhead) ───────────────

class TinyVectorStore:
    def __init__(self):
        self.chunks: List[str] = []
        self.matrix: Optional[np.ndarray] = None

    def is_ready(self) -> bool:
        return self.matrix is not None and len(self.chunks) > 0

    async def build(self, chunks: List[str]):
        self.chunks = chunks
        # HF embedding API has a max batch size — send in batches of 64
        all_vecs = []
        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vecs = await _hf_embed(batch)
            all_vecs.extend(vecs)

        self.matrix = np.array(all_vecs, dtype=np.float32)
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / np.where(norms == 0, 1, norms)

    async def search(self, query: str, k: int = 4) -> List[str]:
        if not self.is_ready():
            return []
        q_vec = await _hf_embed([query])
        q = np.array(q_vec[0], dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.matrix @ q
        indices = np.argsort(scores)[::-1][:k].tolist()
        return [self.chunks[i] for i in indices]


knowledge_store = TinyVectorStore()

# ── Document loaders ──────────────────────────────────────────────────────────

def _extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract all text from a PDF using pypdf.
    Falls back page-by-page and skips unreadable pages gracefully.
    Works on text-based PDFs (not scanned images — those need OCR).
    """
    text_parts: List[str] = []
    try:
        reader = PdfReader(str(pdf_path))
        print(f"  -> {pdf_path.name}: {len(reader.pages)} pages")
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
            except Exception as e:
                print(f"    Skipping page {i+1} of {pdf_path.name}: {e}")
    except Exception as e:
        print(f"  ERROR reading {pdf_path.name}: {e}")
    return "\n".join(text_parts)


def _load_text_chunks(knowledge_dir: Path, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    """
    Load .txt, .md, and .pdf files from knowledge_dir.
    Returns a flat list of text chunks for embedding.
    """
    chunks: List[str] = []
    file_count = {"txt": 0, "md": 0, "pdf": 0}

    def _chunk_text(text: str):
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if len(chunk) > 50:
                chunks.append(chunk)
            start += chunk_size - overlap

    # Plain text and Markdown
    for ext in ("*.txt", "*.md"):
        for f in knowledge_dir.rglob(ext):
            text = f.read_text(encoding="utf-8", errors="replace")
            _chunk_text(text)
            file_count["txt" if ext == "*.txt" else "md"] += 1

    # PDFs
    for f in knowledge_dir.rglob("*.pdf"):
        print(f"Loading PDF: {f.name}")
        pdf_text = _extract_pdf_text(f)
        if pdf_text.strip():
            _chunk_text(pdf_text)
            file_count["pdf"] += 1
        else:
            print(f"  WARNING: No text extracted from {f.name}. "
                  "It may be a scanned/image-only PDF. Convert to .txt manually.")

    print(
        f"Loaded {file_count['txt']} .txt, {file_count['md']} .md, "
        f"{file_count['pdf']} .pdf -> {len(chunks)} total chunks"
    )
    return chunks

# ── Severity helpers ──────────────────────────────────────────────────────────

def _coerce_severity(raw: str) -> str:
    text = (raw or "").strip().upper()
    if "HIGH" in text: return "HIGH"
    if "MILD" in text: return "MILD"
    return "LOW"

async def classify_severity(user_input: str) -> str:
    system = (
        "You are a strict mental health triage classifier for OCD support.\n"
        "Classify the user message severity as exactly one of: LOW, MILD, or HIGH.\n"
        "LOW  – minor intrusive thoughts, little functional impact.\n"
        "MILD – distress present, some functional impact, can still manage.\n"
        "HIGH – severe distress, strong impairment, safety risk, or inability to function.\n"
        "Return EXACTLY one token: LOW or MILD or HIGH. No other text."
    )
    return _coerce_severity(await _hf_chat(system, user_input, max_new_tokens=10))

def _policy_for_severity(severity: str) -> str:
    if severity == "LOW":
        return (
            "Provide coping advice and practical self-help. Console the patient warmly. "
            "Encourage small social activities and joyful hobbies. "
            "Suggest optional therapist check-in if symptoms persist."
        )
    if severity == "MILD":
        return (
            "Offer short coping suggestions. Gently encourage meeting a mental health professional soon, "
            "without pressure. Avoid framing self-help as sufficient alone."
        )
    return (
        "Remain calm and supportive. Strongly advise urgent contact with a licensed mental health professional. "
        "If there is immediate risk or self-harm concern, advise emergency services. "
        "Provide Indian emergency helplines: iCall 9152987821, Vandrevala Foundation 1860-2662-345, "
        "NIMHANS 080-46110007."
    )

# ── Build conversation context string ─────────────────────────────────────────

def _format_history(messages: List[Message]) -> str:
    lines = []
    for m in messages:
        prefix = "User" if m.role == "user" else "Assistant"
        lines.append(f"{prefix}: {m.content}")
    return "\n".join(lines)

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="OCD RAG Support API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    root = Path(__file__).resolve().parent
    kd   = Path(os.getenv("OCD_KNOWLEDGE_DIR", str(root / "ocd_documentation")))
    if kd.is_dir():
        chunks = _load_text_chunks(kd)
        if chunks:
            print(f"Building vector store from {len(chunks)} chunks...")
            await knowledge_store.build(chunks)
            print("Vector store ready.")
        else:
            print("Warning: knowledge_dir exists but no readable files found.")
    else:
        print(f"Warning: knowledge dir not found at {kd}. RAG context will be empty.")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "vector_store_ready": knowledge_store.is_ready(),
        "chunk_count": len(knowledge_store.chunks),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages list is empty")

    last_user_msg = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    ).strip()[:MAX_INPUT_CHARS]

    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found in messages list")

    model_severity = await classify_severity(last_user_msg)
    final_severity = _coerce_severity(req.severity) if req.severity else model_severity

    context_chunks = await knowledge_store.search(last_user_msg, k=4)
    context = "\n".join(context_chunks) if context_chunks else "No specific clinical context available."

    history_text = _format_history(req.messages[:-1])

    system = (
        "You are an OCD support assistant. You are NOT a doctor.\n"
        "Use the provided Clinical Context and Chat History to respond safely.\n\n"
        f"Clinical Context:\n{context}\n\n"
        f"Chat History:\n{history_text}\n\n"
        f"Current severity: {final_severity}\n"
        f"Policy: {_policy_for_severity(final_severity)}\n\n"
        "Rules:\n"
        "- Always refer to context when relevant\n"
        "- Be warm and empathetic\n"
        "- No diagnosis or medication instructions\n"
        "- Max 150 words"
    )

    ai_response = await _hf_chat(system, last_user_msg)

    return ChatResponse(
        session_id=req.session_id,
        ai_response=ai_response,
        severity=final_severity,
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.post("/summary", response_model=SummaryResponse)
async def summary(req: SummaryRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages list is empty")

    history_blob = _format_history(req.messages)

    system = (
        "Create a compact doctor-facing session summary for an OCD patient support session.\n"
        "Include:\n"
        "1. Severity trend across the session\n"
        "2. Main symptoms and triggers mentioned\n"
        "3. Functional impact described\n"
        "4. Any risk or safety notes\n"
        "5. Advice given by the assistant\n"
        "6. Recommended next steps for the clinician\n"
        "Be concise and clinical. Max 300 words."
    )

    summary_text = await _hf_chat(
        system,
        f"Session ID: {req.session_id}\n\n{history_blob}",
        max_new_tokens=600,
    )

    return SummaryResponse(
        session_id=req.session_id,
        generated_at=datetime.now(UTC).isoformat(),
        summary_text=summary_text,
        message_count=len(req.messages),
    )