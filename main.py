"""
OCD RAG Support – FastAPI backend v2.2
Fixes:
  - HF embedding API 410 Gone (api-inference.huggingface.co deprecated)
  - 6318 chunk startup timeout — embeddings now run locally via sentence-transformers
  - LLM still calls HF router API (no GPU needed for inference)
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN       = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN", "")
HF_LLM_REPO_ID = os.getenv("HF_LLM_REPO_ID", "meta-llama/Llama-3.1-8B-Instruct")
# all-MiniLM-L6-v2 is ~90 MB — runs fine on Railway CPU, no GPU needed
EMBED_MODEL_ID = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_INPUT_CHARS = 3500

HF_HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

# ── Pydantic models ───────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str        # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    session_id: str
    messages: List[Message]
    severity: Optional[str] = None

class SummaryRequest(BaseModel):
    session_id: str
    messages: List[Message]

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

# ── Local embedding model (loaded once at startup, ~90 MB) ────────────────────
# Runs on CPU. No HF API quota consumed for embeddings.

_embed_model: Optional[SentenceTransformer] = None

def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print(f"Loading local embedding model: {EMBED_MODEL_ID}")
        _embed_model = SentenceTransformer(EMBED_MODEL_ID)
        print("Embedding model loaded.")
    return _embed_model

def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts locally. Returns (N, D) float32 array."""
    model = get_embed_model()
    vecs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)

# ── HuggingFace LLM helper (remote API) ──────────────────────────────────────
# Updated to new router.huggingface.co endpoint — api-inference.huggingface.co is deprecated.

async def _hf_chat(system: str, user: str, max_new_tokens: int = 512) -> str:
    url = f"https://router.huggingface.co/hf-inference/models/{HF_LLM_REPO_ID}/v1/chat/completions"
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

# ── Tiny local vector store ───────────────────────────────────────────────────

class TinyVectorStore:
    def __init__(self):
        self.chunks: List[str] = []
        self.matrix: Optional[np.ndarray] = None   # (N, D) normalized float32

    def is_ready(self) -> bool:
        return self.matrix is not None and len(self.chunks) > 0

    def build(self, chunks: List[str]):
        """Synchronous — called once at startup before the event loop is busy."""
        self.chunks = chunks
        print(f"Embedding {len(chunks)} chunks locally (CPU)...")
        self.matrix = embed_texts(chunks)
        print(f"Vector store built: {self.matrix.shape}")

    def search(self, query: str, k: int = 4) -> List[str]:
        if not self.is_ready():
            return []
        q = embed_texts([query])[0]          # already normalized
        scores = self.matrix @ q
        indices = np.argsort(scores)[::-1][:k].tolist()
        return [self.chunks[i] for i in indices]


knowledge_store = TinyVectorStore()

# ── Document loaders ──────────────────────────────────────────────────────────

def _extract_pdf_text(pdf_path: Path) -> str:
    text_parts: List[str] = []
    try:
        reader = PdfReader(str(pdf_path))
        print(f"  -> {pdf_path.name}: {len(reader.pages)} pages")
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
                if t.strip():
                    text_parts.append(t)
            except Exception as e:
                print(f"    Skipping page {i+1}: {e}")
    except Exception as e:
        print(f"  ERROR reading {pdf_path.name}: {e}")
    return "\n".join(text_parts)


def _load_text_chunks(knowledge_dir: Path, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    chunks: List[str] = []
    counts = {"txt": 0, "md": 0, "pdf": 0}

    def _chunk(text: str):
        start = 0
        while start < len(text):
            c = text[start : start + chunk_size].strip()
            if len(c) > 50:
                chunks.append(c)
            start += chunk_size - overlap

    for ext in ("*.txt", "*.md"):
        for f in knowledge_dir.rglob(ext):
            _chunk(f.read_text(encoding="utf-8", errors="replace"))
            counts["txt" if ext == "*.txt" else "md"] += 1

    for f in knowledge_dir.rglob("*.pdf"):
        print(f"Loading PDF: {f.name}")
        text = _extract_pdf_text(f)
        if text.strip():
            _chunk(text)
            counts["pdf"] += 1
        else:
            print(f"  WARNING: No text extracted from {f.name} (may be scanned/image PDF).")

    print(f"Loaded {counts['txt']} .txt, {counts['md']} .md, {counts['pdf']} .pdf -> {len(chunks)} chunks")
    return chunks

# ── Severity helpers ──────────────────────────────────────────────────────────

def _coerce_severity(raw: str) -> str:
    t = (raw or "").strip().upper()
    if "HIGH" in t: return "HIGH"
    if "MILD" in t: return "MILD"
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
        "Indian emergency helplines: iCall 9152987821, Vandrevala Foundation 1860-2662-345, "
        "NIMHANS 080-46110007."
    )

def _format_history(messages: List[Message]) -> str:
    return "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
        for m in messages
    )

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="OCD RAG Support API", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    # Load embedding model first (downloads ~90 MB on first run, cached after)
    get_embed_model()

    root = Path(__file__).resolve().parent
    kd   = Path(os.getenv("OCD_KNOWLEDGE_DIR", str(root / "ocd_documentation")))

    if kd.is_dir():
        chunks = _load_text_chunks(kd)
        if chunks:
            knowledge_store.build(chunks)   # synchronous CPU embedding, fast enough
        else:
            print("Warning: no readable files found in knowledge_dir.")
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
        raise HTTPException(status_code=400, detail="No user message found")

    model_severity = await classify_severity(last_user_msg)
    final_severity = _coerce_severity(req.severity) if req.severity else model_severity

    # search() is synchronous CPU — fast on 6k chunks
    context_chunks = knowledge_store.search(last_user_msg, k=4)
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
        "- Refer to context when relevant\n"
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
        f"Session ID: {req.session_id}\n\n{_format_history(req.messages)}",
        max_new_tokens=600,
    )

    return SummaryResponse(
        session_id=req.session_id,
        generated_at=datetime.now(UTC).isoformat(),
        summary_text=summary_text,
        message_count=len(req.messages),
    )