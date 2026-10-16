"""Central model + tracing configuration (Anthropic for chat, local FastEmbed for embeddings, LangSmith for tracing)."""
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import FastEmbedEmbeddings

load_dotenv()

# ── LangSmith tracing ─────────────────────────────────────────────────────────
# Enabled automatically when LANGSMITH_API_KEY is set; LangChain/LangGraph calls are traced with no code changes.
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "SCHOLARAGENT")

# ── Models ────────────────────────────────────────────────────────────────────
MAIN_MODEL = os.getenv("ANTHROPIC_MAIN_MODEL", "claude-sonnet-5-5")   # RAG graph: routing, answering, verification
FAST_MODEL = os.getenv("ANTHROPIC_FAST_MODEL", "claude-haiku-5-5")    # side tasks: /btw, session titles
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")  # runs locally (ONNX): no API key, no per-use cost
EMBEDDING_DIM = 384  # bge-small-en-v1.5


def get_llm(fast: bool = False) -> ChatAnthropic:
    return ChatAnthropic(model=FAST_MODEL if fast else MAIN_MODEL, max_tokens=4096)


def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def text_of(content) -> str:
    """Claude returns content as a str or a list of blocks; flatten to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else b.get("text", "")
            for b in content
            if isinstance(b, str) or (isinstance(b, dict) and b.get("type", "text") == "text")
        )
    return str(content or "")
