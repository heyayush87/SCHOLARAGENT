"""Central model + tracing configuration (OpenAI for chat, local FastEmbed for embeddings, LangSmith for tracing)."""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import FastEmbedEmbeddings

load_dotenv()

# ── LangSmith tracing ─────────────────────────────────────────────────────────
# Enabled automatically when LANGSMITH_API_KEY is set; LangChain/LangGraph calls are traced with no code changes.
_langsmith_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")  # LANGCHAIN_* is the legacy name
if _langsmith_key:
    os.environ.setdefault("LANGSMITH_API_KEY", _langsmith_key)
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "SCHOLARAGENT")

# ── Models ────────────────────────────────────────────────────────────────────
MAIN_MODEL = os.getenv("OPENAI_MAIN_MODEL", "gpt-5-mini")   # RAG graph: routing, answering, verification
FAST_MODEL = os.getenv("OPENAI_FAST_MODEL", "gpt-5-mini")    # side tasks: /btw, session titles
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")  # runs locally (ONNX): no API key, no per-use cost
EMBEDDING_DIM = 384  # bge-small-en-v1.5


MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "6000"))  # per call; reasoning models count hidden reasoning too
REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")       # low keeps reasoning tokens (and cost) down


def get_llm(fast: bool = False) -> ChatOpenAI:
    model = FAST_MODEL if fast else MAIN_MODEL
    extra = {"reasoning_effort": REASONING_EFFORT} if model.startswith(("gpt-5", "o")) else {}  # gpt-4o etc. reject it
    return ChatOpenAI(model=model, max_tokens=MAX_OUTPUT_TOKENS, **extra)


def get_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


def text_of(content) -> str:
    """Chat models return content as a str or a list of blocks; flatten to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else b.get("text", "")
            for b in content
            if isinstance(b, str) or (isinstance(b, dict) and b.get("type", "text") == "text")
        )
    return str(content or "")


_KEY_ERRORS = {"AuthenticationError", "PermissionDeniedError", "InvalidAPIKeyError", "ForbiddenError", "MissingAPIKeyError"}
_QUOTA_ERRORS = {"UsageLimitExceededError", "TavilyKeylessLimitError"}
_QUOTA_PHRASES = ("credit balance", "quota", "usage limit", "billing", "exceeds your plan")
_PROVIDERS = {"openai": "OpenAI (ChatGPT)", "tavily": "Tavily (web search)", "langsmith": "LangSmith"}


def limit_info(exc: BaseException) -> tuple[str, str] | None:
    """(provider, reason) if exc, or anything in its cause chain, is an API limit / key problem; else None."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        name, status, msg = type(exc).__name__, getattr(exc, "status_code", None), str(exc).lower()
        reason = None
        if name in _KEY_ERRORS:
            reason = "the API key was rejected (invalid, disabled or missing)"
        elif name in _QUOTA_ERRORS or status == 402 or any(p in msg for p in _QUOTA_PHRASES):
            reason = "the credit / usage quota has been used up"
        elif name == "RateLimitError" or status == 429 or "rate limit" in msg:
            reason = "the rate limit was exceeded"
        if reason:
            root = type(exc).__module__.split(".")[0]
            return _PROVIDERS.get(root, "an external API"), reason
        exc = exc.__cause__ or exc.__context__
    return None
