FROM python:3.12-slim

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    USER_AGENT=scholaragent

# ── Layer 1: install deps from the lockfile (cached until pyproject/uv.lock change) ──
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Layer 2: backend package and documents ────────────────────────────────────
COPY backend/ backend/
COPY documents/ documents/

# ── Layer 3: application files (change most often — last for cache efficiency) ──
COPY app.py evaluate.py main.py goldens.json ./

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
