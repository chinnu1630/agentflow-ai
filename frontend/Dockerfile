# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip && \
    /opt/venv/bin/python -m pip install "."

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    STREAMLIT_SERVER_ADDRESS="0.0.0.0" \
    STREAMLIT_SERVER_PORT="8501" \
    STREAMLIT_SERVER_HEADLESS="true" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"

WORKDIR /app

RUN groupadd --system agentflow && \
    useradd --system --gid agentflow --home-dir /app agentflow

COPY --from=builder /opt/venv /opt/venv
COPY streamlit_app.py ./streamlit_app.py

RUN chown -R agentflow:agentflow /app

USER agentflow

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)"

CMD ["streamlit", "run", "streamlit_app.py"]
