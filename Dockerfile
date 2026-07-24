FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/chat/', timeout=4)" || exit 1

# config.toml is NOT baked into the image (see .dockerignore); mount it at
# /app/config.toml at runtime.
# --root-path /chat: all routes are served under /chat so the frontend nginx
# can proxy the agent same-origin without rewriting.
CMD ["uv", "run", "--no-sync", "chainlit", "run", "chainlit_app3.py", "-h", "--host", "0.0.0.0", "--port", "8000", "--root-path", "/chat"]
