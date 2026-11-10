# Backend API for Curiosity Daily (search + tracking).
# Works on any container host that sets $PORT (Render, Hugging Face Spaces, Fly.io).
FROM python:3.11-slim

# Install uv (fast Python package manager).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (better layer caching).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --no-dev --frozen || uv sync --no-dev

ENV PORT=8000
EXPOSE 8000

# Bind to 0.0.0.0 and honour the host-provided $PORT.
CMD ["sh", "-c", "uv run curiosity serve --host 0.0.0.0 --port ${PORT:-8000}"]
