# ---- Build stage ----
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (without the project itself)
RUN uv sync --frozen --no-install-project

# Copy entire source code
COPY . .

# Install the project
RUN uv sync --frozen

# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY --from=builder /app /app

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Create necessary directories
RUN mkdir -p logs data knowledge_base

# Expose the default uvicorn port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
