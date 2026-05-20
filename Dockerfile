# --- build stage ---
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[dev]" --target /build/deps

# --- runtime stage ---
FROM python:3.11-slim
WORKDIR /app

# LightGBM/XGBoost need libgomp1 (OpenMP) at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages
COPY src/ ./src/
COPY configs/ ./configs/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
