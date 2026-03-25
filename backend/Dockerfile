FROM python:3.11-slim

WORKDIR /app

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install CPU-only PyTorch FIRST
# -----------------------------
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# -----------------------------
# Install remaining Python deps
# -----------------------------
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir \
    sentence-transformers \
    scikit-learn \
    numpy \
    -r requirements.txt

# -----------------------------
# Copy application code
# -----------------------------
COPY backend/ ./
COPY backend/requirements.txt .

WORKDIR /app

EXPOSE 1007

# -----------------------------
# Start FastAPI
# -----------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "1007"]
