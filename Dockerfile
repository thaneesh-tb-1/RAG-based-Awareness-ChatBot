FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    HOST=0.0.0.0

WORKDIR /app

# System deps (build-essential for any native wheels)
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the service port (for local clarity; Cloud Run detects PORT)
EXPOSE 8080

# Run with Gunicorn, pointing to Flask app object in server.py
# Note: BASE_URL should be set via environment variables when running the container
CMD ["bash", "-lc", "exec gunicorn 'server:app' --bind ${HOST}:${PORT} --workers 2 --threads 4 --timeout 120"]


