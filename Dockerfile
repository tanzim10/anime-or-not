FROM python:3.10.16-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY src/ src/

# Copy model file if it exists, otherwise download it
ARG MODEL_URL
COPY best_model.pth* ./
RUN if [ ! -f best_model.pth ] && [ -n "$MODEL_URL" ]; then \
        curl -L "$MODEL_URL" -o best_model.pth; \
    fi

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/model/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]



