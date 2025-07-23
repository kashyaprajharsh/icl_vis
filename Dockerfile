# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /root/.local /home/app/.local

# Copy application code
COPY app/ ./app/
COPY requirements.txt .

# Create directory for model cache
RUN mkdir -p /home/app/.cache/huggingface && \
    chown -R app:app /home/app/.cache && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Add local bin to PATH
ENV PATH=/home/app/.local/bin:$PATH
ENV PYTHONPATH=/app

# Pre-download GPT-2 medium model to optimize cold starts
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('gpt2-medium'); AutoModelForCausalLM.from_pretrained('gpt2-medium')"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/')" || exit 1

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
