# Dockerfile for Round 1B - Persona-Driven Document Intelligence
FROM --platform=linux/amd64 python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Download and cache sentence transformer model (stays under 1GB limit)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy source code structure
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p /app/input /app/output

# Set Python path to find modules
ENV PYTHONPATH=/app/src

# Default command
CMD ["python", "src/main.py"]