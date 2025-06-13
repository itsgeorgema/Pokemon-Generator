# Optimized Dockerfile for Pokemon Generator
# Multi-stage build to reduce the final image size

# Build stage for Tailwind CSS
FROM node:20-alpine AS css-builder
WORKDIR /build

# Copy only files needed for CSS building
COPY package.json package-lock.json* ./
COPY tailwind.config.js postcss.config.js ./
COPY static/src ./static/src
COPY scripts/build_tailwind.js ./scripts/

# Install only production dependencies and build the CSS
RUN npm ci --only=production && \
    mkdir -p static/dist && \
    node scripts/build_tailwind.js

# Main Python application
FROM python:3.9-slim AS final

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    PYTHONPATH=/app \
    PORT=5001

# Install only necessary system dependencies in a single layer
# - build-essential for compiling Python packages
# - libpq-dev for PostgreSQL client
# - curl for healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        netcat-openbsd && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p static/generated static/samples logs data/images models static/dist

# Copy the built CSS from the first stage
COPY --from=css-builder /build/static/dist/tailwind.css static/dist/

# Copy application code
COPY . .

# Copy fallback CSS script for emergencies
COPY scripts/create_fallback_css.sh scripts/
RUN chmod +x scripts/start_server.sh scripts/create_fallback_css.sh

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5001/ || exit 1

# Command to run the application
CMD ["./scripts/start_server.sh"]
