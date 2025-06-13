# Optimized Dockerfile for Pokemon Generator
# Multi-stage build to reduce the final image size

# Build stage for Tailwind CSS
FROM node:20-alpine AS css-builder
WORKDIR /build

# Copy only files needed for CSS building
COPY package.json package-lock.json* ./
COPY tailwind.config.js postcss.config.js ./

# Create directory structure
RUN mkdir -p static/src static/dist

# Copy the source CSS
COPY static/src/tailwind.css static/src/

# Install Node.js dependencies
RUN if [ -f package-lock.json ]; then \
        npm ci; \
    else \
        npm install; \
    fi

# Build the CSS using Tailwind CLI directly, more reliable than the script
RUN npx tailwindcss -i ./static/src/tailwind.css -o ./static/dist/tailwind.css --minify || \
    (echo "Failed to build Tailwind CSS with npx, falling back to alternative method..." && \
     node -e "const fs=require('fs'); const css=fs.readFileSync('./static/src/tailwind.css', 'utf8'); fs.writeFileSync('./static/dist/tailwind.css', '/*! tailwindcss fallback */'+css)")

# Verify the CSS was built
RUN ls -la static/dist && \
    echo "CSS file size:" && \
    du -h static/dist/tailwind.css

# Main Python application
FROM python:3.9-slim AS final

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    PYTHONPATH=/app \
    PORT=5001 \
    DOCKER_CONTAINER=true

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        netcat-openbsd \
        nodejs \
        npm && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p static/generated static/samples logs data/images models static/dist

# Copy the built CSS from the first stage
COPY --from=css-builder /build/static/dist/tailwind.css static/dist/

# Copy package.json and tailwind config files
COPY package.json tailwind.config.js postcss.config.js ./
COPY static/src ./static/src/

# Install Node.js dependencies for potential runtime rebuilds
# Use npm install with --omit=dev instead of npm ci when package-lock.json might not exist
RUN if [ -f package-lock.json ]; then \
        npm ci --omit=dev; \
    else \
        npm install --omit=dev; \
    fi

# Copy application code
COPY . .

# Make scripts executable
RUN chmod +x scripts/start_server.sh scripts/create_fallback_css.sh

# Verify CSS file exists and has content
RUN ls -la static/dist && \
    echo "CSS file size:" && \
    du -h static/dist/tailwind.css

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5001/ || exit 1

# Expose the application port
EXPOSE 5001

# Command to run the application
CMD ["./scripts/start_server.sh"]
