#!/bin/sh
# Optimized start server script
set -e

if [ -f .env ]; then
  echo "Loading environment variables from .env file"
  export $(grep -v '^#' .env | xargs)
fi

# Set default environment variables
export FLASK_ENV=${FLASK_ENV:-production}
export FLASK_APP=${FLASK_APP:-app.py}
export PORT=${PORT:-5001}
export HOST=${HOST:-0.0.0.0}
export DOCKER_CONTAINER=true

# Optional: run DB migrations on boot (can be disabled with RUN_DB_MIGRATIONS=false)
if [ "${RUN_DB_MIGRATIONS:-true}" = "true" ]; then
  echo "Running database migrations..."
  python scripts/db_migrate.py
fi

# Create static directories if they don't exist
mkdir -p ./static/dist

# At runtime, avoid doing heavy Tailwind builds. The CSS is already built at
# Docker image build time in the multi-stage Dockerfile. Here we only ensure
# that a usable CSS file exists and, if not, fall back to a very lightweight
# shell-generated stylesheet instead of invoking Node tooling.
if [ ! -f ./static/dist/tailwind.css ]; then
  echo "Tailwind CSS not found at runtime, using fallback CSS..."
  ./scripts/create_fallback_css.sh
fi

# Print some debug information
echo "Contents of static directory:"
ls -la ./static
echo "Contents of static/dist directory:"
ls -la ./static/dist
echo "CSS file size:"
du -h ./static/dist/tailwind.css

echo "Starting Pokemon Generator on port $PORT..."
echo "Access the application at: http://localhost:$PORT"

# Use gunicorn for production WSGI serving
exec gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 3 --access-logfile - --error-logfile - app:app
