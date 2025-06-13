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

# Database migration
python scripts/db_migrate.py

# Check if Tailwind CSS file exists
if [ ! -f ./static/dist/tailwind.css ] || [ "$(stat -c%s ./static/dist/tailwind.css 2>/dev/null || stat -f%z ./static/dist/tailwind.css)" -lt "1000" ]; then
  echo "Tailwind CSS file missing or too small, using fallback..."
  ./scripts/create_fallback_css.sh
fi

echo "Starting Pokemon Generator on port $PORT..."
echo "Access the application at: http://localhost:$PORT"

# Use gunicorn for production WSGI serving
exec gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 3 --access-logfile - --error-logfile - app:app
