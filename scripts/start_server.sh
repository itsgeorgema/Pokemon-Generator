#!/bin/sh
set -e

if [ -f .env ]; then
  echo "Loading environment variables from .env file"
  export $(grep -v '^#' .env | xargs)
fi

chmod +x scripts/init_db.py

export FLASK_ENV=${FLASK_ENV:-production}
export FLASK_APP=${FLASK_APP:-app.py}
export PORT=${PORT:-5001}
export DOCKER_CONTAINER=true

# Create necessary directories
mkdir -p static/generated static/samples logs data/images models

# Remove nc check for PostgreSQL (not needed for Render)
# echo "Waiting for PostgreSQL..."
# while ! nc -z db 5432; do
#   echo "PostgreSQL is unavailable - sleeping"
#   sleep 1
# done
# echo "PostgreSQL is up - continuing"

# Run database migrations
python scripts/db_migrate.py

echo "Starting Pokemon Generator on port $PORT..."
echo "Access the application at: http://localhost:$PORT"
exec python main.py 