#!/bin/sh
set -e

if [ -f .env ]; then
  echo "Loading environment variables from .env file"
  export $(grep -v '^#' .env | xargs)
fi

chmod +x scripts/init_db.py
chmod +x scripts/create_fallback_css.sh

export FLASK_ENV=${FLASK_ENV:-production}
export FLASK_APP=${FLASK_APP:-app.py}
export PORT=${PORT:-5001}
export DOCKER_CONTAINER=true

mkdir -p static/generated static/samples logs data/images models

# Remove nc check for PostgreSQL (not needed for Render)
# echo "Waiting for PostgreSQL..."
# while ! nc -z db 5432; do
#   echo "PostgreSQL is unavailable - sleeping"
#   sleep 1
# done
# echo "PostgreSQL is up - continuing"

python scripts/db_migrate.py

# Build Tailwind CSS for production
if [ -f package.json ]; then
  echo "Installing npm dependencies and building Tailwind CSS..."
  # Always install dependencies, including those needed for Tailwind
  npm install || {
    echo "Warning: npm install failed, but continuing as we have a pre-built CSS file"
  }
  
  # Make the build script executable
  chmod +x scripts/build_tailwind.js
  
  # Try various methods to build Tailwind CSS
  echo "Building Tailwind CSS..."
  
  # Method 1: Use npx directly
  if command -v npx &> /dev/null; then
    echo "Trying to build with npx..."
    npx tailwindcss -i ./static/src/tailwind.css -o ./static/dist/tailwind.css --minify || {
      echo "Warning: npx build failed, trying alternative method..."
    }
  else
    echo "npx not found, trying alternative method..."
  fi
  
  # Method 2: Use our custom Node.js script
  if [ ! -f ./static/dist/tailwind.css ] || [ "$(stat -c%s ./static/dist/tailwind.css 2>/dev/null || stat -f%z ./static/dist/tailwind.css)" -lt "1000" ]; then
    echo "Trying to build with Node.js script..."
    node scripts/build_tailwind.js || {
      echo "Warning: Node.js script build failed, continuing with pre-built CSS file"
    }
  fi
  
  # Method 3: Directly use node_modules binary
  if [ ! -f ./static/dist/tailwind.css ] || [ "$(stat -c%s ./static/dist/tailwind.css 2>/dev/null || stat -f%z ./static/dist/tailwind.css)" -lt "1000" ]; then
    echo "Trying to build with node_modules binary..."
    if [ -f ./node_modules/.bin/tailwindcss ]; then
      ./node_modules/.bin/tailwindcss -i ./static/src/tailwind.css -o ./static/dist/tailwind.css --minify || {
        echo "Warning: node_modules binary build failed, continuing with pre-built CSS file"
      }
    else
      echo "node_modules binary not found"
    fi
  fi
  
  # Method 4: Use the fallback script as a last resort
  if [ ! -f ./static/dist/tailwind.css ] || [ "$(stat -c%s ./static/dist/tailwind.css 2>/dev/null || stat -f%z ./static/dist/tailwind.css)" -lt "1000" ]; then
    echo "All build methods failed, using fallback CSS script..."
    ./scripts/create_fallback_css.sh
  fi
  
  # Final verification
  if [ ! -f ./static/dist/tailwind.css ]; then
    echo "Error: Failed to create Tailwind CSS file using all methods. Creating an empty one..."
    touch ./static/dist/tailwind.css
  fi
fi

echo "Starting Pokemon Generator on port $PORT..."
echo "Access the application at: http://localhost:$PORT"
# Use gunicorn for production WSGI serving
exec gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 3 --access-logfile - --error-logfile - app:app 