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

# Create static directories if they don't exist
mkdir -p ./static/dist

# Check if Tailwind CSS file exists and has enough content
if [ ! -f ./static/dist/tailwind.css ] || [ "$(stat -c%s ./static/dist/tailwind.css 2>/dev/null || stat -f%z ./static/dist/tailwind.css)" -lt "10000" ]; then
  echo "Tailwind CSS file missing or too small, rebuilding..."
  
  # Try multiple ways to rebuild the CSS
  if command -v npx > /dev/null; then
    echo "Rebuilding with npx tailwindcss..."
    if npx tailwindcss -i ./static/src/tailwind.css -o ./static/dist/tailwind.css --minify; then
      echo "Successfully built Tailwind CSS with npx."
    else
      echo "Failed with npx, trying npm run build:css..."
      if npm run build:css 2>/dev/null; then
        echo "Successfully built CSS with npm run build:css."
      else
        echo "All build attempts failed, using fallback CSS..."
        ./scripts/create_fallback_css.sh
      fi
    fi
  else
    echo "npx not available, using fallback CSS..."
    ./scripts/create_fallback_css.sh
  fi
  
  # Final verification of CSS file
  if [ ! -f ./static/dist/tailwind.css ] || [ "$(stat -c%s ./static/dist/tailwind.css 2>/dev/null || stat -f%z ./static/dist/tailwind.css)" -lt "1000" ]; then
    echo "CSS still missing or too small, creating emergency inline CSS..."
    echo "/* Emergency fallback CSS */" > ./static/dist/tailwind.css
    cat >> ./static/dist/tailwind.css << 'EOL'
body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;margin:0;padding:0}.bg-red-700{background-color:#b91c1c}.text-white{color:#fff}.flex{display:flex}.flex-col{flex-direction:column}.items-center{align-items:center}.justify-center{justify-content:center}.mx-auto{margin-left:auto;margin-right:auto}.my-4{margin-top:1rem;margin-bottom:1rem}.text-center{text-align:center}.font-bold{font-weight:700}.text-2xl{font-size:1.5rem;line-height:2rem}.rounded{border-radius:0.25rem}.p-4{padding:1rem}.shadow{box-shadow:0 1px 3px 0 rgba(0,0,0,0.1),0 1px 2px 0 rgba(0,0,0,0.06)}.w-full{width:100%}
EOL
  fi
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
