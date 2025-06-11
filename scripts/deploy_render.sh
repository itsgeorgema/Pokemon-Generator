#!/bin/sh
set -e

if ! command -v render &> /dev/null; then
    echo "render-cli is not installed. Please install it first."
    echo "npm install -g @renderinc/cli"
    exit 1
fi

if [ -f .env ]; then
    source .env
fi

if [ -z "$RENDER_API_KEY" ]; then
    echo "RENDER_API_KEY environment variable is not set."
    echo "Please set it in your .env file or export it before running this script."
    exit 1
fi

echo "Deploying to Render..."
render blueprint launch \
    --env APP_VERSION=${APP_VERSION:-1.2.0} \
    --env MODEL_VERSION=${MODEL_VERSION:-2.0.0} \
    --env FLASK_ENV=${FLASK_ENV:-production} \
    --env SECRET_KEY=${SECRET_KEY:-$(openssl rand -hex 32)} \
    --env CHECKPOINT_PATH=${CHECKPOINT_PATH:-models/checkpoint.pth} \
    --env POKEMON_DATA_PATH=${POKEMON_DATA_PATH:-data/Pokemon_stats.csv}

echo "Deployment initiated. Check the Render dashboard for status." 