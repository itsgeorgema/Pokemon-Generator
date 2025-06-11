#!/bin/sh
cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

echo "Pokemon Generator Deployment to Render"
echo "------------------------------------"
echo "Working directory: $ROOT_DIR"
echo ""

if ! command -v render &> /dev/null; then
    echo "Error: Render CLI is not installed."
    echo "Please install it first: npm install -g @render/cli"
    exit 1
fi

render whoami &> /dev/null
if [ $? -ne 0 ]; then
    echo "You need to log in to Render first."
    echo "Run: render login"
    exit 1
fi

echo "Running tests before deployment..."
./scripts/run_tests.sh

if [ $? -ne 0 ]; then
    echo "Tests failed. Do you want to continue with deployment? (y/n)"
    read -r answer
    if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
        echo "Deployment cancelled."
        exit 1
    fi
fi

echo "Deploying to Render..."
render deploy --yaml render.yaml

echo ""
echo "Deployment initiated! Check the Render dashboard for progress."
echo "Visit: https://dashboard.render.com" 