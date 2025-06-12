#!/bin/bash
set -e

echo "Setting up environment for Render deployment..."

# Install Node.js and npm
echo "Installing Node.js and npm..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get update
apt-get install -y nodejs build-essential

# Verify installations
echo "Node.js version:"
node --version
echo "npm version:"
npm --version

# Install project dependencies
echo "Installing npm dependencies..."
npm install

# Build Tailwind CSS
echo "Building Tailwind CSS..."
npx tailwindcss -i ./static/src/tailwind.css -o ./static/dist/tailwind.css --minify

echo "Setup complete!" 