#!/usr/bin/env node

// Simple script to build Tailwind CSS without needing npx
// This provides an alternative method if npx fails on Render

const fs = require('fs');
const path = require('path');

try {
  console.log('Building Tailwind CSS...');
  
  // Check if tailwindcss is installed
  const tailwindPath = path.resolve('./node_modules/.bin/tailwindcss');
  
  if (!fs.existsSync(tailwindPath)) {
    console.error('Tailwind CSS binary not found. Trying to use require...');
    
    try {
      // Alternative approach using the module directly
      const tailwind = require('tailwindcss');
      const postcss = require('postcss');
      const autoprefixer = require('autoprefixer');
      
      // Read the input file
      const css = fs.readFileSync('./static/src/tailwind.css', 'utf8');
      
      // Get the config
      const config = require('../tailwind.config.js');
      
      // Process the CSS
      postcss([
        tailwind(config),
        autoprefixer
      ])
        .process(css, { from: './static/src/tailwind.css', to: './static/dist/tailwind.css' })
        .then(result => {
          fs.writeFileSync('./static/dist/tailwind.css', result.css);
          console.log('Tailwind CSS built successfully!');
        })
        .catch(err => {
          console.error('Error building Tailwind CSS:', err);
          process.exit(1);
        });
    } catch (innerError) {
      console.error('Error requiring Tailwind modules:', innerError);
      process.exit(1);
    }
  } else {
    // Execute the tailwindcss binary directly
    const { execSync } = require('child_process');
    execSync(`${tailwindPath} -i ./static/src/tailwind.css -o ./static/dist/tailwind.css --minify`);
    console.log('Tailwind CSS built successfully using binary!');
  }
} catch (error) {
  console.error('Error building Tailwind CSS:', error);
  process.exit(1);
} 