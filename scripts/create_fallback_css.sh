#!/bin/sh
# This script creates a minimal Tailwind CSS file as a fallback
# This is used if all build methods fail

echo "Creating fallback Tailwind CSS file..."

DIST_DIR="./static/dist"
FALLBACK_FILE="${DIST_DIR}/tailwind.css"

# Create the directory if it doesn't exist
mkdir -p "${DIST_DIR}"

# Check if the file already exists and has content
if [ -f "${FALLBACK_FILE}" ] && [ "$(stat -c%s "${FALLBACK_FILE}" 2>/dev/null || stat -f%z "${FALLBACK_FILE}")" -gt "10000" ]; then
  echo "Tailwind CSS file already exists and appears to have sufficient content. Keeping existing file."
  exit 0
fi

# Try multiple build methods in sequence, stopping at first success
echo "Attempting to build Tailwind using various methods..."

# Method 1: npm run build:css
if command -v npm > /dev/null; then
  echo "Attempting to build with npm run build:css..."
  if npm run build:css 2>/dev/null; then
    echo "Successfully built Tailwind CSS with npm run build:css."
    if [ -f "${FALLBACK_FILE}" ] && [ "$(stat -c%s "${FALLBACK_FILE}" 2>/dev/null || stat -f%z "${FALLBACK_FILE}")" -gt "1000" ]; then
      exit 0
    fi
  fi
fi

# Method 2: npx tailwindcss
if command -v npx > /dev/null; then
  echo "Attempting to build with npx tailwindcss..."
  if npx tailwindcss -i ./static/src/tailwind.css -o "${FALLBACK_FILE}" --minify 2>/dev/null; then
    echo "Successfully built Tailwind CSS with npx tailwindcss."
    exit 0
  fi
fi

# Method 3: direct node execution
if command -v node > /dev/null; then
  echo "Attempting to build with Node.js directly..."
  if node -e "try { require('tailwindcss'); require('postcss'); require('autoprefixer'); const fs=require('fs'); console.log('Dependencies loaded'); const config=require('./tailwind.config.js'); const postcss=require('postcss'); postcss([require('tailwindcss')(config), require('autoprefixer')]).process('@tailwind base; @tailwind components; @tailwind utilities;', {from:undefined}).then(result => fs.writeFileSync('${FALLBACK_FILE}', result.css)); } catch(e) { console.error(e); process.exit(1); }"; then
    echo "Successfully built minimal Tailwind CSS with Node.js."
    exit 0
  fi
fi

echo "All build methods failed. Using predefined minimal CSS..."

# Create a minimal Tailwind CSS file
cat > "${FALLBACK_FILE}" << 'EOL'
/* Fallback Tailwind CSS */
*,:after,:before{box-sizing:border-box;border:0 solid #e5e7eb}:after,:before{--tw-content:""}html{line-height:1.5;-webkit-text-size-adjust:100%;-moz-tab-size:4;-o-tab-size:4;tab-size:4;font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,Noto Sans,sans-serif,Apple Color Emoji,Segoe UI Emoji,Segoe UI Symbol,Noto Color Emoji;font-feature-settings:normal;font-variation-settings:normal}body{margin:0;line-height:inherit}hr{height:0;color:inherit;border-top-width:1px}abbr:where([title]){-webkit-text-decoration:underline dotted;text-decoration:underline dotted}h1,h2,h3,h4,h5,h6{font-size:inherit;font-weight:inherit}a{color:inherit;text-decoration:inherit}b,strong{font-weight:bolder}code,kbd,pre,samp{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,Liberation Mono,Courier New,monospace;font-size:1em}small{font-size:80%}sub,sup{font-size:75%;line-height:0;position:relative;vertical-align:initial}sub{bottom:-.25em}sup{top:-.5em}table{text-indent:0;border-color:inherit;border-collapse:collapse}button,input,optgroup,select,textarea{font-family:inherit;font-feature-settings:inherit;font-variation-settings:inherit;font-size:100%;font-weight:inherit;line-height:inherit;color:inherit;margin:0;padding:0}button,select{text-transform:none}[type=button],[type=reset],[type=submit],button{-webkit-appearance:button;background-color:initial;background-image:none}:-moz-focusring{outline:auto}:-moz-ui-invalid{box-shadow:none}progress{vertical-align:initial}::-webkit-inner-spin-button,::-webkit-outer-spin-button{height:auto}[type=search]{-webkit-appearance:textfield;outline-offset:-2px}::-webkit-search-decoration{-webkit-appearance:none}::-webkit-file-upload-button{-webkit-appearance:button;font:inherit}summary{display:list-item}blockquote,dd,dl,figure,h1,h2,h3,h4,h5,h6,hr,p,pre{margin:0}fieldset{margin:0}fieldset,legend{padding:0}menu,ol,ul{list-style:none;margin:0;padding:0}dialog{padding:0}textarea{resize:vertical}input::-moz-placeholder,textarea::-moz-placeholder{opacity:1;color:#9ca3af}input::placeholder,textarea::placeholder{opacity:1;color:#9ca3af}[role=button],button{cursor:pointer}:disabled{cursor:default}audio,canvas,embed,iframe,img,object,svg,video{display:block;vertical-align:middle}img,video{max-width:100%;height:auto}[hidden]{display:none}

.flex{display:flex}.flex-col{flex-direction:column}.items-center{align-items:center}.justify-center{justify-content:center}
.mx-auto{margin-left:auto;margin-right:auto}.my-4{margin-top:1rem;margin-bottom:1rem}
.mt-2{margin-top:0.5rem}.mb-4{margin-bottom:1rem}
.p-4{padding:1rem}.px-4{padding-left:1rem;padding-right:1rem}.py-2{padding-top:0.5rem;padding-bottom:0.5rem}
.text-center{text-align:center}
.text-2xl{font-size:1.5rem;line-height:2rem}.text-lg{font-size:1.125rem;line-height:1.75rem}
.font-bold{font-weight:700}
.bg-blue-500{--tw-bg-opacity:1;background-color:rgb(59 130 246/var(--tw-bg-opacity))}
.text-white{--tw-text-opacity:1;color:rgb(255 255 255/var(--tw-text-opacity))}
.rounded{border-radius:0.25rem}.rounded-lg{border-radius:0.5rem}
.shadow{--tw-shadow:0 1px 3px 0 rgba(0,0,0,0.1),0 1px 2px 0 rgba(0,0,0,0.06);box-shadow:var(--tw-ring-offset-shadow,0 0 #0000),var(--tw-ring-shadow,0 0 #0000),var(--tw-shadow)}
.w-full{width:100%}.max-w-4xl{max-width:56rem}
.hover\:bg-blue-700:hover{--tw-bg-opacity:1;background-color:rgb(29 78 216/var(--tw-bg-opacity))}
EOL

echo "Fallback Tailwind CSS file created." 