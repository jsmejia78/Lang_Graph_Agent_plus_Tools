# Frontend

This is a manually set up React app for the AI Chat frontend.

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```
2. Run the development server:
   ```bash
   npm run dev
   ```
3. Build for production:
   ```bash
   npm run build
   ```
4. Preview the production build:
   ```bash
   npm run preview
   ```

## Deployment

This app is ready for deployment on Vercel. The default output is compatible with Vercel's static hosting.

## Installing nvm

1. Install nvm if you don't have it:
   ```bash
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
   ```
2. Activate nvm (or restart your shell):
   ```bash
   export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
   [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
   ```
3. Install a stable Node.js version (e.g., 20):
   ```bash
   nvm install 20
   ```
4. Use it:
   ```bash
   nvm use 20
   ```
5. Set as default:
   ```bash
   nvm alias default 20
   ```