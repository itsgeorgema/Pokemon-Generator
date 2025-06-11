# Scripts Directory

This directory contains utility scripts for the project.

## Available Scripts

### `start_server.sh`

Server startup script used by Docker. It:
- Checks for PostgreSQL connection
- Creates necessary directories
- Initializes the database
- Starts the Gunicorn server

### `deploy_to_render.sh`

Script to deploy the application to Render:

```bash
./scripts/deploy_to_render.sh
```