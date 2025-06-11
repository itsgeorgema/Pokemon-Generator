# PostgreSQL Configuration

This directory contains configuration files and initialization scripts for the PostgreSQL database used by the Pokemon Generator application.

## Directory Structure

- `init/` - Contains initialization scripts that are run when the PostgreSQL container starts for the first time
  - `01-init.sql` - Creates extensions, tables, and functions needed by the application

## Running with Docker

When running with Docker Compose, these scripts will be automatically executed when the PostgreSQL container starts for the first time. The scripts are mounted at `/docker-entrypoint-initdb.d/` in the container.

## Local Development

For local development without Docker, you can run the initialization scripts manually:

```bash
psql -U postgres -d pokemon_dev -f postgres/init/01-init.sql
```

## Production Deployment

For production deployment on Render, the database initialization is handled by the `scripts/db_migrate.py` script, which is called by the `scripts/start_server.sh` script when the application starts.

## Database Schema

The schema is defined in `init/01-init.sql.template` and is rendered using environment variables.