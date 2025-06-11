#!/bin/sh
set -e

until pg_isready -U "$POSTGRES_USER"; do
    echo "Waiting for PostgreSQL to be ready..."
    sleep 1
done

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "postgres" <<EOSQL
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = '$POSTGRES_DB') THEN
        CREATE DATABASE "$POSTGRES_DB";
    END IF;
END
$$;
EOSQL

export OWNER="$POSTGRES_USER"
export DBNAME="$POSTGRES_DB"
if [ -f /docker-entrypoint-initdb.d/01-init.sql.template ]; then
  if ! envsubst < /docker-entrypoint-initdb.d/01-init.sql.template > /docker-entrypoint-initdb.d/01-init.sql; then
    echo "envsubst failed" >&2
    exit 1
  fi
fi

if ! psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" \  
     --dbname "$POSTGRES_DB" \
     -f /docker-entrypoint-initdb.d/01-init.sql; then
  echo "psql failed to execute 01-init.sql" >&2
  exit 1
fi

echo "Database initialization complete."