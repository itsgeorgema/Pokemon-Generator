#!/usr/bin/env python3
"""
Database migration script for the Pokemon Generator.
This script creates and updates database tables for production deployments.
"""
import os
import sys
from dotenv import load_dotenv
import time

# Ensure the parent directory is in sys.path for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

def wait_for_database(max_retries=10, retry_interval=5):
    """Wait for the database to be ready."""
    from sqlalchemy import create_engine
    from sqlalchemy.exc import OperationalError

    # Get database connection parameters
    postgres_user = os.getenv('POSTGRES_USER')
    postgres_password = os.getenv('POSTGRES_PASSWORD')
    postgres_db = os.getenv('POSTGRES_DB')
    postgres_host = os.getenv('POSTGRES_HOST', 'db' if os.getenv('DOCKER_CONTAINER') == 'true' else 'localhost')
    postgres_port = os.getenv('POSTGRES_PORT', '5432')
    
    # Get database URL from environment variable or construct it
    db_uri = os.getenv('DATABASE_URL')
    if not db_uri:
        db_uri = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
        pass
    
    # If Docker container, ensure we're using the correct host
    if os.getenv('DOCKER_CONTAINER') == 'true' and 'localhost' in db_uri:
        db_uri = db_uri.replace('localhost', 'db')
        pass

    # If using Render's postgres:// format, convert to postgresql://
    if db_uri.startswith("postgres://"):
        db_uri = db_uri.replace("postgres://", "postgresql://", 1)
        pass

    # Mask the password in the URI for logging
    masked_uri = db_uri
    if '@' in db_uri:
        parts = db_uri.split('@')
        prefix_parts = parts[0].split(':')
        if len(prefix_parts) > 2:  # Contains username and password
            masked_uri = f"{prefix_parts[0]}:****@{parts[1]}"
    
    pass
    retries = 0
    while retries < max_retries:
        try:
            engine = create_engine(db_uri)
            conn = engine.connect()
            conn.close()
            pass
            return True
        except OperationalError as e:
            retries += 1
            pass
            time.sleep(retry_interval)

    pass
    return False

def check_table_column_exists(db, table_name, column_name):
    """Check if a column exists in a table."""
    from sqlalchemy import text
    
    try:
        query = text(f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}' 
        AND column_name = '{column_name}'
        """)
        
        result = db.session.execute(query)
        return result.fetchone() is not None
    except Exception as e:
        pass
        return False

def migrate_database():
    """Create or update database tables using SQLAlchemy models."""
    try:
        if not wait_for_database():
            pass
            return False

        from app import app, db, GeneratedImage
        
        with app.app_context():
            db.create_all()
            pass
            return True
    except Exception as e:
        pass
        return False

if __name__ == "__main__":
    success = migrate_database()
    sys.exit(0 if success else 1) 