#!/usr/bin/env python3
"""
Database migration script for the Pokemon Generator.
This script creates and updates database tables for production deployments.
"""
import os
import sys
import logging
import time
from dotenv import load_dotenv

# Ensure the parent directory is in sys.path for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
        logging.info("DATABASE_URL not set, constructing from parameters")
    
    # If Docker container, ensure we're using the correct host
    if os.getenv('DOCKER_CONTAINER') == 'true' and 'localhost' in db_uri:
        db_uri = db_uri.replace('localhost', 'db')
        logging.info("Running in Docker, using db service instead of localhost")

    # If using Render's postgres:// format, convert to postgresql://
    if db_uri.startswith("postgres://"):
        db_uri = db_uri.replace("postgres://", "postgresql://", 1)
        logging.info("Converted DATABASE_URL from postgres:// to postgresql://")

    # Mask the password in the URI for logging
    masked_uri = db_uri
    if '@' in db_uri:
        parts = db_uri.split('@')
        prefix_parts = parts[0].split(':')
        if len(prefix_parts) > 2:  # Contains username and password
            masked_uri = f"{prefix_parts[0]}:****@{parts[1]}"
    
    logging.info(f"Attempting to connect to database: {masked_uri}")
    retries = 0
    while retries < max_retries:
        try:
            engine = create_engine(db_uri)
            conn = engine.connect()
            conn.close()
            logging.info("Successfully connected to the database")
            return True
        except OperationalError as e:
            retries += 1
            logging.warning(f"Database connection attempt {retries}/{max_retries} failed: {e}")
            time.sleep(retry_interval)

    logging.error("Failed to connect to the database after multiple attempts")
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
        logging.error(f"Error checking if column exists: {e}")
        return False

def migrate_database():
    """Create or update database tables using SQLAlchemy models."""
    try:
        if not wait_for_database():
            logging.error("Could not connect to database, exiting migration")
            return False

        from app import app, db, GeneratedImage
        
        with app.app_context():
            db.create_all()
            logging.info("Successfully created/updated database tables")

            table_name = getattr(GeneratedImage, '__tablename__', 'generated_image').lower()
            
            if not check_table_column_exists(db, table_name, 'image_data'):
                logging.info(f"Adding image_data column to {table_name} table")
                from sqlalchemy import text
                
                db.session.execute(text(f"""
                ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS image_data BYTEA;
                ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS content_type VARCHAR(100) DEFAULT 'image/png';
                ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS type1 VARCHAR(50);
                ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS type2 VARCHAR(50);
                """))
                db.session.commit()
                logging.info(f"Added image storage columns to {table_name} table")

            from sqlalchemy import text
            result = db.session.execute(text("SELECT to_regclass('public.generation_stats');"))
            table_exists = result.scalar() is not None

            if not table_exists:
                logging.info("Creating generation_stats table")
                db.session.execute(text("""
                CREATE TABLE IF NOT EXISTS generation_stats (
                    id SERIAL PRIMARY KEY,
                    type1 VARCHAR(50) NOT NULL,
                    type2 VARCHAR(50),
                    legendary BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    generation INTEGER NOT NULL,
                    success BOOLEAN DEFAULT TRUE
                );
                CREATE INDEX IF NOT EXISTS idx_generation_stats_created_at ON generation_stats(created_at);
                """))
                db.session.commit()
                logging.info("Created generation_stats table")
            
            return True
    except Exception as e:
        logging.error(f"Error migrating database: {e}")
        return False

if __name__ == "__main__":
    success = migrate_database()
    sys.exit(0 if success else 1) 