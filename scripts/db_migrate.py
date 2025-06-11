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

    db_uri = os.getenv('DATABASE_URL')
    if not db_uri:
        print("[ERROR] DATABASE_URL environment variable is not set. Exiting.")
        return False
    
    # If using Render's postgres:// format, convert to postgresql://
    if db_uri.startswith("postgres://"):
        db_uri = db_uri.replace("postgres://", "postgresql://", 1)
        print("[INFO] Converted DATABASE_URL from postgres:// to postgresql://")

    # Mask the password in the URI for printing
    masked_uri = db_uri
    if '@' in db_uri:
        parts = db_uri.split('@')
        prefix_parts = parts[0].split(':')
        if len(prefix_parts) > 2:  # Contains username and password
            masked_uri = f"{prefix_parts[0]}:****@{parts[1]}"

    print(f"[INFO] Attempting to connect to database: {masked_uri}")
    retries = 0
    while retries < max_retries:
        try:
            engine = create_engine(db_uri)
            conn = engine.connect()
            conn.close()
            print("[INFO] Successfully connected to the database")
            return True
        except OperationalError as e:
            retries += 1
            print(f"[WARN] Database connection attempt {retries}/{max_retries} failed: {e}")
            time.sleep(retry_interval)

    print("[ERROR] Failed to connect to the database after multiple attempts")
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
        print("[INFO] Starting database migration...")
        if not wait_for_database():
            print("[ERROR] Could not connect to database, exiting migration")
            return False

        from app import app, db, GeneratedImage
        print("[INFO] Connected to app and database, creating tables...")
        with app.app_context():
            db.create_all()
            print("[INFO] Successfully created/updated database tables")

            table_name = getattr(GeneratedImage, '__tablename__', 'generated_image').lower()
            if not check_table_column_exists(db, table_name, 'image_data'):
                print(f"[INFO] Adding image_data and related columns to {table_name} table")
                from sqlalchemy import text
                db.session.execute(text(f"""
                ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS image_data BYTEA;
                ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS content_type VARCHAR(100) DEFAULT 'image/png';
                ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS type1 VARCHAR(50);
                ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS type2 VARCHAR(50);
                """))
                db.session.commit()
                print(f"[INFO] Added image storage columns to {table_name} table")

            from sqlalchemy import text
            result = db.session.execute(text("SELECT to_regclass('public.generation_stats');"))
            table_exists = result.scalar() is not None

            if not table_exists:
                print("[INFO] Creating generation_stats table")
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
                print("[INFO] Created generation_stats table")
            print("[INFO] Database migration completed successfully.")
            return True
    except Exception as e:
        print(f"[ERROR] Error migrating database: {e}")
        return False

if __name__ == "__main__":
    success = migrate_database()
    sys.exit(0 if success else 1) 