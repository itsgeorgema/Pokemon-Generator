#!/usr/bin/env python3
"""
Database initialization script for Pokemon Generator.
This script creates all the required database tables.
"""
import sys
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

load_dotenv()

def init_db():
    """Initialize the database with required tables."""
    try:
        from app import app, db
        
        with app.app_context():
            db.create_all()
            logging.info("Successfully created database tables")
            
            db_uri = app.config['SQLALCHEMY_DATABASE_URI']
            if db_uri and ':' in db_uri and '@' in db_uri:
                parts = db_uri.split('@')
                prefix = parts[0].split(':')
                masked_uri = f"{prefix[0]}:****@{parts[1]}"
                logging.info(f"Using database: {masked_uri}")
            else:
                logging.info(f"Using database: {db_uri}")
            
            return True
    except Exception as e:
        logging.error(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    success = init_db()
    sys.exit(0 if success else 1) 