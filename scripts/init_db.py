#!/usr/bin/env python3
"""
Database initialization script for Pokemon Generator.
This script creates all the required database tables.
"""
import sys
from dotenv import load_dotenv

load_dotenv()

def init_db():
    """Initialize the database with required tables."""
    try:
        from app import app, db
        
        with app.app_context():
            db.create_all()
            pass
            
            db_uri = app.config['SQLALCHEMY_DATABASE_URI']
            pass
            
            return True
    except Exception as e:
        pass
        return False

if __name__ == "__main__":
    success = init_db()
    sys.exit(0 if success else 1) 