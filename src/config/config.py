#!/usr/bin/env python3
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration."""
    REQUIRED_ENV_VARS = [
        'SECRET_KEY',
        'CHECKPOINT_PATH',
        'POKEMON_DATA_PATH',
        'APP_VERSION',
        'MODEL_VERSION',
        'DATABASE_URL',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD',
        'POSTGRES_DB'
    ]

    @classmethod
    def validate_env_vars(cls):
        """Validate that all required environment variables are set."""
        missing_vars = []
        for var in cls.REQUIRED_ENV_VARS:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    SECRET_KEY = os.getenv('SECRET_KEY')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CHECKPOINT_PATH = os.getenv('CHECKPOINT_PATH')
    POKEMON_DATA_PATH = os.getenv('POKEMON_DATA_PATH')
    APP_VERSION = os.getenv('APP_VERSION')
    MODEL_VERSION = os.getenv('MODEL_VERSION')
    
    DATABASE_URL = os.getenv('DATABASE_URL')
    POSTGRES_USER = os.getenv('POSTGRES_USER')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
    POSTGRES_DB = os.getenv('POSTGRES_DB')
    
    if DATABASE_URL and os.getenv('DOCKER_CONTAINER') == 'true' and 'localhost' in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.replace('localhost', 'db')

class DevelopmentConfig(Config):
    """Development configuration."""
    FLASK_ENV = 'development'
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    if os.getenv('DOCKER_CONTAINER') == 'true' and SQLALCHEMY_DATABASE_URI and 'localhost' in SQLALCHEMY_DATABASE_URI:
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace('localhost', 'db')

class ProductionConfig(Config):
    """Production configuration."""
    FLASK_ENV = 'production'
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL')
    if os.getenv('DOCKER_CONTAINER') == 'true' and SQLALCHEMY_DATABASE_URI and 'localhost' in SQLALCHEMY_DATABASE_URI:
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace('localhost', 'db')
    # Handle Render's postgres:// prefix (convert to postgresql://)
    if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith("postgres://"):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql://", 1)

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 