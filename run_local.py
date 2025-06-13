#!/usr/bin/env python3
"""
Simplified script for running the Pokemon Generator locally without PostgreSQL.
This uses SQLite instead for easier local development.
"""
import os
import sys
import socket
from dotenv import load_dotenv

# Load any environment variables from .env
load_dotenv()

def get_ip_address():
    """Get the primary IP address of the machine"""
    try:
        # Get the primary IP address (works in most environments)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

if __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    port = int(os.environ['PORT'])
    ip = get_ip_address()
    print("\n" + "=" * 60)
    print(f"Pokemon Generator (LOCAL MODE) is running!")
    print(f"Local URL:     \033[94mhttp://localhost:{port}/\033[0m")
    print(f"Network URL:   \033[94mhttp://{ip}:{port}/\033[0m")
    print("=" * 60)
    print("Using environment-provided database for local development")
    print("=" * 60 + "\n")
    
    from app import app
    app.run(host='0.0.0.0', port=port, debug=True) 