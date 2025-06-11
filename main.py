#!/usr/bin/env python3
"""
Main entry point for the Pokemon Generator application.
This script imports and runs the Flask app.
"""
import os
from dotenv import load_dotenv
import socket

load_dotenv()

from app import app

def get_ip_address():
    """Get the primary IP address of the machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

if __name__ == '__main__':
    port = int(os.environ['PORT'])
    host = os.environ['HOST']
    debug = os.environ['FLASK_DEBUG'].lower() in ('true', '1', 't')
    ip = get_ip_address()
    print("\n" + "=" * 60)
    print(f"Pokemon Generator is running!")
    print(f"Local URL:     \033[94mhttp://localhost:{port}/\033[0m")
    print(f"Network URL:   \033[94mhttp://{ip}:{port}/\033[0m")
    if os.environ.get('DOCKER_CONTAINER', '') == 'true':
        print(f"Docker URL:    \033[94mhttp://localhost:{port}/\033[0m (if using port mapping)")
    print("=" * 60 + "\n")
    app.run(host=host, port=port, debug=debug) 