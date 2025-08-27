#!/usr/bin/env python3
"""
Health Check Script for Docker Container

This script is used by Docker's HEALTHCHECK to verify the service is running properly.
"""

import sys
import requests
import os

def health_check():
    """Perform health check by calling the health endpoint"""
    try:
        health_port = int(os.getenv('HEALTH_PORT', '8080'))
        response = requests.get(f'http://localhost:{health_port}/health', timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            
            # Check if service status is healthy
            if health_data.get('status') in ['healthy', 'starting']:
                print("Health check passed")
                return 0
            else:
                print(f"Service status: {health_data.get('status')}")
                return 1
        else:
            print(f"Health endpoint returned status code: {response.status_code}")
            return 1
            
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error during health check: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(health_check())