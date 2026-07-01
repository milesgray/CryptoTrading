import os
import asyncio
import logging
import signal
import time
import datetime as dt    
import threading
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cryptotrading.rollbit.prices.price import PriceSystem
from cryptotrading.util.status import StatusManager
from cryptotrading.config import SYMBOLS, REFRESH_INTERVAL_MS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class Service:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Service, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.status = StatusManager('price_system_service')
        self.status.running = True
        self.status.start_time = time.time()
        self.status.info('Service initialized')

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    async def run(self):
        """Main function to run the price system"""
        try:
            await self.price_system.initialize()
            self.status.info('Price system initialized')
            while self.status.running:
                if not self.status.paused:
                    start_time = time.time()
                    try:
                        await self.price_system.run()
                    except Exception as e:
                        error_msg = f"Unexpected error: {str(e)}"
                        self.status.last_error = error_msg
                        self.status.error(error_msg)
                    finally:
                        elapsed = (time.time() - start_time) * 1000
                        sleep_time = max(0, REFRESH_INTERVAL_MS - elapsed) / 1000
                        if sleep_time > 0:
                            await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(1)  # Sleep when paused
        except Exception as e:
            self.status.error(f"Unexpected error: {str(e)}")
            return 1
        finally:
            await self.price_system.shutdown()
            self.status.info("Price system shutdown complete")
            
        return 0

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.status.info(f'Received signal {signum}, initiating graceful shutdown...')
        self.status.running = False

    def pause(self):
        """Pause the service"""
        if not self.status.paused:
            self.status.paused = True
            self.status.info('Service paused')
            return {"status": "paused", "message": "Service has been paused"}
        return {"status": "already_paused", "message": "Service is already paused"}

    def resume(self):
        """Resume the service"""
        if self.status.paused:
            self.status.paused = False
            self.status.info('Service resumed')
            return {"status": "resumed", "message": "Service has been resumed"}
        return {"status": "not_paused", "message": "Service is not paused"}

    def stop(self):
        """Stop the service"""
        if self.status.running:
            self.status.running = False
            self.status.info('Service stop requested')
            return {"status": "stopping", "message": "Service is stopping"}
        return {"status": "not_running", "message": "Service is not running"}

    def get_status(self):
        """Get current service status"""
        return {
            "running": self.status.running,
            "paused": self.status.paused,
            "uptime_seconds": time.time() - (self.status.start_time or 0),
            "last_error": self.status.last_error,
            "logs_count": len(self.status.logs),
            "data": self.status.get_data(),
        }

    def get_logs(self, limit: int = 100, level: Optional[str] = None):
        """Get service logs with optional filtering"""
        logs = self.status.logs
        if level:
            logs = [log for log in logs if log['level'].upper() == level.upper()]
        return logs[-limit:]
