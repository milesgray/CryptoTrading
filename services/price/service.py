import os
import asyncio
import logging
import signal
import time
import datetime as dt
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cryptotrading.rollbit.prices.price import PriceSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('price_system')

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}

REFRESH_INTERVAL_MS = int(os.getenv("REFRESH_INTERVAL_MS", 500))

class ServiceStatus:
    def __init__(self):
        self.running = False
        self.paused = False
        self.start_time = None
        self.last_error = None
        self.logs: List[Dict[str, Any]] = []
        self.max_logs = 1000  # Keep last 1000 log entries

    def _parse_level(self, level: str) -> int:
        level = level.lower() if isinstance(level, str) else level
        level = LOG_LEVELS[level] if level in LOG_LEVELS else level
        level = level if isinstance(level, int) else logging.INFO
        return level

    def add_log(self, level: str, message: str):
        """Add a log entry to the service status"""
        level = self._parse_level(level)
        logger.log(level, message)
        self.logs.append({
            'timestamp': datetime.now(dt.timezone.utc).isoformat(),
            'level': level,
            'message': message
        })
        if level >= logging.WARNING:
            self.last_error = message
        # Keep only the last max_logs entries
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]

class PriceSystemService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PriceSystemService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.price_system = PriceSystem()
        self.status = ServiceStatus()
        self.status.running = True
        self.status.start_time = time.time()
        self.status.add_log('INFO', 'Service initialized')

        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
    async def run(self):
        """Main function to run the price system"""
        try:
            await self.price_system.initialize()
            self.status.add_log('INFO', 'Price system initialized')
            while self.status.running:
                if not self.status.paused:
                    start_time = time.time()
                    try:
                        await self.price_system.run()
                    except Exception as e:
                        self.status.add_log('ERROR', str(e))
                    finally:
                        elapsed = (time.time() - start_time) * 1000
                        sleep_time = max(0, REFRESH_INTERVAL_MS - elapsed) / 1000
                        if sleep_time > 0:
                            await asyncio.sleep(sleep_time)
                else:
                    await asyncio.sleep(1)  # Sleep when paused
        except Exception as e:
            self.status.add_log('ERROR', str(e))
            return 1
        finally:
            await self.price_system.shutdown()
            self.status.add_log('INFO', 'Price system shutdown complete')
            
        return 0

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.status.add_log('INFO', f'Received signal {signum}, initiating graceful shutdown...')
        self.status.running = False

    def pause(self):
        """Pause the service"""
        if not self.status.paused:
            self.status.paused = True
            self.status.add_log('INFO', 'Service paused')
            return {"status": "paused", "message": "Service has been paused"}
        return {"status": "already_paused", "message": "Service is already paused"}

    def resume(self):
        """Resume the service"""
        if self.status.paused:
            self.status.paused = False
            self.status.add_log('INFO', 'Service resumed')
            return {"status": "resumed", "message": "Service has been resumed"}
        return {"status": "not_paused", "message": "Service is not paused"}

    def stop(self):
        """Stop the service"""
        if self.status.running:
            self.status.running = False
            self.status.add_log('INFO', 'Service stop requested')
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
        }

    def get_price_system_status(self):
        """Get current price system status"""
        if not self.price_system.running:
            return {
                "running": False,
                "index_prices": {},
            }
        return {
            "running": self.price_system.running,
            "index_prices": {symbol:{"price": price, "time": time} for ((symbol, price), (symbol, time)) in 
            zip(self.price_system.last_index_prices.items(), self.price_system.last_price_times.items())},
        }

    def get_logs(self, limit: int = 100, level: Optional[str] = None):
        """Get service logs with optional filtering"""
        logs = self.status.logs
        if level:
            logs = [log for log in logs if log['level'].upper() == level.upper()]
        return logs[-limit:]

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Trading Service",
    description="API for controlling and monitoring the Crypto Trading Service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service instance
service = PriceSystemService()

# API Models
class StatusResponse(BaseModel):
    running: bool
    paused: bool
    uptime_seconds: float
    last_error: Optional[str]
    logs_count: int

class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str

# API Endpoints
@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current service status"""
    return service.get_status()

@app.get("/logs", response_model=List[LogEntry])
async def get_logs(limit: int = 100, level: Optional[str] = None):
    """Get service logs"""
    if level and level.upper() not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise HTTPException(status_code=400, detail="Invalid log level")
    return service.get_logs(limit, level)

@app.post("/pause")
async def pause_service():
    """Pause the service"""
    return service.pause()

@app.post("/resume")
async def resume_service():
    """Resume the service"""
    return service.resume()

@app.post("/stop")
async def stop_service(background_tasks: BackgroundTasks):
    """Stop the service"""
    result = service.stop()
    if result["status"] == "stopping":
        # Schedule the actual shutdown after the response is sent
        background_tasks.add_task(lambda: asyncio.get_event_loop().stop())
    return result

async def run_service():
    """Run the price system service"""
    await service.run()

def start_service():
    """Start the service with FastAPI"""
    import uvicorn
    import threading
    
    # Start the FastAPI server in a separate thread
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8300)
    
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    # Run the main service in the main thread
    asyncio.run(run_service())

def main():
    """Main entry point"""
    start_service()

if __name__ == "__main__":
    main()