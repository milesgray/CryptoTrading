import logging
from typing import List
import datetime as dt

from .exchange import ExchangePriceClient
from .server import PriceServerClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PriceClientStatus:
    log_levels = {
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "DEBUG": logging.DEBUG
    }
    def __init__(self):
        self.running = False
        self.paused = False
        self.connected = False
        self.start_time = None
        self.last_error = None
        self.logs: dict[str, List[dict[str, any]]] = {}
        self.max_logs = 1000  # Keep last 1000 log entries

    def add_log(self, level: str, message: str):
        """Add a log entry to the service status"""
        try:
            logger.log(self.log_levels[level], message)
        except Exception as e:
            logger.error(f"Error adding log: {e}")
        self.logs[level].append({
            'timestamp': dt.datetime.now(dt.timezone.utc).isoformat(),
            'level': level,
            'message': message
        })
        # Keep only the last max_logs entries
        if len(self.logs[level]) > self.max_logs:
            self.logs[level] = self.logs[level][-self.max_logs:]

__all__ = [
    'ExchangePriceClient',
    'PriceServerClient',
    'PriceClientStatus',
]