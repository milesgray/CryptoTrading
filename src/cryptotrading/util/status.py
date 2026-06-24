import logging
import datetime as dt    
from typing import Dict, Any
from collections import deque
import threading

LOG_LEVELS = {
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'DEBUG': logging.DEBUG,
    'CRITICAL': logging.CRITICAL
}

class StatusManager:
    def __init__(self, name: str, verbose: bool = False, max_logs: int = 1000):
        self.name = name
        self.verbose = verbose
        self.logger = logging.getLogger(name)
        self.running = False
        self.paused = False
        self.start_time = None
        self.last_error = None
        self.max_logs = max_logs
        
        # Bounded deque for thread-safe O(1) appends and rotations
        self._logs = deque(maxlen=max_logs)
        
        # Lock for logical thread-safety of data updates and log copying
        self._lock = threading.Lock()
        self.data: Dict[str, Any] = {}

    @property
    def logs(self) -> list:
        """Expose log buffer as a list copy to support slicing and thread-safety"""
        with self._lock:
            return list(self._logs)

    def update_data(self, data: Dict[str, Any]) -> None:
        """Add or update a data entry thread-safely"""
        with self._lock:
            self.data.update(data)

    def get_data(self, key: str | None = None) -> Any:
        """Get a data entry thread-safely"""
        with self._lock:
            if key:
                return self.data.get(key)
            return self.data.copy()

    def add_log(self, level: str, message: str) -> None:
        """Log a message and append it to the in-memory buffer"""
        self.logger.log(LOG_LEVELS.get(level, logging.INFO), message)
        
        log_entry = {
            'timestamp': dt.datetime.now(dt.timezone.utc).isoformat(),
            'level': level,
            'message': message
        }
        with self._lock:
            self._logs.append(log_entry)

    def info(self, message: str) -> None:
        self.add_log('INFO', message)

    def warning(self, message: str) -> None:
        self.add_log('WARNING', message)

    def error(self, message: str) -> None:
        self.add_log('ERROR', message)

    def debug(self, message: str) -> None:
        self.add_log('DEBUG', message)

    def critical(self, message: str) -> None:
        self.add_log('CRITICAL', message)