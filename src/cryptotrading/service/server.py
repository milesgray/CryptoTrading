import logging
from typing import List, Callable, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("service_server")

class ServiceServer:
    """
    Reusable FastAPI Service Server abstraction.
    
    Handles app creation, CORS middleware setup, lifespan lifecycle management,
    default health check endpoints, and route registration.
    """
    def __init__(
        self,
        title: str,
        description: str = "",
        version: str = "1.0.0",
        cors_origins: Optional[List[str]] = None,
        enable_health_check: bool = True
    ):
        self.title = title
        self.description = description
        self.version = version
        self.cors_origins = cors_origins or [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://localhost:8080",
            "http://localhost:5173",
        ]
        self._startup_handlers: List[Callable[[], Any]] = []
        self._shutdown_handlers: List[Callable[[], Any]] = []
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            for handler in self._startup_handlers:
                res = handler()
                if hasattr(res, "__await__"):
                    await res
            yield
            for handler in self._shutdown_handlers:
                res = handler()
                if hasattr(res, "__await__"):
                    await res
                    
        self.app = FastAPI(
            title=self.title,
            description=self.description,
            version=self.version,
            lifespan=lifespan
        )
        
        self._configure_cors()
        if enable_health_check:
            self._add_default_health_check()

    def _configure_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.cors_origins,
            allow_origin_regex=r"http(s)?://(localhost|127\.0\.0\.1)(:[0-9]+)?",
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _add_default_health_check(self):
        @self.app.get("/health")
        async def health():
            return {"status": "ok", "service": self.title}

    def on_startup(self, func: Callable[[], Any]):
        """Register a startup callback (sync or async)."""
        self._startup_handlers.append(func)
        return func

    def on_shutdown(self, func: Callable[[], Any]):
        """Register a shutdown callback (sync or async)."""
        self._shutdown_handlers.append(func)
        return func
