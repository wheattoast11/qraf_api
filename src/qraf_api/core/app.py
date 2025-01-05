from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional, Dict, Any

from ..config.settings import get_settings, Settings
from ..services.state_service import router as state_router
from ..services.optimization_service import router as optimization_router
from ..services.error_correction_service import router as error_correction_router

# Initialize settings
settings = get_settings()

# Initialize FastAPI app
app = FastAPI(
    title="QRAF API",
    description="Quantum-enhanced API service for infinite context and state management",
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
)

# Include routers
app.include_router(
    state_router,
    prefix=settings.API_PREFIX
)
app.include_router(
    optimization_router,
    prefix=settings.API_PREFIX
)
app.include_router(
    error_correction_router,
    prefix=settings.API_PREFIX
)

# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "cuda_available": "yes" if settings.USE_CUDA else "no",
        "services": {
            "state": "up",
            "optimization": "up",
            "error_correction": "up"
        }
    }

# Basic error handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail},
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    ) 