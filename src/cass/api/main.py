"""FastAPI application"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn
from .routes import router
from ..utils.config import load_config
from ..utils.logger import setup_logger

# Initialize logger
logger = setup_logger("API")

# Load configuration
config = load_config()
api_config = config.get('api', {})

# Create FastAPI app
app = FastAPI(
    title="CASS API",
    description="Campus AI Safety & Surveillance System API",
    version="0.1.0"
)

# CORS middleware
cors_origins = api_config.get('cors_origins', ['*'])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api")

# Serve static files (for frontend)
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CASS API",
        "version": "0.1.0",
        "endpoints": {
            "api": "/api",
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


def run():
    """Run the API server"""
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    
    logger.info(f"Starting CASS API on {host}:{port}")
    
    uvicorn.run(
        "cass.api.main:app",
        host=host,
        port=port,
        reload=True
    )


if __name__ == "__main__":
    run()
