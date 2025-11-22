"""API routes"""

from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import cv2
import numpy as np
from datetime import datetime
from ..datasets.manager import DatasetManager
from ..models import YOLODetector
from ..inference import VideoProcessor
from ..vlm_llm import IncidentAnalyzer
from ..utils.config import load_config
from ..utils.logger import setup_logger

# Initialize
logger = setup_logger("APIRoutes")
config = load_config()

# Initialize managers
dataset_manager = DatasetManager()
video_processor = VideoProcessor(config.get_all())

# Create router
router = APIRouter()


# Pydantic models
class DatasetInfo(BaseModel):
    category: str
    name: str
    path: str
    format: str
    enabled: bool = True


class DetectionRequest(BaseModel):
    detections: Optional[List[str]] = None
    analyze_with_llm: bool = False


class RestrictedZone(BaseModel):
    zone_id: str
    name: str
    polygon: List[List[int]]  # List of [x, y] points


# Dataset endpoints
@router.get("/datasets")
async def list_datasets(category: Optional[str] = None, enabled_only: bool = False):
    """List available datasets"""
    try:
        datasets = dataset_manager.list_datasets(category, enabled_only)
        return {"success": True, "datasets": datasets}
    except Exception as e:
        logger.error(f"Failed to list datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/datasets")
async def add_dataset(dataset: DatasetInfo):
    """Add a new dataset"""
    try:
        success = dataset_manager.add_dataset(
            category=dataset.category,
            name=dataset.name,
            path=dataset.path,
            format=dataset.format,
            enabled=dataset.enabled
        )
        
        if success:
            return {"success": True, "message": "Dataset added successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to add dataset")
    
    except Exception as e:
        logger.error(f"Failed to add dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{category}/{name}")
async def get_dataset_info(category: str, name: str):
    """Get dataset information"""
    try:
        dataset = dataset_manager.registry.get_dataset(category, name)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        stats = dataset_manager.get_statistics(category, name)
        
        return {
            "success": True,
            "dataset": dataset,
            "statistics": stats
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/datasets/{category}/{name}/enable")
async def enable_dataset(category: str, name: str):
    """Enable a dataset"""
    try:
        success = dataset_manager.enable_dataset(category, name)
        if success:
            return {"success": True, "message": "Dataset enabled"}
        else:
            raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        logger.error(f"Failed to enable dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/datasets/{category}/{name}/disable")
async def disable_dataset(category: str, name: str):
    """Disable a dataset"""
    try:
        success = dataset_manager.disable_dataset(category, name)
        if success:
            return {"success": True, "message": "Dataset disabled"}
        else:
            raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as e:
        logger.error(f"Failed to disable dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Detection endpoints
@router.post("/detect/image")
async def detect_in_image(
    file: UploadFile = File(...),
    request: DetectionRequest = DetectionRequest()
):
    """Run detection on uploaded image"""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process image
        results = video_processor.pipeline.process_frame(
            image,
            detections_to_run=request.detections
        )
        
        # Analyze with LLM if requested
        if request.analyze_with_llm:
            analysis = video_processor.pipeline.analyze_incident(image, results)
            if analysis:
                results['analysis'] = analysis
        
        results['timestamp'] = datetime.now().isoformat()
        
        return {"success": True, "results": results}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/status")
async def get_models_status():
    """Get status of all models"""
    try:
        pipeline = video_processor.pipeline
        
        status = {
            "yolo": pipeline.yolo_detector.is_available(),
            "fall": pipeline.fall_detector.is_available(),
            "crowd": pipeline.crowd_detector.is_available(),
            "face": pipeline.face_detector.is_available(),
            "emotion": pipeline.emotion_detector.is_available(),
            "phone": pipeline.phone_detector.is_available(),
            "area": pipeline.area_detector.is_available(),
            "vlm_llm": pipeline.analyzer.is_available() if pipeline.analyzer else False
        }
        
        return {"success": True, "models": status}
    
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/zones/restricted")
async def add_restricted_zone(zone: RestrictedZone):
    """Add a restricted zone"""
    try:
        area_detector = video_processor.pipeline.area_detector
        
        # Convert polygon format
        polygon = [(p[0], p[1]) for p in zone.polygon]
        
        area_detector.add_restricted_zone(
            zone_id=zone.zone_id,
            polygon=polygon,
            name=zone.name
        )
        
        return {"success": True, "message": "Restricted zone added"}
    
    except Exception as e:
        logger.error(f"Failed to add restricted zone: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/zones/restricted/{zone_id}")
async def remove_restricted_zone(zone_id: str):
    """Remove a restricted zone"""
    try:
        area_detector = video_processor.pipeline.area_detector
        
        success = area_detector.remove_restricted_zone(zone_id)
        
        if success:
            return {"success": True, "message": "Restricted zone removed"}
        else:
            raise HTTPException(status_code=404, detail="Zone not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove restricted zone: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_config():
    """Get current configuration"""
    try:
        return {"success": True, "config": config.get_all()}
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            # Echo back for now
            await websocket.send_json({
                "type": "response",
                "timestamp": datetime.now().isoformat(),
                "message": "Connected to CASS API"
            })
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
