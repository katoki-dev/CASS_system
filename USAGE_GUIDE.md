# CASS System - Installation and Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Dataset Management](#dataset-management)
4. [Training Models](#training-models)
5. [Testing Models](#testing-models)
6. [Running Inference](#running-inference)
7. [API Usage](#api-usage)
8. [Configuration](#configuration)
9. [Advanced Usage](#advanced-usage)

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (recommended for better performance)
- ffmpeg (for video processing)

### Step 1: Clone the Repository
```bash
git clone https://github.com/katoki-dev/CASS_system.git
cd CASS_system
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install CASS Package
```bash
pip install -e .
```

## Quick Start

### 1. Test Detection on an Image
```bash
cass-infer --source path/to/image.jpg --image-mode --display --output output.jpg
```

### 2. Process a Video
```bash
cass-infer --source path/to/video.mp4 --display --output output.mp4
```

### 3. Use Webcam for Live Detection
```bash
cass-infer --source 0 --display
```

### 4. Start Web Interface
```bash
cass-server
```

Then open http://localhost:8000 in your browser.

## Dataset Management

### List Available Datasets
```bash
python -c "from cass.datasets import DatasetManager; dm = DatasetManager('configs/config.yaml'); print(dm.list_datasets())"
```

### Add a New Dataset
```python
from cass.datasets import DatasetManager

dm = DatasetManager('configs/config.yaml')

# Add COCO format dataset
dm.add_dataset(
    category='object_detection',
    name='my_custom_dataset',
    path='/path/to/dataset',
    format='coco',
    enabled=True
)

# Save configuration
dm.save_config('configs/config.yaml')
```

### Using the API
```bash
curl -X POST http://localhost:8000/api/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "category": "fall_detection",
    "name": "campus_falls",
    "path": "/path/to/dataset",
    "format": "video",
    "enabled": true
  }'
```

## Training Models

### Train YOLOv8 Model
```bash
cass-train \
  --model-type yolo \
  --dataset-category object_detection \
  --dataset-name COCO \
  --epochs 100 \
  --batch-size 16 \
  --device cuda \
  --output-dir runs/train
```

### Train with Custom Configuration
```python
from cass.training import ModelTrainer
from cass.utils.config import load_config

config = load_config('configs/config.yaml')
trainer = ModelTrainer(config.get('training', {}))

# Train YOLO
results = trainer.train_yolo(
    dataset_category='object_detection',
    dataset_name='my_dataset',
    epochs=50,
    batch_size=8,
    device='cuda'
)

print(f"Model saved to: {results['model_path']}")
```

## Testing Models

### Test a Single Model
```bash
cass-test \
  --model yolov8n.pt \
  --dataset-category object_detection \
  --dataset-name COCO \
  --max-samples 100 \
  --report test_report.txt
```

### Benchmark Multiple Models
```bash
cass-test \
  --model "yolov8n.pt,yolov8s.pt,yolov8m.pt" \
  --dataset-category object_detection \
  --dataset-name COCO \
  --benchmark \
  --max-samples 100
```

### Programmatic Testing
```python
from cass.testing import ModelTester
from cass.utils.config import load_config

config = load_config('configs/config.yaml')
tester = ModelTester(config.get('testing', {}))

# Test model
results = tester.test_yolo_model(
    model_path='models/weights/my_model.pt',
    dataset_category='object_detection',
    dataset_name='test_set',
    max_samples=100
)

print(f"Average FPS: {results['fps']:.2f}")
print(f"Inference Time: {results['avg_inference_time']:.4f}s")

# Generate report
tester.generate_report('test_report.txt')
```

## Running Inference

### Image Detection
```bash
cass-infer \
  --source image.jpg \
  --image-mode \
  --output output.jpg \
  --detections "fall,crowd,face"
```

### Video Processing
```bash
cass-infer \
  --source video.mp4 \
  --output output.mp4 \
  --detections "fall,crowd,phone" \
  --display
```

### RTSP Stream
```bash
cass-infer \
  --source "rtsp://camera-ip:554/stream" \
  --display \
  --detections "fall,area"
```

### Programmatic Usage
```python
from cass.inference import VideoProcessor
from cass.utils.config import load_config

config = load_config('configs/config.yaml')
processor = VideoProcessor(config.get_all())

# Process image
results = processor.process_image(
    image_path='test.jpg',
    output_path='output.jpg',
    detections_to_run=['fall', 'crowd', 'face']
)

# Check alerts
if results.get('alerts'):
    for alert in results['alerts']:
        print(f"Alert: {alert['type']} - Severity: {alert['severity']}")

# Process video
stats = processor.process_video(
    source='video.mp4',
    output_path='output.mp4',
    display=True,
    detections_to_run=['fall', 'crowd']
)

print(f"Processed {stats['frames_processed']} frames")
print(f"Total alerts: {stats['total_alerts']}")
```

## API Usage

### Start API Server
```bash
cass-server
```

Or programmatically:
```python
from cass.api import run
run()
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### List Datasets
```bash
curl http://localhost:8000/api/datasets
```

#### Detect in Image
```bash
curl -X POST http://localhost:8000/api/detect/image \
  -F "file=@test.jpg"
```

#### Get Model Status
```bash
curl http://localhost:8000/api/models/status
```

#### Add Restricted Zone
```bash
curl -X POST http://localhost:8000/api/zones/restricted \
  -H "Content-Type: application/json" \
  -d '{
    "zone_id": "zone1",
    "name": "Server Room",
    "polygon": [[100, 100], [300, 100], [300, 300], [100, 300]]
  }'
```

## Configuration

The main configuration file is `configs/config.yaml`. Key sections:

### Dataset Configuration
```yaml
datasets:
  fall_detection:
    - name: "UR-Fall"
      path: "datasets/fall_detection/ur_fall"
      format: "video"
      enabled: true
```

### Model Configuration
```yaml
models:
  yolo:
    version: "yolov8"
    weights: "yolov8n.pt"
    device: "cuda"
    conf_threshold: 0.25
```

### VLM/LLM Configuration
```yaml
vlm_llm:
  enabled: true
  llm:
    model_type: "llama"
    model_path: "models/llm/llama-2-7b-chat.gguf"
    device: "cuda"
  vlm:
    model_type: "blip"
    device: "cuda"
```

### Training Configuration
```yaml
training:
  batch_size: 16
  epochs: 100
  learning_rate: 0.001
  device: "cuda"
```

## Advanced Usage

### Using VLM/LLM for Incident Analysis
```python
from cass.vlm_llm import IncidentAnalyzer
from cass.utils.config import load_config
import cv2

config = load_config('configs/config.yaml')
analyzer = IncidentAnalyzer(config.get('vlm_llm', {}))

# Analyze incident with image
image = cv2.imread('incident.jpg')
incident_data = {
    'event_type': 'fall_detected',
    'severity': 'critical',
    'timestamp': '2025-11-22T10:30:00',
    'location': 'Building A, Floor 2'
}

analysis = analyzer.analyze_incident(incident_data, image)

print("Visual Description:", analysis['visual_description'])
print("Text Analysis:", analysis['text_analysis'])
print("Recommendations:", analysis['recommendations'])
```

### Custom Detection Pipeline
```python
from cass.inference import DetectorPipeline
from cass.models import RestrictedAreaDetector
import cv2

# Initialize pipeline
config = load_config('configs/config.yaml')
pipeline = DetectorPipeline(config.get_all())

# Add restricted zones
pipeline.area_detector.add_restricted_zone(
    zone_id='server_room',
    polygon=[(100, 100), (400, 100), (400, 400), (100, 400)],
    name='Server Room'
)

# Process frame
frame = cv2.imread('camera_frame.jpg')
results = pipeline.process_frame(frame, detections_to_run=['fall', 'area'])

# Visualize
vis_frame = pipeline.visualize_detections(frame, results)
cv2.imwrite('output.jpg', vis_frame)
```

### Alert Callback
```python
from cass.inference import VideoProcessor

def on_alert(alert, results, frame):
    print(f"ALERT: {alert['type']} - Severity: {alert['severity']}")
    
    # Save frame
    import cv2
    from datetime import datetime
    filename = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, frame)
    
    # Send notification (implement your notification logic)
    # send_sms(alert)
    # send_email(alert, frame)

config = load_config('configs/config.yaml')
processor = VideoProcessor(config.get_all())
processor.set_alert_callback(on_alert)

# Process video with callback
processor.process_video('camera_stream.mp4')
```

## Troubleshooting

### GPU Not Detected
If CUDA is not available, models will run on CPU. To use GPU:
1. Install CUDA Toolkit
2. Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Memory Issues
Reduce batch size or image resolution:
```yaml
training:
  batch_size: 8  # Reduce from 16

inference:
  frame_skip: 4  # Process fewer frames
```

### Model Not Found
Download models:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Will auto-download
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/katoki-dev/CASS_system/issues
- Documentation: See PRODUCT_DESCRIPTION.md and campus_ai_surveillance_plan.md

## License

See LICENSE file for details.
