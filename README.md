# Campus AI Safety & Surveillance System (CASS)

AI-first safety and security platform for educational campuses, combining YOLOv8-based video analytics, multi-modal detection, automated alerting, VLM/LLM integration, and role-aware web tools.

---

## 🚀 Key Capabilities
- **Real-time video analytics** for crowd density, falls, faces, emotions, mobile-phone usage, and restricted-area trespass
- **Auto-clipping engine** that captures pre/post event footage and stores clips with metadata
- **Local VLM/LLM support** for incident analysis and report generation with privacy preservation
- **Dataset management system** supporting COCO, YOLO, video, and custom formats with easy dataset addition
- **Training & testing framework** for model development and comparative benchmarking
- **Role-based web interface** with REST API and WebSocket support for real-time updates
- **Privacy-first design** with opt-in face recognition, face blurring, and complete audit trails

---

## 🧱 Architecture Overview
1. **Detection Layer** – YOLOv8 base + specialized models (pose, face, emotion) for comprehensive detection
2. **Dataset Management** – Flexible registry supporting multiple dataset formats with easy addition
3. **VLM/LLM Integration** – Local vision-language models (LLaVA, BLIP) and LLMs (LLaMA, Mistral) for analysis
4. **Training & Testing** – Complete pipeline for model training, evaluation, and benchmarking
5. **Inference Pipeline** – Real-time processing with video streams, auto-clipping, and alert generation
6. **Web API** – FastAPI backend with WebSocket support and interactive web interface

---

## 📦 Repository Layout
- `PRODUCT_DESCRIPTION.md` – Detailed product vision, feature breakdown, and technical blueprint
- `campus_ai_surveillance_plan.md` – Implementation plan covering data, models, infrastructure, and SOPs
- `USAGE_GUIDE.md` – Comprehensive installation and usage documentation
- `src/cass/` – Main source code
  - `datasets/` – Dataset management system
  - `models/` – Detection models (YOLO, fall, crowd, face, emotion, phone, area)
  - `vlm_llm/` – VLM/LLM integration for incident analysis
  - `training/` – Model training pipeline
  - `testing/` – Model testing and benchmarking
  - `inference/` – Real-time detection pipeline
  - `api/` – FastAPI web interface
- `configs/` – Configuration files
- `tests/` – Test suite

---

## 🧠 Models & Datasets
- **Object detection:** YOLOv8 pre-trained on COCO, customizable for campus-specific objects
- **Fall detection:** MediaPipe pose estimation with temporal fall detection algorithms
- **Crowd detection:** YOLO-based person counting with density estimation
- **Face detection:** OpenCV Haar Cascade with optional deep learning embeddings
- **Emotion detection:** Facial expression analysis (placeholder for custom models)
- **Phone detection:** YOLO-based mobile phone detection in context
- **VLM Support:** BLIP, LLaVA, GIT for visual understanding
- **LLM Support:** LLaMA, Mistral, GPT4All for incident analysis

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- ffmpeg for video processing

### Quick Start
```bash
# Clone repository
git clone https://github.com/katoki-dev/CASS_system.git
cd CASS_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install CASS package
pip install -e .
```

### Run Examples

**Test detection on an image:**
```bash
cass-infer --source test.jpg --image-mode --display --output output.jpg
```

**Process a video:**
```bash
cass-infer --source video.mp4 --display --output output.mp4
```

**Start web interface:**
```bash
cass-server
```
Then open http://localhost:8000

**Train a model:**
```bash
cass-train --model-type yolo --dataset-category object_detection --dataset-name COCO --epochs 50
```

**Test a model:**
```bash
cass-test --model yolov8n.pt --dataset-category object_detection --dataset-name COCO --max-samples 100
```

For detailed usage, see [USAGE_GUIDE.md](USAGE_GUIDE.md).

---

## 🎯 Key Features

### Dataset Management
- Support for multiple formats: COCO, YOLO, video collections, image datasets
- Easy dataset addition through API or Python interface
- Dataset registry with enable/disable controls
- Statistics and validation tools

### Detection Capabilities
1. **Fall Detection** - MediaPipe pose estimation with fall detection algorithms
2. **Crowd Detection** - Person counting and density estimation
3. **Face Detection** - Face detection with optional recognition and privacy blurring
4. **Emotion Detection** - Facial expression analysis
5. **Mobile Phone Detection** - Context-aware phone usage detection
6. **Restricted Area** - Geo-fenced zone monitoring with trespass detection

### VLM/LLM Integration
- **Local VLM**: BLIP, LLaVA, or GIT for visual understanding
- **Local LLM**: LLaMA, Mistral, or GPT4All for incident analysis
- Incident analysis and report generation
- Query answering with visual context
- Privacy-preserving (runs locally)

### Training & Testing
- YOLOv8 training pipeline with custom datasets
- Model benchmarking and comparison
- Performance metrics (FPS, inference time, accuracy)
- Report generation

### Web Interface
- FastAPI backend with OpenAPI documentation
- WebSocket support for real-time updates
- REST API for all operations
- Interactive web UI for testing
- Dataset management interface

---

## 🗺️ API Endpoints

### Datasets
- `GET /api/datasets` - List all datasets
- `POST /api/datasets` - Add new dataset
- `GET /api/datasets/{category}/{name}` - Get dataset info
- `PUT /api/datasets/{category}/{name}/enable` - Enable dataset

### Detection
- `POST /api/detect/image` - Detect in uploaded image
- `GET /api/models/status` - Get model status

### Zones
- `POST /api/zones/restricted` - Add restricted zone
- `DELETE /api/zones/restricted/{zone_id}` - Remove zone

### System
- `GET /health` - Health check
- `GET /api/config` - Get configuration
- `WS /api/ws` - WebSocket for real-time updates

Full API documentation at http://localhost:8000/docs

---

## 📖 Documentation
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** - Complete installation and usage guide
- **[PRODUCT_DESCRIPTION.md](PRODUCT_DESCRIPTION.md)** - Product vision and technical details
- **[campus_ai_surveillance_plan.md](campus_ai_surveillance_plan.md)** - Implementation roadmap

---

## 🧾 License
This repository aggregates multiple third-party datasets and notebooks (see `datasets/fall detection/`). Consult each upstream license before redistribution or commercial use.

---

## 🤝 Contributing
1. Fork the repo and create a feature branch.
2. Align proposals with `campus_ai_surveillance_plan.md` and update documentation.
3. Submit PRs with tests, evaluation metrics, and deployment notes where applicable.

---

## 📬 Support & Next Steps
- Choose a “Next Action” from `PRODUCT_DESCRIPTION.md` §21 to continue development (data labeling, API spec, edge tooling, UI mockups, training configs).
- For issues or enhancements, open a GitHub Issue with reproduction steps and impacted modules.

---

**Maintainer:** `katoki-dev`  |  **Last Updated:** 2025-11-12
