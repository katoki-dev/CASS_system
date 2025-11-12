# Campus AI Safety & Surveillance System (CASS)

> World-class AI-driven safety and surveillance suite for educational campuses, covering recording, crowd and fall detection, face and emotion analysis, mobile phone spotting, unethical-activity identification, restricted-area enforcement, automated notifications, role-based web experience, and optional auto-clipping. Built on YOLOv12, OpenCV, and curated datasets.

---

## Table of Contents
1. System Overview
2. High-Level Data Flow
3. Components & Responsibilities
4. Core Features & Capabilities
5. Models, Datasets & Application Context
6. Training & Fine-Tuning Guidance
7. Inference & Performance Tuning
8. Auto-Clipping & Storage Strategy
9. Notification Flow & Templates
10. Website Experience (Admin / Staff / Guest)
11. Integration & APIs
12. Hardware & Infrastructure Recommendations
13. Security, Privacy & Ethical Safeguards
14. Adversarial Robustness & System Hardening
15. Human Workflows & SOPs
16. Implementation Roadmap
17. Example Inference Pseudocode
18. Performance Metrics & KPI Targets
19. Evaluation, Monitoring & Retraining
20. Future Enhancements & Expansion
21. Next Actions

---

## 1. System Overview
CASS delivers a modular pipeline with four tightly coupled layers:
- **Edge Capture Layer** — IP cameras and edge compute nodes ensure low-latency detections and instant alerts.
- **Inference Layer** — YOLOv12 plus specialized models (pose, face, emotion, action) running on edge accelerators or a central GPU cluster.
- **Backend & Orchestration Layer** — REST/WebSocket APIs, event management, video/object storage, analytics, logging, audit, and escalation services.
- **Notification & Response Layer** — Multi-channel delivery (SMS, push, email, webhooks) to incharge, security, and medical teams, paired with a live dispatch dashboard.

**Non-functional goals:** end-to-end latency <3 seconds for critical events, high recall for safety-critical detections, privacy-preserving defaults, model explainability, and a polished UX for all stakeholders.

---

## 2. High-Level Data Flow
1. IP camera streams (RTSP) feed an edge node (Jetson Orin/AGX, Intel NUC, or GPU workstation).
2. Frames are sampled, pre-processed, and passed through YOLOv12 and secondary models (pose/emotion/action).
3. Event rules evaluate detections (e.g., fall + immobility threshold, trespass within geo-fenced polygon).
4. Auto-clipping pulls pre/post-event footage from a circular buffer and uploads to an S3-compatible store.
5. Backend registers the incident, persists metadata, audits the action, and kicks off notification workflows.
6. Admin and staff access live dashboards to acknowledge, escalate, or resolve alerts while reviewing linked clips.

---

## 3. Components & Responsibilities
### Cameras & Edge Devices
- ONVIF/RTSP-capable 1080p IP cameras (PoE preferred) with local circular buffers (10–30 seconds) per stream.
- Edge accelerators: NVIDIA Jetson Orin/AGX Xavier (primary), Intel Neural Compute Stick 2 or GPU desktops as alternates.

### Inference Stack
- **YOLOv12** as the unified object detector (people, mobile phones, bags, suspicious objects).
- **Face analytics**: WIDER FACE detector + ArcFace/ResNet embeddings for opt-in recognition.
- **Fall detection**: OpenPose/MediaPipe skeletal keypoints + temporal models (1D-CNN/LSTM) trained on campus scenarios.
- **Crowd analytics**: YOLO person counting with DeepSORT tracking or CSRNet density maps.
- **Emotion detection**: Facial crops processed by AffectNet/FER2013-fine-tuned classifiers (deployed with conservative thresholds and human review).
- **Unethical activity detection**: SlowFast/I3D action models highlighting aggression, harassment, or vandalism (human-in-the-loop mandatory).
- **Restricted-area enforcement**: Homography mapping between camera views and geo-fenced polygons; rule engine for trespass detection.

### Backend & Orchestration
- Event ingestion APIs, messaging queue, Postgres (events/config), Redis (caching), S3-compatible object storage, notification gateways (SMS/email/push), audit logging, configuration service, analytics ETL.

### Website / UI
- Role-based web portal with live camera grid overlays, incident queue, map view, clip viewer, rules editor, analytics dashboards, and admin configuration tools.

---

## 4. Core Features & Capabilities
### Recording & Storage
- 24/7 recording with configurable retention (30/60/90 days) and encrypted archives.
- Intelligent storage tiers (hot vs. cold) and immutable audit logs.
- Optional auto-clipping (15s pre / 30s post default) with metadata tagging, export, and share links.

### Event Detection Portfolio
- **Crowd detection**: Density heat maps, live counts, and occupancy alerts for auditoriums, cafeterias, and sports arenas.
- **Fall detection**: High-recall falls vs. intentional actions, severity scoring, and medical response triggers.
- **Face detection & recognition (opt-in)**: Authorized personnel identification, visitor management, and consent-aware masking for others.
- **Emotion detection**: Distress/aggression/ fear cues powering welfare checks (subject to human validation).
- **Mobile phone detection**: Policy enforcement in examination halls, labs, or no-record zones with temporal rule context.
- **Unethical activity detection**: Early warning for bullying, fights, vandalism, substance use; escalated after human confirmation.
- **Restricted-area trespass**: Real-time violation alerts with map overlays, after-hours monitoring, and access-control integration.

### Notification & Response
- Severity-based rule engine (Critical, High, Medium, Low) delivering SMS, push, email, and dashboard alerts.
- Role-specific escalations (Admin, Security, Medical) with acknowledgment workflows and audit trails.
- Automated ticket creation, incident lifecycle management, and follow-up tasks.

---

## 5. Models, Datasets & Application Context
- **YOLOv12 base training**: Pre-train on COCO/OpenImages, fine-tune with campus-labeled frames for people, phones, bags, suspicious objects.
- **Face detection/recognition**: WIDER FACE detector; VGGFace2/ArcFace embeddings refined with opt-in faculty/student datasets.
- **Fall detection**: UR-Fall, UP-Fall, Le2i, plus cloned GitHub resources (`Fall-Detection-Dataset`, `Fall-Detection-System`, `Real-Time-Fall-Detection-using-YOLO`, `roboflow/notebooks`); continue with campus-specific annotations.
- **Crowd analytics**: ShanghaiTech A/B, UCF_CC_50, WorldExpo'10, complemented by high-occupancy campus footage.
- **Emotion detection**: AffectNet, FER2013, EmotiW — deploy with strict thresholds and review queues.
- **Unethical activity**: Kinetics-700, AVA datasets, and curated campus scenarios; ensure privacy/legal review.
- **Restricted-zone logic**: Calibration frames, homography matrices, and CAD-based polygon definitions per building.

---

## 6. Training & Fine-Tuning Guidance
- Annotation tools: CVAT, LabelBox, Supervisely.
- Dataset splits: time-based train/val/test to avoid leakage; ensure class balance.
- Augmentations: brightness/contrast, Gaussian blur, scaling, occlusion, motion blur for night scenarios.
- Hyperparameters: start with YOLOv12 defaults, cosine LR scheduler (~1e-3), warm restarts for small datasets.
- Small-object handling: higher-resolution training (1280×1280), focal loss adjustments, class re-weighting.
- Evaluate with mAP@0.5 for detection, F1 for event-level alarms, MAE for counting.

---

## 7. Inference & Performance Tuning
- Pipeline: RTSP decode → adaptive frame sampling → resize → YOLOv12 inference → NMS → tracking (DeepSORT/SORT) → event rules.
- Edge vs. cloud split: life-critical detections on edge; compute-heavy analytics (crowd density, action recognition) can offload to central GPU.
- Frame-rate management: dynamic FPS increase during detected motion; degrade gracefully during bandwidth constraints.
- TensorRT/ONNX optimizations for YOLOv12 and pose models; mixed precision (FP16/INT8) on Jetson devices.

---

## 8. Auto-Clipping & Storage Strategy
- Edge circular buffers retain recent frames (default 10–30 seconds pre-event).
- On trigger, combine pre/post frames, transcode to H.264/H.265, encrypt, and upload to object storage with signed URLs.
- Retention policies per event type (critical: 90 days, medium: 30 days, low: 14 days) with role-based access and legal hold options.
- Storage tiers: hot (SSD, quick retrieval), warm (S3 standard), cold (Glacier/Deep Archive) with lifecycle policies.

---

## 9. Notification Flow & Templates
- **Fall (Critical)**: instant SMS/push to security + medical with camera ID, timestamp, location, snapshot, clip link.
- **Crowd (High)**: security alert with occupancy count, duration, recommended dispersal actions.
- **Restricted-area (High)**: alarm to security, map overlay, access history.
- **Unethical/Mobile phone (Medium)**: routed to review queue; optional alerts to exam invigilators or deans.

Example payload:
```json
{
  "event": "fall_detected",
  "camera": "BuildingA_Cam12",
  "time": "2025-11-12T04:30:12+05:30",
  "location": "Building A, Level 2",
  "clip_url": "s3://cass-events/fall_123.mp4",
  "snapshot_url": "s3://cass-events/fall_123.jpg",
  "priority": "critical"
}
```

---

## 10. Website Experience (Admin / Staff / Guest)
- **Admin**: full configuration, user/role management, model retraining scheduler, retention governance, integration settings, audit explorer, analytics.
- **Security Staff**: live camera grid with overlays, incident queue, acknowledgment/escalation buttons, map view, clip viewer, shift summaries.
- **Medical Staff**: prioritized fall/emergency alerts, patient location history, clip access, status updates.
- **Guest**: limited public feeds, emergency contacts, incident reporting form, transparency dashboard.
- Cross-role features: WebSocket-driven real-time updates, dark/light themes, accessibility compliance (WCAG 2.1 AA).

---

## 11. Integration & APIs
- REST endpoints (illustrative):
  - `POST /api/events` — ingest detections from edge nodes.
  - `GET /api/incidents?status=open` — list actionable events.
  - `POST /api/incidents/{id}/acknowledge` — workflow actions.
  - `POST /api/zones` — manage restricted areas.
- WebSocket channels for live incidents and camera overlays.
- Webhooks for third-party systems (paging, ticketing, communication platforms).
- SDKs (Python/TypeScript) for campus IT integration.

---

## 12. Hardware & Infrastructure Recommendations
- **Edge**: Jetson Orin (32GB) or AGX Xavier per 4–6 camera cluster; Jetson Nano for pilot deployments.
- **Core servers**: GPU nodes (NVIDIA A10/A100) or cloud GPU instances for centralized inference/training.
- **Storage**: S3-compatible object store (MinIO/AWS S3), Postgres for relational data, Redis for caching/queues.
- **Networking**: Dedicated VLANs for cameras, VPN tunnels between buildings, QoS prioritization for alert traffic.
- **Power & resilience**: UPS for edge nodes, redundant network paths, environmental monitoring for racks.

---

## 13. Security, Privacy & Ethical Safeguards
- Default face blurring for stored clips; opt-in recognition only with consent and encrypted embeddings.
- Compliance alignment with FERPA, GDPR-equivalent regulations, and local privacy laws.
- Mandatory signage and privacy notices; clear opt-out procedures for sensitive detections.
- Fine-grained RBAC, MFA, session management, and immutable audit logs.
- Legal review for high-sensitivity detections (emotion, intimate interactions); always human-reviewed before action.

---

## 14. Adversarial Robustness & System Hardening
- Secure boot and full-disk encryption on edge nodes; mutual TLS for edge-backend communications.
- Continuous monitoring of detection confidence distributions to flag adversarial attacks or sensor tampering.
- Regular vulnerability scanning, penetration testing, and dependency patching.
- Rate limiting, anomaly detection, and circuit breakers on APIs.

---

## 15. Human Workflows & SOPs
- SOP libraries for fall response, crowd control, trespass intervention, and unethical-activity escalation.
- UI quick actions: dispatch guard, call ambulance, notify dean, mark false alarm—all auditable.
- Training drills with simulated alerts; post-incident reviews feeding continuous improvement loops.

---

## 16. Implementation Roadmap
- **Weeks 0–2 (Planning)**: Hardware selection, campus mapping, PoC camera selection, data governance approval.
- **Weeks 2–6 (PoC)**: Deploy YOLOv12 person detection on 3–5 cameras, configure circular buffers, deliver basic UI, pilot fall detection.
- **Weeks 6–10 (Expansion)**: Add crowd counting, restricted-area enforcement, mobile phone detection, integrate notification stack.
- **Weeks 10–14 (Face & Emotion)**: Launch opt-in face enrollment, deploy emotion detection with human review queues.
- **Weeks 14–16 (Hardening & Pilot)**: MLOps pipeline, monitoring dashboards, security audit, full pilot operations.

---

## 17. Example Inference Pseudocode
```python
import cv2
from yolov12 import YOLOv12Model
from fall_detector import FallDetector
from buffer import CircularBuffer
from notifier import notify

model = YOLOv12Model.load("weights/campus_yolov12.pt")
fall_detector = FallDetector.load("weights/fall_lstm.pt")
cap = cv2.VideoCapture("rtsp://BuildingA_Cam12")
buffer = CircularBuffer(seconds=20)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    buffer.push(frame)
    detections = model.detect(frame)
    tracked = track_people(detections)
    if fall_detector.is_fall(tracked):
        clip = buffer.dump(pre_seconds=10, post_seconds=20)
        clip_url = upload_clip(clip)
        snapshot = save_snapshot(frame)
        notify({
            "event": "fall_detected",
            "camera": "BuildingA_Cam12",
            "clip_url": clip_url,
            "snapshot": snapshot,
            "priority": "critical"
        })
```

---

## 18. Performance Metrics & KPI Targets
- Fall detection recall ≥ 98% after campus-specific tuning.
- Crowd detection MAE ≤ 5 for gatherings under 50 people.
- Alert latency (camera → notification) < 3 seconds for Critical events.
- System availability ≥ 99.9% with redundant failover.
- False positive rate for emotion/unethical activity < 5% (post-human review).

---

## 19. Evaluation, Monitoring & Retraining
- Runtime metrics: per-class precision/recall, confusion matrices, drift detection.
- Scheduled model evaluations with hold-out datasets; periodic re-labeling using active learning.
- Human sampling of alerts for QA; feedback used to retrain detectors.
- Monitoring stack (Prometheus/Grafana) for system health, latency, GPU utilization.

---

## 20. Future Enhancements & Expansion
- Predictive analytics for risk assessment and preventative deployment of resources.
- NLP-powered chatbots and voice interfaces for command/control and report generation.
- IoT sensor fusion (environmental sensors, access badges) for richer situational awareness.
- Native mobile apps (iOS/Android) with offline-ready alerts and secure remote viewing.
- AR/VR visualizations for emergency drills and situational playback.

---

## 21. Next Actions
1. **Data Labeling Plan** — Define schema, annotation guidelines, and QA processes (option A in plan).
2. **API & Schema Specification** — Draft REST/WebSocket contracts and Postgres schema (option B).
3. **Edge Deployment Tooling** — Author Dockerfiles and deploy scripts for Jetson/Ubuntu + YOLOv12 (option C).
4. **Admin UI Mockups** — Build React component list and low-fidelity wireframes (option D).
5. **Training Configurations** — Prepare YOLOv12 fine-tuning configs and command templates (option E).

---

**Document Version**: 2.0  
**Last Updated**: 2025-11-12  
**Prepared For**: Campus Safety & Surveillance Stakeholders

