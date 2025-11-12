# World-Class AI-Based Safety & Surveillance System for Campuses

> Comprehensive plan including recording, crowd detection, fall detection, face detection, emotion detection, mobile phone detection, unethical-activity detection, restricted-area trespass detection, notification flows, role-based website, and optional auto-clipping. Uses YOLOv12, OpenCV and recommended datasets.

---

## Table of Contents

1. System Overview
2. High-level Data Flow
3. Components & Responsibilities
   - Cameras & Edge devices
   - Inference Stack
   - Backend & Orchestration
   - Website / UI
4. Models, Datasets & Where to Use Them
5. Training & Fine-tuning Guidance
6. Inference & Performance Tuning
7. Auto-clipping & Storage
8. Notification Flow & Templates
9. Website: Roles & Features
10. Integration & APIs
11. Hardware & Infrastructure Recommendations
12. Evaluation, Monitoring & Retraining
13. Privacy, Legal & Ethical Safeguards
14. Security & Adversarial Robustness
15. Human Workflows & SOPs
16. Implementation Roadmap
17. Example Inference Pseudocode
18. Risk Areas & Mitigations
19. KPI Targets
20. Tools & Libraries
21. Next Actions

---

## 1. System Overview
A modular pipeline with three layers:

- **Edge capture layer** — Cameras + local compute for low-latency detection and immediate alerts.
- **Processing & inference layer** — YOLOv12 + specialized models (emotion, face recognition, pose/fall) running on edge or central GPU cluster.
- **Backend & orchestration** — APIs, DB, video store, alert service, admin web UI, logging, and audit.
- **Notification & response** — SMS/voice/push/email/webhooks to security, in-charge, medical staff; dispatch dashboard with live feed & auto-clips.

Non-functional goals: low latency, high accuracy, privacy-preserving defaults, explainability, and polished UX.

---

## 2. High-level Data Flow

1. IP camera → RTSP stream → Edge node (Jetson/Intel NCS/GPU).
2. Frame sampling → Preprocessing → YOLOv12 + specialized models.
3. Event logic triggers alert (e.g., fall + immobility > threshold).
4. Clip generation from circular buffer → store in object store and attach to alert.
5. Backend registers event, stores metadata, notifies staff, logs audit.
6. Admin/staff view via website with acknowledge/escalate controls.

---

## 3. Components & Responsibilities

### Cameras & Edge devices
- RTSP-capable IP cameras (1080p recommended, PoE), Jetson Orin/AGX or Intel alternatives.
- Circular buffer on edge for last N minutes for auto-clips.

### Inference Stack
- **YOLOv12**: base detector for people, phones, bags, weapons, etc.
- **Face detection & recognition**: WIDER FACE + ArcFace/ResNet embeddings.
- **Fall detection**: pose estimation (OpenPose/MediaPipe) + temporal classifier (LSTM/1D-CNN).
- **Crowd detection**: person-count + density estimation (CSRNet) or YOLOv12 aggregation.
- **Emotion detection**: face-crop → classifier (AffectNet/FER2013 fine-tune).
- **Mobile phone detection**: YOLOv12 fine-tuned to detect phone-in-hand contexts.
- **Unethical activity detection**: action recognition (I3D/SlowFast) for interactions like kissing — human-in-loop required.
- **Restricted-area trespass**: homography mapping + geo-polygon rules.

### Backend & Orchestration
- REST + WebSocket API, event manager, object store (S3), Postgres DB, Redis, notification gateway (SMS/push), audit logs.

### Website / UI
- Role-based views (Admin / Security / Medical / Guest), live camera grid with overlays, incident dashboard, clip viewer, rules & area editor, analytics.

---

## 4. Models, Datasets & Where to Use Them

**YOLOv12 (base)**
- Pretrain: COCO, OpenImages
- Fine-tune: campus-labeled frames for people/phones/bags

**Face detection & recognition**
- Detector: WIDER FACE
- Embeddings: VGGFace2, fine-tune with campus enrollment

**Fall detection**
- Datasets: UR-Fall, UP-Fall, Le2i
- Approach: keypoint extraction → temporal classifier

**Crowd detection**
- Datasets: ShanghaiTech A/B, UCF_CC_50
- Options: YOLO person-count or CSRNet density

**Emotion detection**
- Datasets: AffectNet, FER2013 (use conservatively)

**Mobile phone detection**
- Datasets: OpenImages (phone class) + custom campus annotations

**Unethical/activity detection**
- Datasets: AVA, Kinetics, custom campus-labeled interactions (HIGHLY sensitive)

**Restricted-area trespass**
- Use homography mapping from camera frame to campus map polygons

---

## 5. Training & Fine-tuning Guidance

- Annotation tools: CVAT, LabelBox, Supervisely.
- Splits: train/val/test with temporal split.
- Augmentations: brightness, blur, scaling, occlusion.
- Hyperparams: start with YOLOv12 defaults, LR ~1e-3 with cosine scheduler.
- Small-object handling: higher-res inputs, positive re-weighting.
- Evaluation: mAP@0.5 for detection; precision/recall/F1 for event-level alerts.

---

## 6. Inference & Performance Tuning

- Pipeline: RTSP decode → sample → resize → detect → NMS → tracking.
- Tracking: SORT / DeepSORT for consistent IDs.
- Edge vs Cloud: critical events on edge; analytics and heavy models in cloud/GPU cluster.
- Frame-rate control: raise fps on motion detection.

---

## 7. Auto-clipping & Storage

- Circular buffer on each camera (default 10–30s pre-event).
- On event, persist pre/post buffer to S3-compatible store.
- Retention policy (e.g., default 30 days) and role-based access.
- Encryption at rest & in transit.

---

## 8. Notification Flow & Templates

**Fall (HIGH)** — immediate SMS/push to security + medical
Payload includes: camera ID, timestamp, GPS/map, snapshot, clip link.

**Crowd** — alert security with count, duration, snapshot, suggested action.

**Restricted-area trespass** — alarm + snapshot + guard dispatch.

**Unethical / phone use** — send to review queue; exam invigilator optional notification for phone use.

Example JSON payload snippet:

```json
{
  "event":"fall_detected",
  "camera":"BuildingA_Cam12",
  "time":"2025-11-12T04:30:12+05:30",
  "location":"Building A, 2nd floor (lat,lon)",
  "clip_url":"s3://.../clip_123.mp4",
  "snapshot_url":"s3://.../snap_123.jpg",
  "priority":"high"
}
```

---

## 9. Website: Roles & Features

- **Admin:** full config, user mgmt, model retrain scheduling, retention policies.
- **Security:** live view, incidents, assign/acknowledge.
- **Medical:** receive fall alerts, access clips.
- **Guest:** limited public feeds.

Features: Live camera grid with overlays, incident lists, map view, clip viewer, rules editor.

---

## 10. Integration & APIs

Suggested endpoints:

- `POST /api/events` — receive events
- `GET /api/cameras` — list cameras
- `GET /api/events?status=open` — incidents
- `POST /api/acknowledge` — acknowledge incident

Use WebSocket for real-time UI pushes.

---

## 11. Hardware & Infrastructure Recommendations

- **Edge:** NVIDIA Jetson Orin/AGX (on-prem inference). Jetson Nano for light deployments.
- **Server:** GPU nodes (A10 / A100) for training/central inference.
- **Storage:** S3-compatible object store; Postgres; Redis; message queue.
- **Networking:** private VPN for camera links; QoS for alerts.

---

## 12. Evaluation, Monitoring & Retraining

- Metrics: per-class precision/recall, mAP, MAE for counting.
- SLOs: fall detection latency < 2s; tune false negative rates per campus policy.
- Monitor model drift, provide human-review sampling & active learning loop.

---

## 13. Privacy, Legal & Ethical Safeguards

- Default: blur faces in stored clips unless consent.
- Face recognition: **opt-in only**. Store embeddings encrypted.
- Sensitive detections (emotion, sexual activity) must be human-reviewed.
- Maintain audit logs, retention policy and compliance with local laws.

---

## 14. Security & Adversarial Robustness

- Harden edge (secure boot, encryption). Mutual TLS for edge-backend.
- Monitor confidence distributions to detect adversarial inputs.
- Access control and rate limiting.

---

## 15. Human Workflows & SOPs

- SOPs for fall alerts, crowd dispersal, and suspected unethical activity.
- Quick UI actions: Dispatch guard, Call ambulance, Mark false alarm.
- Each action logged for audit.

---

## 16. Implementation Roadmap (Phased)

**Week 0–2 — Planning**: hardware selection, map campus, select PoC cameras.

**Week 2–6 — PoC**: YOLOv12 person detection on 3–5 cameras; circular buffer; basic UI; pose-based fall detection.

**Week 6–10 — Expand**: add crowd counting, restricted-area detection, phone detection; integrate notifications.

**Week 10–14 — Face & Emotion**: opt-in enrollment flow; human-review queues for sensitive detections.

**Week 14–16 — Hardening**: MLOps, monitoring, security review, pilot-run.

---

## 17. Example Inference Pseudocode

```python
# pseudocode
import cv2
from yolov12 import YOLOv12  # conceptual

model = YOLOv12.load("yolov12_campus.pt")
cap = cv2.VideoCapture("rtsp://cam1.stream")
buffer = CircularBuffer(seconds=20)

while True:
    ret, frame = cap.read()
    if not ret: break
    buffer.push(frame)
    detections = model.detect(frame)
    tracked = tracker.update(detections)
    fall = fall_detector.check(tracked)
    if fall:
        clip = buffer.dump(pre_seconds=10, post_seconds=20)
        clip_url = upload_clip(clip)
        notify({
          "event":"fall_detected",
          "camera":"BuildingA_Cam12",
          "clip":clip_url,
          "snapshot":save_snapshot(frame)
        })
```

---

## 18. Risk Areas & Mitigations

- **False positives** (emotion/unethical): human review & conservative thresholds.
- **Face recognition bias**: diverse test sets, opt-in policy.
- **Legal/privacy objections**: signage, transparency, legal review.

---

## 19. KPI Targets (Example)

- Fall detection recall ≥ 98% (after tuning).
- Crowd detection MAE ≤ 5 for small crowds (<50).
- Alert latency (camera→notification) < 3s for critical events.

---

## 20. Tools & Libraries

- YOLOv12 (custom), PyTorch, OpenCV, TensorRT, DeepSort, OpenPose/MediaPipe, FastAPI, React, Mapbox, CVAT for labeling.

---

## 21. Next Actions (Pick one — I will generate it immediately)

- **A.** Detailed label schema & data labelling plan for each feature.
- **B.** Full API spec + Postgres DB schema with sample SQL.
- **C.** Edge deployment scripts & Dockerfiles for Jetson/Ubuntu + YOLOv12.
- **D.** Starter React admin UI mockups & component list.
- **E.** Sample YOLOv12 training config & fine-tune commands.


---

*Prepared on 2025-11-12.*

