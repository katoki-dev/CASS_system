# Campus AI Safety & Surveillance System (CASS)

AI-first safety and security platform for educational campuses, combining YOLOv12-based video analytics, multi-modal detection, automated alerting, and role-aware web tools.

---

## 🚀 Key Capabilities
- **Real-time video analytics** for crowd density, falls, faces, emotions, mobile-phone usage, unethical behavior, and restricted-area trespass.
- **Auto-clipping engine** that captures pre/post event footage and stores encrypted clips with metadata.
- **Role-based dashboards** for administrators, security, medical responders, and guest stakeholders.
- **Multi-channel notifications** (push, SMS, email, webhooks) with severity-aware escalation paths.
- **Privacy-first design** with opt-in face recognition, configurable retention, and complete audit trails.

---

## 🧱 Architecture Overview
1. **Edge Capture Layer** – ONVIF/RTSP cameras + Jetson/edge GPUs maintain low-latency detections.
2. **Inference Layer** – YOLOv12 and specialized models (pose, emotion, action) optimized with TensorRT.
3. **Backend & Orchestration** – REST/WebSocket APIs, event services, S3-compatible storage, Postgres, Redis.
4. **Notification & Response** – Severity-based workflows and live dispatch tools for security and medical teams.

![Architecture Diagram](docs/architecture-diagram.png)

---

## 📦 Repository Layout
- `PRODUCT_DESCRIPTION.md` – Detailed product vision, feature breakdown, and technical blueprint.
- `campus_ai_surveillance_plan.md` – 21-section implementation plan covering data, models, infrastructure, and SOPs.
- `datasets/fall detection/` – Cloned fall-detection datasets and notebooks for model training and benchmarking.

---

## 🧠 Models & Datasets
- **Object detection:** YOLOv12 pre-trained on COCO/OpenImages, fine-tuned on campus footage.
- **Fall detection:** Pose-based sequence classifiers trained with UR-Fall, UP-Fall, Le2i, and custom datasets.
- **Emotion detection:** AffectNet/FER2013 with conservative thresholds and human-in-the-loop validation.
- **Unethical activity:** SlowFast/I3D models trained on AVA/Kinetics plus curated campus scenarios.

Detailed guidance lives in `PRODUCT_DESCRIPTION.md` §§5–7.

---

## 🛠️ Getting Started (Prototype Workflow)
1. **Clone datasets** (already under `datasets/fall detection/`).
2. **Set up environment**
   ```bash
   conda create -n cass python=3.10
   conda activate cass
   pip install -r requirements.txt  # once generated
   ```
3. **Export models** (example)
   ```bash
   python tools/export_yolov12.py --weights weights/campus_yolov12.pt
   ```
4. **Run edge pipeline**
   ```bash
   python services/edge_node/main.py --config configs/edge.yaml
   ```

> Note: scripts/configs are placeholders—use the implementation plan to scaffold actual modules.

---

## 🗺️ Roadmap Snapshot
- Weeks 0–2: Planning, camera mapping, governance approval.
- Weeks 2–6: PoC for person + fall detection with baseline UI.
- Weeks 6–10: Expand to crowd, restricted-zone, phone detection, alerting stack.
- Weeks 10–14: Face & emotion features with opt-in enrollment and review queues.
- Weeks 14–16: MLOps hardening, security audit, pilot deployment.

Full roadmap lives in `PRODUCT_DESCRIPTION.md` §16.

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
