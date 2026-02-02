# EdgeRunner

**Real-Time Edge AI Vision System on NVIDIA Jetson Orin Nano**

EdgeRunner is a dual-thread computer vision system that runs YOLO object detection and Vision Language Model (VLM) inference entirely on edge hardware — no cloud, no latency, no connectivity dependency.

Built by [Nosis AI](https://github.com/nosis-ai) for industrial safety and autonomous monitoring applications.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EdgeRunner Runtime                        │
│                  (Jetson Orin Nano 8GB)                      │
│                                                             │
│  ┌─────────────────────────┐  ┌───────────────────────────┐ │
│  │  Thread 1: "The Reflex" │  │  Thread 2: "The Brain"    │ │
│  │       (30 FPS)          │  │       (0.1 Hz)            │ │
│  │                         │  │                           │ │
│  │  ┌─────────────────┐   │  │  ┌─────────────────────┐  │ │
│  │  │ YOLO TensorRT   │   │  │  │ VLM (Qwen3-VL 2B)  │  │ │
│  │  │ + ByteTrack     │   │  │  │ + JSON Constrained  │  │ │
│  │  └────────┬────────┘   │  │  │   Decoding (GBNF)   │  │ │
│  │           │            │  │  └──────────┬──────────┘  │ │
│  │           ▼            │  │             ▼             │ │
│  │  ┌─────────────────┐   │  │  ┌─────────────────────┐  │ │
│  │  │  SLA Metrics    │◄──┼──┼──│  Decision Contract  │  │ │
│  │  │  (FPS, P50/P95) │   │  │  │  (Pydantic Schema)  │  │ │
│  │  └─────────────────┘   │  │  └─────────────────────┘  │ │
│  └─────────────────────────┘  └───────────────────────────┘ │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Dashboard: MJPEG Stream + SSE Events + SLA Metrics     │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance

| Metric | Value | Hardware |
|--------|-------|----------|
| Detection FPS | **28+ FPS** | YOLO TensorRT FP16 + ByteTrack |
| VLM Latency | **8-9s** | Qwen3-VL 2B, constrained decode |
| JSON Parse Rate | **100%** | Grammar-enforced (GBNF) |
| VLM Fallback Rate | **0%** | Over 30+ min stability runs |
| Memory Usage | **6.5GB / 8GB** | Stable, no OOM |
| Power Draw | **~25W** | MAXN mode |

---

## Key Innovation: Constrained Decoding

Traditional VLM pipelines rely on prompt engineering and hope the output parses correctly. EdgeRunner uses **decode-time grammar constraints** via llama.cpp's GBNF support to guarantee 100% valid JSON output:

```python
# Traditional approach (unreliable)
response = vlm.generate("Output JSON: {detected: true/false}")
result = json.loads(response)  # May fail

# EdgeRunner approach (guaranteed)
response_format = {
    "type": "json_schema",
    "json_schema": {"schema": DecisionContract.model_json_schema()}
}
# VLM can ONLY output tokens that form valid JSON matching the schema
```

The decision contract is defined as a Pydantic v2 model with strict validation, ensuring every VLM output is machine-actionable.

---

## Project Structure

```
edge-runner/
├── src/                      # Core source
│   ├── main.py               # Triple-thread orchestrator + dashboard
│   ├── cv_loop.py            # YOLO TensorRT + ByteTrack detection loop
│   ├── vlm_client.py         # VLM client with constrained decoding
│   ├── vlm_worker.py         # Async VLM inference worker
│   ├── machina_schema.py     # Pydantic decision contract schema
│   ├── state_machine.py      # FSM for system control states
│   ├── state.py              # Thread-safe shared state
│   ├── server.py             # FastAPI backend + MJPEG stream
│   ├── alert_manager.py      # Event-driven alert system
│   ├── proximity_detector.py # Ground-plane distance estimation
│   ├── bbox_smoother.py      # Kalman-style bounding box stabilizer
│   ├── ttc.py                # Time-to-contact predictor
│   ├── zone_manager.py       # ROI zone management
│   ├── zone_calibrator.py    # VLM-assisted zone calibration
│   └── ...
├── tools/                    # Utilities
│   ├── stability_test.py     # 30-min+ soak test with metrics
│   ├── camera_check.py       # Camera diagnostics + focus tool
│   ├── calibrate_floor.py    # Ground plane calibration
│   └── safety_audit_report.py # Automated audit report generation
├── scripts/                  # Deployment scripts
│   ├── start.sh              # Quick start
│   ├── start_test.sh         # Long-running stability test
│   └── install_services.sh   # systemd service installer
├── configs/                  # Configuration files
│   ├── json.gbnf             # Grammar for constrained JSON decoding
│   ├── bytetrack_stable.yaml # ByteTrack tracker configuration
│   ├── zones.yaml            # Zone definitions
│   └── proximity_config.yaml # Proximity detection parameters
├── services/                 # systemd service files
│   ├── edge-runner.service   # Main application service
│   └── llama-server.service  # VLM inference server
├── tests/                    # Test suite
└── requirements.txt
```

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| **Device** | NVIDIA Jetson Orin Nano 8GB |
| **JetPack** | 6.0+ (Ubuntu 22.04, CUDA 12.x) |
| **Power** | MAXN mode (25W) |
| **Camera** | USB3 webcam (720p+ @ 30fps) |
| **Storage** | 32GB+ NVMe recommended |

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/nosis-ai/edge-runner.git
cd edge-runner
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download models (YOLO + VLM)
# YOLO: Export via ultralytics (see docs)
# VLM: Download GGUF model for llama.cpp

# 3. Start VLM service
sudo systemctl start llama-server

# 4. Run EdgeRunner
python -m src.main

# 5. Open dashboard
# http://<DEVICE_IP>:8090/dashboard
```

---

## Modules

### Detection Pipeline (`src/cv_loop.py`)
- YOLO TensorRT inference with FP16 quantization
- ByteTrack multi-object tracking for temporal consistency
- Configurable ROI filtering and exclusion zones

### VLM Decision Engine (`src/vlm_client.py`, `src/machina_schema.py`)
- Async VLM inference via llama.cpp HTTP API
- GBNF grammar-constrained JSON output
- Pydantic v2 decision contracts with runtime validation
- Automatic fallback to safe state on timeout

### Safety Modules
- **Proximity Detection** (`src/proximity_detector.py`): Ground-plane calibrated distance estimation
- **Time-to-Contact** (`src/ttc.py`): Predictive collision timing
- **BBox Smoother** (`src/bbox_smoother.py`): Exponential moving average for flicker reduction
- **Alert Manager** (`src/alert_manager.py`): Event-driven alerting with configurable thresholds

### Observability
- Real-time MJPEG video stream
- Server-Sent Events for live metrics
- SLA tracking: FPS P50/P95, VLM latency, fallback rate
- Automated stability test suite

---

## License

Apache 2.0

---

*Built for a world where AI thinks locally.* 🤖
