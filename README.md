# ANPR Backend

**Production-grade Automatic Number Plate Recognition backend built with FastAPI, YOLO, PaddleOCR, and MongoDB.**

---

## Architecture Overview

```
Request → FastAPI Route → Service Layer → Repository → MongoDB
                        ↓
              Vehicle Detector (YOLO)
                        ↓
              Plate Detector (YOLO)
                        ↓
              Image Preprocessor (OpenCV)
                        ↓
              OCR Service (PaddleOCR)
                        ↓
              Validator + Corrector
                        ↓
              Storage Service (filesystem)
                        ↓
              Repository (MongoDB Motor)
```

---

## Project Structure

```
alpr_backend/
├── app/
│   ├── api/
│   │   └── routes.py              # Route definitions only — no business logic
│   ├── core/
│   │   ├── config.py              # Pydantic-settings configuration
│   │   ├── logging.py             # Colorised console + rotating file logging
│   │   └── constants.py           # Immutable application constants
│   ├── database/
│   │   ├── mongodb.py             # Motor async client lifecycle
│   │   └── collections.py        # Collection names + index bootstrapping
│   ├── models/
│   │   └── plate_models.py        # MongoDB document models
│   ├── schemas/
│   │   ├── request.py             # API request schemas
│   │   └── response.py            # API response schemas
│   ├── services/
│   │   ├── detector.py            # Vehicle detection (YOLO)
│   │   ├── plate_detector.py      # Plate detection (YOLO)
│   │   ├── image_preprocessor.py  # OpenCV enhancement pipeline
│   │   ├── ocr_service.py         # PaddleOCR wrapper
│   │   ├── validator.py           # Regex validation + OCR correction
│   │   ├── storage_service.py     # Filesystem image persistence
│   │   └── tracking_service.py    # Phase 2 stub (ByteTrack/DeepSORT)
│   ├── utils/
│   │   ├── image_utils.py         # Pure image helper functions
│   │   ├── regex_utils.py         # Indian plate regex helpers
│   │   ├── correction_utils.py    # Position-aware OCR correction
│   │   └── timer.py               # Perf-counter timer utility
│   ├── repositories/
│   │   └── plate_repository.py    # MongoDB data access layer
│   ├── dependencies/
│   │   └── providers.py           # FastAPI Depends() providers
│   ├── middleware/
│   │   └── timing.py              # X-Process-Time-Ms response header
│   ├── startup/
│   │   └── lifespan.py            # Startup/shutdown lifecycle handler
│   └── main.py                    # FastAPI application factory
├── uploads/
│   ├── plates/                    # Cropped plate images (UUID filenames)
│   └── vehicles/                  # Cropped vehicle images
├── weights/
│   ├── vehicle_yolo.pt            # ← Place your vehicle YOLO model here
│   └── license_plate_detector.pt  # ← Place your plate detector model here
├── tests/                         # Test suite (Phase 1+)
├── logs/                          # Rotating log files
├── .env                           # Local environment (gitignored)
├── .env.example                   # Template for .env
├── requirements.txt
├── run.py                         # Production entry point
└── README.md
```

---

## Prerequisites

- Python 3.11+
- MongoDB 6.0+ (local or Atlas)
- (Optional) CUDA-capable GPU for faster inference

---

## Installation

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd alpr_backend
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

**CPU build (default):**
```bash
pip install -r requirements.txt
```

**GPU build (CUDA):**
```bash
# Replace paddlepaddle with the GPU variant
pip install paddlepaddle-gpu
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env to match your environment
nano .env
```

Key variables to update:
```
MONGO_URI=mongodb://localhost:27017
OCR_USE_GPU=false        # Set true if GPU available
OCR_MIN_CONFIDENCE=0.60  # Adjust per your dataset
```

### 5. Place model weights

```bash
# Vehicle detection model (YOLOv8/YOLOv11 trained on COCO or custom)
cp /path/to/your/vehicle_yolo.pt weights/vehicle_yolo.pt

# License plate detector (trained specifically for plates)
# Recommended: https://github.com/nickmuchi/yolo-v8-license-plate-detection
cp /path/to/your/license_plate_detector.pt weights/license_plate_detector.pt
```

> **Note:** If model weights are not present, the system gracefully degrades:
> - Vehicle detector falls back to treating the full image as the vehicle region.
> - Plate detector falls back to treating the full vehicle crop as the plate region.
> OCR will still run on the full image, which is useful for development.

### 6. Start MongoDB

```bash
# Local MongoDB
mongod --dbpath /data/db

# Or via Docker
docker run -d -p 27017:27017 --name mongodb mongo:6.0
```

---

## Running the Server

### Development

```bash
python run.py
```

Or with auto-reload:
```bash
RELOAD=true python run.py
```

### Production

```bash
# Single worker via run.py
python run.py

# Multi-worker via Gunicorn (recommended for production)
pip install gunicorn
gunicorn app.main:app \
    -k uvicorn.workers.UvicornWorker \
    -w 4 \
    --bind 0.0.0.0:8000 \
    --access-logfile logs/access.log
```

---

## API Reference

### Base URL
```
http://localhost:8000/api/v1
```

### Swagger UI
```
http://localhost:8000/docs
```

### ReDoc
```
http://localhost:8000/redoc
```

---

### `GET /api/v1/health`

Health check — verifies MongoDB connectivity.

**Response:**
```json
{
  "status": "ok",
  "app_version": "1.0.0",
  "mongodb": "connected",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

---

### `POST /api/v1/detect/image`

Upload a vehicle image for ANPR processing.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | Vehicle image (JPG, PNG, BMP, WEBP, TIFF, max 20 MB) |
| `camera_id` | string | ❌ | CCTV camera identifier |

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect/image" \
  -F "file=@/path/to/vehicle.jpg" \
  -F "camera_id=CAM_001"
```

**Success Response (`200 OK`):**
```json
{
  "status": "SUCCESS",
  "plate_number": "MH12AB1234",
  "confidence": 0.9241,
  "vehicle": {
    "detected": true,
    "vehicle_type": "car",
    "confidence": 0.8912,
    "bounding_box": { "x1": 12, "y1": 45, "x2": 640, "y2": 480 }
  },
  "plate_detection": {
    "detected": true,
    "confidence": 0.9103,
    "bounding_box": { "x1": 200, "y1": 350, "x2": 440, "y2": 420 }
  },
  "ocr": {
    "raw_text": "MH12AB1234",
    "confidence": 0.9241
  },
  "storage": {
    "plate_image_id": "uuid-...",
    "plate_record_id": "uuid-...",
    "image_path": "uploads/plates/abc123.jpg"
  },
  "camera_id": "CAM_001",
  "processing_time_ms": 342.17,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Status Codes:**

| Status | Meaning |
|--------|---------|
| `SUCCESS` | Plate detected, validated, and saved |
| `LOW_CONFIDENCE` | OCR confidence below threshold |
| `INVALID_PLATE_FORMAT` | OCR text failed Indian plate regex |
| `NO_VEHICLE_DETECTED` | No vehicle found in image |
| `NO_PLATE_DETECTED` | Vehicle found but no plate detected |
| `ERROR` | Unhandled internal error |

---

## MongoDB Schema

### `plate_images` collection
```json
{
  "_id": "uuid-string",
  "plate_number": "MH12AB1234",
  "image_path": "uploads/plates/abc.jpg",
  "vehicle_type": "car",
  "timestamp": "2024-01-01T12:00:00Z",
  "confidence": 0.9241,
  "camera_id": "CAM_001"
}
```

### `plate_records` collection
```json
{
  "_id": "uuid-string",
  "plate_number": "MH12AB1234",
  "timestamp": "2024-01-01T12:00:00Z",
  "confidence": 0.9241,
  "camera_id": "CAM_001",
  "raw_ocr_text": "MH12AB1234",
  "vehicle_type": "car",
  "processing_time_ms": 342.17
}
```

---

## OCR Correction Logic

Position-aware character substitution based on Indian plate format `XX 00 XX 0000`:

| Position | Expected | Correction Applied |
|----------|----------|--------------------|
| State (1–2) | Letters | `0→O`, `1→I`, `5→S`, `8→B`, `2→Z` |
| District (3–4) | Digits | `O→0`, `I→1`, `S→5`, `B→8`, `Z→2` |
| Series (5–7) | Letters | `0→O`, `1→I`, `5→S`, `8→B`, `2→Z` |
| Number (8–11) | Digits | `O→0`, `I→1`, `S→5`, `B→8`, `Z→2` |

---

## Phase 2 Roadmap

- [ ] `POST /detect/video` — MP4 video file processing
- [ ] `POST /detect/stream` — RTSP/CCTV stream processing
- [ ] ByteTrack / DeepSORT integration (`tracking_service.py`)
- [ ] Duplicate suppression (same vehicle, consecutive frames)
- [ ] WebSocket endpoint for real-time stream results

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | `development` | `development` \| `staging` \| `production` |
| `PORT` | `8000` | Server listen port |
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB_NAME` | `anpr_db` | Target database name |
| `OCR_MIN_CONFIDENCE` | `0.60` | Below this → `LOW_CONFIDENCE` |
| `OCR_USE_GPU` | `false` | Enable PaddlePaddle GPU inference |
| `VEHICLE_CONF_THRESHOLD` | `0.45` | YOLO vehicle detection confidence |
| `PLATE_CONF_THRESHOLD` | `0.40` | YOLO plate detection confidence |
| `MAX_UPLOAD_SIZE_MB` | `20` | Maximum upload file size |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
