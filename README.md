# Abnormal Behavior and Stranger Detection System

This repository aims to implement a modular deep learning system designed to detect both **abnormal behaviors** and **strangers** in **real-time surveillance video streams**.

> **Warning**: Only the **stranger detection pipeline** is currently implemented. The abnormal behavior classification component is planned but not fully integrated into the real-time pipeline.

---

## Project Structure

```
project_root/
│
├── face_system/                 # Modular face recognition components
│   ├── __init__.py
│   ├── face_bank.py             # Embedding storage and matching
│   ├── face_detector.py         # Face detection using YOLO + ByteTrack
│   ├── face_mesh.py             # MediaPipe landmark extraction
│   ├── face_processor.py        # Alignment, normalization, filtering
│   ├── face_recognizer.py       # Embedding extractor (e.g., EdgeFace)
│   └── face_tracker.py          # Persistent tracking and ID aging
│
├── utils/                       # Visualization, FPS, video writer, CLI args
│
├── build_face_bank.py           # Script to generate embedding bank from images
├── live_camera.py               # Real-time webcam inference
├── dataset.py                   
├── main.py                      
├── requirements.txt
└── README.md
```

---

## Features

- **Face Detection**: Fast YOLOv11s-based detector with ByteTrack for stable tracking.
- **Face Alignment**: MediaPipe face mesh for 5-point landmark normalization.
- **Face Recognition**: Embedding extraction via pretrained EdgeFace.
- **Stranger Detection**: Unknown faces detected via cosine similarity against known embeddings.
- **Persistent Tracking**: Maintains identity across frames and filters out short-lived detections.

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

> Ensure you have Python 3.8+ and a compatible CUDA-enabled GPU for best performance.

---

### 2. Build the Face Bank
Create a folder `face_imgs/` with face images named in the format `PersonName_1.jpg`, `PersonName_2.jpg`, etc.

Then run:
```bash
python build_face_bank.py \
    --image-folder face_imgs/ \
    --output-path data/face-bank.pkl
```


### 3. Run Live Face Recognition
```bash
python live_camera.py \
    --face-bank-path data/face-bank.pkl \
    --face-detector models/YOLOv11s-face.pt \
    --face-recognizer edgeface_s_gamma_05 \
    --tracker-config bytetrack.yaml \
    --source 0 \
    --rec-threshold 0.5 \
    --min-face-size 80 \
    --blurry-threshold 60 \
    --min-track-age 3 \
    --show-fps True \
    --save-video True \
    --output-path outputs/demo.avi
```

#### Arguments for `live_camera.py`
- `--face-bank-path`: Path to the face bank `.pkl` file.
- `--face-detector`: Path to the YOLOv11 face detection model.
- `--face-recognizer`: Name of the face recognition model (e.g., `edgeface_s_gamma_05`).
- `--tracker-config`: Path to the ByteTrack config YAML file.
- `--source`: Input source for the video stream (0 for webcam).
- `--rec-threshold`: Cosine similarity threshold for recognizing faces.
- `--min-face-size`: Minimum face size to accept for recognition.
- `--blurry-threshold`: Laplacian variance threshold to filter blurry faces.
- `--min-track-age`: Minimum age in frames before a tracked face is displayed.
- `--show-fps`: Show FPS on screen (True/False).
- `--save-video`: Whether to save the output video.
- `--output-path`: Path to save the output video.

---

## Notes
- Default embedding model: `edgeface_s_gamma_05` from [otroshi/edgeface](https://github.com/otroshi/edgeface)
- Real-time performance tested on RTX 2060 with ~20–30 FPS
- Easily extensible: replace recognizer, detector, or add external tracking (e.g., DeepSORT)
- Use `torch.hub.load()` to fetch EdgeFace at runtime (no pip install needed)
