
# 🎯 Multi-View Player Tracking with YOLOv11 & DeepSORT

This project implements a high-performance multi-view player tracking pipeline using a fine-tuned YOLOv11 model and DeepSORT. It is capable of detecting and tracking players across multiple camera angles — such as broadcast and tacticam feeds — with consistent ID assignment and exportable tracking metadata.

---

## 🧠 Overview

The goal is to identify and track all players in real-time across two synchronized football match videos. Each detected player is assigned a consistent ID which persists across frames. The system outputs both annotated video and structured JSON logs for downstream analytics, tactical modeling, or video indexing.

This work is optimized for GPU execution (e.g., Google Colab, CUDA-enabled systems), and is suitable for large-scale sports analytics, broadcast augmentation, or coaching tools.

---

## 🚀 Key Features

- ✅ Real-time detection with fine-tuned **YOLOv11** on custom-trained player/ball dataset
- ✅ Robust **DeepSORT** tracking with re-identification and Kalman filtering
- ✅ Tight, persistent bounding boxes even under occlusion
- ✅ Annotated video output (`.mp4`) at high resolution (e.g., 1080p)
- ✅ Frame-by-frame structured JSON output (`.json`) with track metadata
- ✅ Easily extendable to ball tracking or player re-ID across cameras

---

## 🧰 Tech Stack

| Component    | Technology         |
|--------------|--------------------|
| Detection    | YOLOv11 (Ultralytics) |
| Tracking     | DeepSORT (Real-time) |
| Video I/O    | OpenCV              |
| Execution    | Google Colab / CUDA |
| Export       | MP4 (video), JSON (tracks) |

---

## 📁 Directory Layout

\`\`\`bash
.
├── broadcast.mp4                 # Input broadcast video
├── tacticam.mp4                  # Input top-down or tactical view
├── best.pt                       # Fine-tuned YOLOv11 weights
├── tracker.py                    # Core tracking script
├── /output/
│   ├── broadcast_tracked.mp4     # Output annotated video
│   ├── broadcast_tracks.json     # Frame-by-frame tracking data
│   ├── tacticam_tracked.mp4
│   └── tacticam_tracks.json
└── README.md
\`\`\`

---

## 📦 Setup Instructions (Colab / Local GPU)

### 1. Install Dependencies

\`\`\`bash
# YOLOv11 (Ultralytics)
!git clone https://github.com/ultralytics/ultralytics
%cd ultralytics
!pip install -e .

# DeepSORT (lightweight version)
!pip install deep_sort_realtime
\`\`\`

### 2. Load Model

\`\`\`python
from ultralytics import YOLO
model = YOLO('/content/drive/MyDrive/internship/best.pt')  # Your trained weights
\`\`\`

---

## 🧩 Usage

### Run Tracking on a Video

\`\`\`python
from tracker import run_tracker

run_tracker(
    video_path="/content/broadcast.mp4",
    model=model,
    output_path_prefix="/content/drive/MyDrive/try2output/broadcast",
    upscale_to=(1920, 1080)  # Optional: high-res output
)
\`\`\`

---

## 📊 Output Format

### JSON (\`*_tracks.json\`)
Each record represents one player in one frame:

\`\`\`json
{
  "frame": 134,
  "id": 7,
  "bbox": [345, 122, 412, 190]
}
\`\`\`

### Video (\`*_tracked.mp4\`)
- Green box with label \`ID:Player\`
- Output resolution configurable

---

## 📌 Parameters

| Parameter         | Description                           |
|------------------|---------------------------------------|
| \`max_age=60\`     | Frames to keep a track alive           |
| \`n_init=2\`       | Frames needed to confirm a track       |
| \`confidence > 0.2\` | Detection threshold for YOLO boxes     |
| \`upscale_to\`     | Final video resolution (e.g., 1080p)   |

---

## 🧠 Applications

- Multi-angle match analysis
- Player performance tracking
- Broadcast enhancement overlays
- Automated dataset creation for jersey recognition
- Cross-camera re-ID and event detection

---

## 📌 Future Work

- 🎽 Jersey number recognition using OCR
- 🔁 Cross-view ID mapping using temporal + appearance models
- 📈 Player trajectory analysis & heatmap generation
- 🧠 Integrate LSTM for long-term motion smoothing

---

## 📄 License

This project is released under the MIT License. You may use, modify, and redistribute it with proper attribution.

---

## 👤 Author

**Aditya Tak**  
Machine Learning Engineer | Computer Vision | Sports Analytics  
_B.Tech. CSE | Specialization in AI & ML_

📫 [LinkedIn](https://linkedin.com/in/adityatak) | [GitHub](https://github.com/your-username)

---

## ✅ Demo Ready

📌 _Tested on NVIDIA T4 and A100 GPUs on Colab Pro_  
📦 _End-to-end tracking on HD video under 2 minutes_
