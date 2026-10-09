# 🚧 Smart Road Pothole Detection System

**AI-powered pothole detection with severity scoring using YOLOv8 for smart city infrastructure**

Author: Akshata Kapre

---

##  Problem

India has over 3 million km of road network. Pothole-related accidents cause thousands of deaths and injuries every year. Manual road inspection is slow, expensive, and inconsistent.

This project builds an automated system that:
- **Detects** potholes from dashcam/road images using YOLOv8
- **Classifies severity** as Minor, Moderate, or Severe
- **Deploys as a web app** for municipality officers to use directly

---

## ✨ Features

✅ **YOLOv8 Deep Learning** - Fast pothole detection model  
✅ **Fallback Detection** - Edge-based pothole detection (dark region finder)  
✅ **Severity Scoring** - Minor/Moderate/Severe classification  
✅ **Streamlit Web App** - Easy-to-use web interface  
✅ **Image & Video Support** - Process single images or videos  
✅ **Dashboard Analytics** - Pothole statistics and business impact  
✅ **Security** - Enhanced `.gitignore` prevents accidental credential leaks  
✅ **Production Ready** - Can be deployed to Streamlit Cloud or Docker  

---

## 🎯 How It Works

### **Detection Pipeline**
1. Upload image/video
2. **Primary Detection**: YOLOv8 model tries to find potholes
3. **Fallback Detection**: If YOLO finds nothing, edge detection finds dark regions
4. **Severity Scoring**: Classify each detection by size and impact
5. **Display Results**: Bounding boxes + severity labels + statistics

### **Model**
- **Base Model**: YOLOv8n (3.2M parameters, 6.2 MB)
- **Training Data**: 130 synthetic road images with potholes
- **Training Method**: Transfer learning (pretrained COCO → fine-tuned on potholes)
- **Device**: CPU or GPU (auto-detected)

---

##  Model Performance

| Metric | Baseline (COCO) | Fine-tuned (Potholes) | Note |
|---|---|---|---|
| **Detection** | Generic objects | Potholes + fallback | Optimized for roads |
| **Speed** | <50ms per image | <100ms per image | CPU-friendly |
| **Accuracy** | 0.6% on potholes | High on synthetic data | Real-world varies |

**Training Details:**
- Dataset: 100 training + 30 validation synthetic road images
- Optimizer: AdamW
- Device: CPU (no GPU required)
- Training Time: ~15 minutes

---

## 🔴🟡🟢 Severity Scoring

Beyond simple detection, each pothole is classified by repair urgency:

| Severity | Condition | Municipal Response |
|---|---|---|
| 🟢 **Minor** | Area < 2% of image | Routine monitoring |
| 🟡 **Moderate** | Area 2-5% of image | Priority repair within 2 weeks |
| 🔴 **Severe** | Area > 5% of image | Emergency response |

---

##  Project Structure

```
pothole-detection-yolov8/
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies (6 packages)
├── train_real_model.py                 # Synthetic dataset + training script
├── test_model.py                       # Model diagnostic tool
├── Dockerfile                          # Docker for cloud deployment
├── Pothole_Smart_Road_Monitoring.ipynb # Full training notebook
├── README.md                           # This file
├── .gitignore                          # Prevent credential leaks
├── pothole_best.pt                     # Fine-tuned model weights (6.2 MB)
├── yolov8n.pt                          # Base model (auto-downloaded)
└── .streamlit/
    └── config.toml                     # Streamlit configuration
```

---

##  Installation & Usage

### **Quick Start**
```bash
# Clone repository
git clone https://github.com/AkshataKapre09/pothole-detection.git
cd pothole-detection-yolov8

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### **Upload & Detect**
1. Navigate to **Image** tab
2. Upload any road image
3. Adjust confidence threshold if needed
4. Click predict
5. View detections with severity labels

### **View Analytics**
- Switch to **Dashboard** tab for pothole statistics
- See severity distribution and business impact

---

##  Tech Stack

- **Model**: YOLOv8n (Ultralytics)
- **App Framework**: Streamlit
- **Computer Vision**: OpenCV
- **Deep Learning**: PyTorch
- **Deployment**: Docker, Streamlit Cloud
- **Image Processing**: Pillow, NumPy

**Requirements** (Minimal):
```
streamlit
ultralytics
opencv-python-headless
numpy
Pillow
requests
```

---

## 🔒 Security Features

### **Sensitive Data Protection**
The `.gitignore` file prevents accidental commits of:
- ✅ `.env` files (API keys, secrets)
- ✅ Credential files (`credentials.json`, `kaggle.json`)
- ✅ AWS/GCP/Azure credentials
- ✅ Private keys (`.pem`, `.key`, `.pfx`)
- ✅ Authentication tokens
- ✅ Streamlit secrets
- ✅ Model weights (`.pt`, `.pth`, `.h5`)
- ✅ Training data (large files)
- ✅ Temporary/cache files

**Best Practice**: Never commit secrets. Use environment variables instead.

---

##  Improving Accuracy

To train on real pothole data instead of synthetic:

1. **Option A**: Use [Kaggle Annotated Potholes Dataset](https://www.kaggle.com/datasets/chitholian/annotated-potholes-dataset)
   ```bash
   python train_real_model.py  # Downloads and trains automatically
   ```

2. **Option B**: Collect your own images and label them with [Roboflow](https://roboflow.com)

3. **Option C**: Merge with [RDD2022 dataset](https://github.com/sekilab/RoadDamageDetector) (47K images from 6 countries)

---

##  Business Impact

| Method | Cost per km | Speed | Monthly Coverage |
|---|---|---|---|
| Manual inspection | ₹2,000-5,000 | 5-10 km/day | 200 km |
| AI dashcam system | ₹50-200 | 50+ km/hour | 5,000+ km |

**Estimated 90% cost reduction** and **25x more road coverage** per month.

---

##  Troubleshooting

### **Model not detecting potholes?**
- Check that `pothole_best.pt` exists (6.2 MB file)
- Lower confidence threshold in the app
- Upload a test image with obvious dark spots
- Fallback edge detection will activate if ML model finds nothing

### **Slow performance?**
- Use CPU inference (already optimized)
- Reduce image resolution
- Lower batch size (not applicable for single images)

### **Installation errors?**
```bash
# Recreate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

##  Future Improvements

1. **Real dataset training** - RDD2022 (47,000 images) for production accuracy
2. **GPS integration** - EXIF metadata for location-based road mapping
3. **Video streaming** - Real-time dashcam processing
4. **Mobile app** - React Native + FastAPI for field inspectors
5. **Edge deployment** - TensorRT optimization for Jetson Nano
6. **Multi-class detection** - Detect cracks, flooding, debris too
7. **Depth estimation** - Stereo cameras for actual pothole depth measurements

---

##  License

MIT License - Free to use and modify

**Last Updated:** October 9, 2026  
**Author:** Akshata Kapre  
**Repository:** [github.com/AkshataKapre09/pothole-detection](https://github.com/AkshataKapre09/pothole-detection)

---
