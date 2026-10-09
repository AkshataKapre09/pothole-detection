#!/usr/bin/env python
"""Diagnostic script to test YOLO model loading."""

import os
import sys

print("=" * 60)
print("YOLO Model Diagnostic Test")
print("=" * 60)

# Check working directory
print(f"\n✓ Current directory: {os.getcwd()}")
print(f"✓ Files in directory:")
for f in os.listdir("."):
    size = os.path.getsize(f) if os.path.isfile(f) else "DIR"
    print(f"  - {f}: {size}")

# Check if pothole_best.pt exists
print(f"\n📁 Model file checks:")
if os.path.exists("pothole_best.pt"):
    size_mb = os.path.getsize("pothole_best.pt") / (1024 * 1024)
    print(f"  ✓ pothole_best.pt found ({size_mb:.2f} MB)")
else:
    print(f"  ✗ pothole_best.pt NOT found")

if os.path.exists("yolov8n.pt"):
    size_mb = os.path.getsize("yolov8n.pt") / (1024 * 1024)
    print(f"  ✓ yolov8n.pt found ({size_mb:.2f} MB)")
else:
    print(f"  ✗ yolov8n.pt NOT found (will be downloaded)")

# Test YOLO loading
print(f"\n🤖 Loading YOLO model...")
try:
    from ultralytics import YOLO
    
    # Try pothole model first
    if os.path.exists("pothole_best.pt"):
        print("  → Attempting to load pothole_best.pt...")
        model = YOLO("pothole_best.pt")
        print("  ✓ pothole_best.pt loaded successfully")
    else:
        print("  → pothole_best.pt not found, falling back to yolov8n...")
        model = YOLO("yolov8n.pt")
        print("  ✓ yolov8n.pt loaded successfully")
    
    # Test inference
    print(f"\n🔍 Testing model inference...")
    import numpy as np
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.predict(source=dummy_img, conf=0.25, verbose=False)
    print(f"  ✓ Inference successful")
    print(f"  ✓ Model names/classes: {model.names if hasattr(model, 'names') else 'N/A'}")
    print(f"  ✓ Detection boxes in test: {len(results[0].boxes)}")
    
    print("\n✅ All tests passed! Model is working.")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
