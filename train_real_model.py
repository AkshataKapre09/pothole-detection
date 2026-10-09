#!/usr/bin/env python
"""Create realistic synthetic pothole dataset for training."""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import random
from ultralytics import YOLO
import shutil

print("=" * 70)
print("GENERATING REALISTIC SYNTHETIC POTHOLE DATASET")
print("=" * 70)

# Create dataset structure
yolo_dir = Path("synthetic_potholes")
for split in ['train', 'val']:
    (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
    (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

def create_road_image_with_potholes(img_size=640, num_potholes=None):
    """Create realistic road image with pothole-like dark spots."""
    
    if num_potholes is None:
        num_potholes = random.randint(1, 4)
    
    # Create road-like background (gray with slight variations)
    road_color = (180 + random.randint(-30, 30), 
                  180 + random.randint(-30, 30), 
                  180 + random.randint(-30, 30))
    
    img = Image.new('RGB', (img_size, img_size), road_color)
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Add road texture (cracks, lines)
    for _ in range(random.randint(2, 5)):
        y = random.randint(0, img_size)
        draw.line([(0, y), (img_size, y)], 
                 fill=(100, 100, 100, 50), width=random.randint(1, 3))
    
    labels = []
    
    # Add potholes
    for _ in range(num_potholes):
        # Random center
        cx = random.randint(100, img_size - 100)
        cy = random.randint(100, img_size - 100)
        
        # Random size (potholes are various sizes)
        radius = random.randint(20, 80)
        
        # Dark circular pothole
        pothole_color = (30 + random.randint(-20, 20),
                        30 + random.randint(-20, 20), 
                        30 + random.randint(-20, 20))
        
        # Draw filled circle
        draw.ellipse(
            [(cx - radius, cy - radius), 
             (cx + radius, cy + radius)],
            fill=pothole_color,
            outline=(20, 20, 20)
        )
        
        # Add inner shadow for depth
        inner_radius = int(radius * 0.7)
        draw.ellipse(
            [(cx - inner_radius, cy - inner_radius),
             (cx + inner_radius, cy + inner_radius)],
            fill=(50, 50, 50),
            outline=None
        )
        
        # Random edge damage
        for _ in range(random.randint(3, 8)):
            angle = random.uniform(0, 2*np.pi)
            dist = radius * random.uniform(0.8, 1.0)
            ex = int(cx + dist * np.cos(angle))
            ey = int(cy + dist * np.sin(angle))
            damage_size = random.randint(5, 15)
            draw.ellipse(
                [(ex - damage_size, ey - damage_size),
                 (ex + damage_size, ey + damage_size)],
                fill=(60, 60, 60)
            )
        
        # YOLO format: class, cx_norm, cy_norm, w_norm, h_norm
        cx_norm = cx / img_size
        cy_norm = cy / img_size
        w_norm = (2 * radius) / img_size
        h_norm = (2 * radius) / img_size
        
        labels.append(f"0 {cx_norm:.4f} {cy_norm:.4f} {w_norm:.4f} {h_norm:.4f}")
    
    # Apply slight blur to make it more realistic
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    return np.array(img), labels

print("\n🎨 Generating synthetic training images...")

# Generate training images
num_train = 100
num_val = 30

for i in range(num_train):
    img_array, labels = create_road_image_with_potholes()
    img = Image.fromarray(img_array.astype('uint8'))
    
    img_path = yolo_dir / 'train' / 'images' / f'synthetic_{i:04d}.jpg'
    label_path = yolo_dir / 'train' / 'labels' / f'synthetic_{i:04d}.txt'
    
    img.save(img_path, quality=85)
    label_path.write_text('\n'.join(labels) if labels else '')
    
    if (i + 1) % 25 == 0:
        print(f"  ✓ Generated {i+1}/{num_train} train images")

print(f"  ✓ Total train images: {num_train}")

# Generate validation images
for i in range(num_val):
    img_array, labels = create_road_image_with_potholes()
    img = Image.fromarray(img_array.astype('uint8'))
    
    img_path = yolo_dir / 'val' / 'images' / f'synthetic_val_{i:04d}.jpg'
    label_path = yolo_dir / 'val' / 'labels' / f'synthetic_val_{i:04d}.txt'
    
    img.save(img_path, quality=85)
    label_path.write_text('\n'.join(labels) if labels else '')

print(f"  ✓ Total val images: {num_val}")

# Create YAML config
yaml_content = f"""path: {yolo_dir.absolute()}
train: train/images
val: val/images
nc: 1
names:
  0: pothole
"""

yaml_file = yolo_dir / "data.yaml"
yaml_file.write_text(yaml_content)

print(f"\n📊 Dataset statistics:")
print(f"  - Train: {num_train} images")
print(f"  - Val: {num_val} images")
print(f"  - Total: {num_train + num_val} images")
print(f"  - Classes: 1 (pothole)")

# Train model
print(f"\n🚀 Training YOLOv8n (30 epochs)...")
print(f"  Estimated time: 10-15 minutes on CPU\n")

try:
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(yaml_file),
        epochs=30,
        imgsz=640,
        batch=8,
        device="cpu",
        patience=10,
        verbose=True,  # Show progress
        project=None,
        name="pothole_synthetic_train"
    )
    
    # Save model
    print("\n💾 Saving model...")
    model.save("pothole_best.pt")
    size_mb = os.path.getsize("pothole_best.pt") / (1024 * 1024)
    print(f"✅ Model saved: pothole_best.pt ({size_mb:.2f} MB)")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    shutil.rmtree(yolo_dir, ignore_errors=True)
    shutil.rmtree("pothole_synthetic_train", ignore_errors=True)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nRestart Streamlit to use the new pothole detection model:")
    print("  streamlit run app.py\n")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
