# -*- coding: utf-8 -*-
"""
Garbage Classifier - Advanced Data Exploration
Phân tích kỹ lưỡng dataset trước khi training
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 1. SETUP & CONFIG
PROJECT_PATH = '/content/drive/MyDrive/Garbage_Classifier'
DATA_PATH = os.path.join(PROJECT_PATH, 'dataset')
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

print("="*70)
print("ADVANCED DATA EXPLORATION - GARBAGE CLASSIFIER")
print("="*70)

# 2. PRIORITY 1: CLASS DISTRIBUTION ANALYSIS
print("\n" + "="*70)
print("PRIORITY 1: CLASS DISTRIBUTION ANALYSIS")
print("="*70)

class_names = sorted(os.listdir(DATA_PATH))
num_classes = len(class_names)
class_counts = {}
class_stats = {}

# Count images per class
for class_name in class_names:
    class_path = os.path.join(DATA_PATH, class_name)
    if os.path.isdir(class_path):
        count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))])
        class_counts[class_name] = count
        class_stats[class_name] = {'count': count, 'issues': []}

total_images = sum(class_counts.values())
min_class = min(class_counts.values())
max_class = max(class_counts.values())
imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')

print(f"\nClass Distribution:")
print(f"  Total classes: {num_classes}")
print(f"  Total images: {total_images}")
print(f"  Min images/class: {min_class}")
print(f"  Max images/class: {max_class}")
print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")

if imbalance_ratio > 2.0:
    print(f"WARNING: Dataset imbalanced! Ratio > 2.0x")
else:
    print(f"Dataset well-balanced!")

print(f"\nPer-Class Breakdown:")
for class_name in sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True):
    count = class_counts[class_name]
    pct = (count / total_images) * 100
    bar = "" * int(pct / 2)
    print(f"  {class_name:15} | {count:4} images | {pct:5.1f}% {bar}")

# Plot class distribution
fig, ax = plt.subplots(figsize=(12, 5))
classes_sorted = sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True)
counts_sorted = [class_counts[c] for c in classes_sorted]
colors = ["#DC5151" if count < (total_images/num_classes)*0.8 else '#4ECDC4' for count in counts_sorted]

bars = ax.bar(classes_sorted, counts_sorted, color=colors)
ax.axhline(y=total_images/num_classes, color='r', linestyle='--', label=f'Mean ({total_images/num_classes:.0f})', linewidth=2)
ax.set_xlabel('Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
ax.set_title('Class Distribution - Garbage Classification Dataset', fontsize=14, fontweight='bold')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_PATH, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()

# 3. PRIORITY 1: IMAGE QUALITY CHECK
print("\n" + "="*70)
print("PRIORITY 1: IMAGE QUALITY CHECK")
print("="*70)

quality_report = {
    'corrupted': [],
    'too_small': [],
    'too_large': [],
    'grayscale': [],
    'formats': defaultdict(int)
}

MIN_SIZE = 50
MAX_SIZE = 5000

for class_name in class_names:
    class_path = os.path.join(DATA_PATH, class_name)
    for filename in os.listdir(class_path):
        filepath = os.path.join(class_path, filename)
        
        try:
            img = Image.open(filepath)
            width, height = img.size
            
            # Check size
            if width < MIN_SIZE or height < MIN_SIZE:
                quality_report['too_small'].append((class_name, filename, f"{width}x{height}"))
                class_stats[class_name]['issues'].append(f"Too small: {width}x{height}")
            
            if width > MAX_SIZE or height > MAX_SIZE:
                quality_report['too_large'].append((class_name, filename, f"{width}x{height}"))
                class_stats[class_name]['issues'].append(f"Too large: {width}x{height}")
            
            # Check if grayscale
            if img.mode != 'RGB' and img.mode != 'RGBA':
                quality_report['grayscale'].append((class_name, filename, img.mode))
                class_stats[class_name]['issues'].append(f"Not RGB: {img.mode}")
            
            # Track format
            quality_report['formats'][img.format] += 1
            
        except Exception as e:
            quality_report['corrupted'].append((class_name, filename, str(e)))
            class_stats[class_name]['issues'].append(f"Corrupted: {str(e)}")

print(f"\nQuality Check Results:")
print(f"  Corrupted images: {len(quality_report['corrupted'])}")
print(f"  Too small (<{MIN_SIZE}px): {len(quality_report['too_small'])}")
print(f"  Too large (>{MAX_SIZE}px): {len(quality_report['too_large'])}")
print(f"  Grayscale/Non-RGB:{len(quality_report['grayscale'])}")

if len(quality_report['corrupted']) > 0:
    print(f"\nCORRUPTED IMAGES:")
    for class_name, filename, error in quality_report['corrupted'][:5]:
        print(f"{class_name}/{filename}: {error}")

if len(quality_report['too_small']) > 0:
    print(f"\nTOO SMALL IMAGES:")
    for class_name, filename, size in quality_report['too_small'][:5]:
        print(f"{class_name}/{filename}: {size}")

print(f"\nImage Formats:")
for fmt, count in quality_report['formats'].items():
    print(f"{fmt}: {count} images")

# 4. PRIORITY 1: PIXEL STATISTICS
print("\n" + "="*70)
print("PRIORITY 1: PIXEL STATISTICS ANALYSIS")
print("="*70)

pixel_stats = {}

for class_name in class_names:
    class_path = os.path.join(DATA_PATH, class_name)
    images_array = []
    
    for filename in os.listdir(class_path):
        filepath = os.path.join(class_path, filename)
        try:
            img = Image.open(filepath).convert('RGB')
            img_resized = img.resize(IMAGE_SIZE)
            img_array = np.array(img_resized)/255.0
            images_array.append(img_array)
        except:
            pass
    
    if images_array:
        images_array = np.array(images_array)
        pixel_stats[class_name] = {
            'mean': np.mean(images_array, axis=(0, 1, 2)),
            'std': np.std(images_array, axis=(0, 1, 2)),
            'min': np.min(images_array),
            'max': np.max(images_array),
        }

print(f"\nPixel Statistics (normalized 0-1):")
print(f"{'Class':<15} {'Mean RGB':<25} {'Std RGB':<25}")
print("-" * 65)
for class_name in sorted(class_names):
    if class_name in pixel_stats:
        mean_rgb = pixel_stats[class_name]['mean']
        std_rgb = pixel_stats[class_name]['std']
        print(f"{class_name:<15} {str(np.round(mean_rgb, 3)):<25} {str(np.round(std_rgb, 3)):<25}")

# Plot pixel distributions
fig, axes = plt.subplots(num_classes, 1, figsize=(12, 3*num_classes))
if num_classes == 1:
    axes = [axes]

for idx, class_name in enumerate(sorted(class_names)):
    class_path = os.path.join(DATA_PATH, class_name)
    pixels = []
    
    for filename in os.listdir(class_path)[:20]:  # Sample 20 images
        filepath = os.path.join(class_path, filename)
        try:
            img = Image.open(filepath).convert('RGB')
            img_array = np.array(img.resize(IMAGE_SIZE)).flatten()
            pixels.extend(img_array)
        except:
            pass
    
    axes[idx].hist(pixels, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[idx].set_title(f'{class_name} - Pixel Value Distribution', fontweight='bold')
    axes[idx].set_xlabel('Normalized Pixel Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_PATH, 'pixel_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()

# 5. PRIORITY 2: SAMPLE VISUALIZATION (Original + Augmented)
print("\n" + "="*70)
print("PRIORITY 2: SAMPLE VISUALIZATION (Original + Augmented)")
print("="*70)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=35,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    channel_shift_range=20.0,
    horizontal_flip=True,
    brightness_range=[0.75, 1.15],
    fill_mode='nearest',
    validation_split=0.2,
    vertical_flip=True
)

fig, axes = plt.subplots(num_classes, 7, figsize=(16, 3*num_classes))
if num_classes == 1:
    axes = [axes]

for class_idx, class_name in enumerate(sorted(class_names)):
    class_path = os.path.join(DATA_PATH, class_name)
    
    # Get first valid image
    for filename in os.listdir(class_path):
        filepath = os.path.join(class_path, filename)
        try:
            img = Image.open(filepath).convert('RGB')
            img_resized = img.resize(IMAGE_SIZE)
            img_array = np.array(img_resized)
            
            # Original image
            axes[class_idx, 0].imshow(img_array)
            axes[class_idx, 0].set_title(f'{class_name}\n(Original)', fontweight='bold')
            axes[class_idx, 0].axis('off')
            
            # Augmented versions
            img_array_3d = np.expand_dims(img_array, axis=0)
            
            for aug_idx in range(6):
                aug_data = train_datagen.flow(img_array_3d, batch_size=1)
                aug_img = next(aug_data)[0]
                axes[class_idx, aug_idx + 1].imshow(np.clip(aug_img, 0, 1))
                axes[class_idx, aug_idx + 1].set_title(f'Augmented {aug_idx + 1}')
                axes[class_idx, aug_idx + 1].axis('off')
            
            break
        except:
            continue

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_PATH, 'augmentation_samples.png'), dpi=150, bbox_inches='tight')
plt.show()

# 6. PRIORITY 2: DATA GENERATORS STATISTICS
print("\n" + "="*70)
print("PRIORITY 2: DATA GENERATORS STATISTICS")
print("="*70)

validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\nData Generators Created:")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {validation_generator.samples}")
print(f"  Number of classes: {num_classes}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Class indices: {train_generator.class_indices}")

# Batch statistics
x_batch, y_batch = next(train_generator)
print(f"\nBatch Statistics (First Batch):")
print(f"  Batch shape: {x_batch.shape}")
print(f"  Pixel range: [{x_batch.min():.3f}, {x_batch.max():.3f}]")
print(f"  Pixel mean: {x_batch.mean():.3f}, std: {x_batch.std():.3f}")
print(f"  Class distribution in batch:")
for class_name, class_idx in train_generator.class_indices.items():
    count = np.sum(np.argmax(y_batch, axis=1) == class_idx)
    print(f"    {class_name}: {count}")

# 7. PRIORITY 2: PER-CLASS QUALITY REPORT
print("\n" + "="*70)
print("PRIORITY 2: PER-CLASS QUALITY REPORT")
print("="*70)

print(f"\n{'Class':<15} {'Count':<8} {'Issues':<30}")
print("-" * 53)
for class_name in sorted(class_names):
    count = class_counts[class_name]
    issues = len(class_stats[class_name]['issues'])
    issues_str = f"{issues} issue(s)" if issues > 0 else "OK"
    print(f"{class_name:<15} {count:<8} {issues_str:<30}")
print("\n" + "="*70)