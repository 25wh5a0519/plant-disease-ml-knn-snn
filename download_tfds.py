import tensorflow_datasets as tfds
import os
import shutil
from PIL import Image
import numpy as np

# Download PlantVillage dataset using TFDS
print("Downloading PlantVillage dataset (approx 1.1 GB)...")
dataset, info = tfds.load('plant_village', split='train', with_info=True, as_supervised=True)

# Class names
class_names = info.features['label'].names
print(f"Classes: {class_names}")

# Create directories
train_dir = 'data/raw/train'
val_dir = 'data/raw/val'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Create class subdirectories
for cls in class_names:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

# Split dataset (80% train, 20% val)
total = info.splits['train'].num_examples
train_count = int(0.8 * total)
val_count = total - train_count

# Iterate through dataset and save images
for i, (image, label) in enumerate(tfds.as_numpy(dataset)):
    # Convert to PIL Image
    img = Image.fromarray(image.astype('uint8'))
    class_name = class_names[label]
    
    if i < train_count:
        dest_dir = os.path.join(train_dir, class_name)
    else:
        dest_dir = os.path.join(val_dir, class_name)
    
    img_path = os.path.join(dest_dir, f'img_{i}.jpg')
    img.save(img_path)
    
    if (i+1) % 1000 == 0:
        print(f'Processed {i+1}/{total} images')

print(f'Dataset ready at data/raw/train and data/raw/val')
