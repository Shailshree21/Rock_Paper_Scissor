from datasets import load_dataset
from PIL import Image
import os

# Load dataset
dataset = load_dataset("Javtor/rock-paper-scissors")

# ✅ Correct label names from dataset metadata
label_names = dataset['train'].features['label'].names

# Prepare base directory
base_dir = "dataset"
splits = ['train', 'test']

def save_images_to_folder(split_name):
    split_dataset = dataset[split_name]
    for idx, sample in enumerate(split_dataset):
        label = label_names[sample['label']]
        image = sample['image']

        # Convert RGBA to RGB
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Create directory
        save_dir = os.path.join(base_dir, split_name, label)
        os.makedirs(save_dir, exist_ok=True)

        # Save image
        image_path = os.path.join(save_dir, f"{split_name}_{label}_{idx}.jpg")
        image.save(image_path)

for split in splits:
    save_images_to_folder(split)

print("✅ Dataset extracted correctly with proper class-label mapping.")
