import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
DATA_DIR = 'labeled_dataset_kaggle'  # Your folder
IMG_HEIGHT = 32
IMG_WIDTH = 128
# --------------

def show_augmented_images():
    # This must match your training code EXACTLY
    aug_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        # Randomly rotate (-5 to +5 degrees)
        transforms.RandomRotation(5, fill=255), 
        # Randomly change brightness
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        # We skip Normalize here so we can see the image clearly
    ])

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    if not files:
        print("No images found!")
        return

    # Pick the first image and augment it 5 times
    img_path = os.path.join(DATA_DIR, files[0])
    original = Image.open(img_path).convert('L')

    plt.figure(figsize=(10, 5))
    
    # Show Original
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')

    # Show 5 Augmented Versions
    for i in range(5):
        # Apply the transformation
        aug_tensor = aug_transform(original)
        # Convert back to image format for display
        aug_img = aug_tensor.squeeze(0).numpy()
        
        plt.subplot(2, 3, i+2)
        plt.title(f"Augmented {i+1}")
        plt.imshow(aug_img, cmap='gray')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    show_augmented_images()