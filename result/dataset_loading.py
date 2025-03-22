import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def load_images_from_folder(folder):
    images = []
    labels = []

    # Define the exact subfolder paths
    cancer_path ='/Users/anshulshukla/Desktop/oral cancer detection/images 2/CANCER'
    non_cancer_path = "/Users/anshulshukla/Desktop/oral cancer detection/images 2/NON CANCER"

    # Load cancer images
    if os.path.isdir(cancer_path):
        for img_name in os.listdir(cancer_path):
            img_path = os.path.join(cancer_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (150, 150))
                img = img / 255.0  # Normalize to [0, 1]
                images.append(img)
                labels.append("Cancer")
    else:
        print(f"Warning: Subfolder 'cancer' not found in '{folder}'.")

    # Load non-cancer images
    if os.path.isdir(non_cancer_path):
        for img_name in os.listdir(non_cancer_path):
            img_path = os.path.join(non_cancer_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (150, 150))
                img = img / 255.0  # Normalize to [0, 1]
                images.append(img)
                labels.append("Non-Cancer")
    else:
        print(f"Warning: Subfolder 'NON CANCER' not found at '{non_cancer_path}'.")

    return np.array(images), np.array(labels)

def display_images(images, labels, num_cols=3, max_images_to_display=15):
    num_images = len(images)
    if num_images == 0:
        print("No images to display.")
        return

    # Randomly sample images if there are more than max_images_to_display
    if num_images > max_images_to_display:
        indices = random.sample(range(num_images), max_images_to_display)
        images = [images[i] for i in indices]
        labels = [labels[i] for i in indices]

    num_rows = (max_images_to_display + num_cols - 1) // num_cols
    plt.figure(figsize=(5 * num_cols, 5 * num_rows))

    for i in range(len(images)):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i])  # No need to convert, images are already in [0,1]
        plt.title(labels[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_path = input("Enter the path to your dataset (images folder): ")
    images, labels = load_images_from_folder(dataset_path)
    print(f"Loaded {len(images)} images with labels.")
    if len(images) > 0:
        max_images = int(input("Enter the maximum number of images to display: "))
        display_images(images, labels, max_images_to_display=max_images)
