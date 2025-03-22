import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    images = []
    labels = []

    # Iterate through all files in the specified folder
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        # Check if the file is a valid image
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (150, 150))  # Resize images to match model input shape
            img = img / 255.0  # Normalize images to the range [0, 1]
            images.append(img)

            # Assign labels based on filename; customize as needed
            # Assuming the naming convention indicates cancer or non-cancer
            if "cancer" in img_name.lower():  # Example condition for cancer
                labels.append("Cancer")
            else:
                labels.append("Non-Cancer")

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    dataset_path = input("Enter the path to your dataset (images folder): ")
    images, labels = load_images_from_folder(dataset_path)
    print(f"Loaded {len(images)} images with labels.")
