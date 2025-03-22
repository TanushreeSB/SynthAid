import os
import cv2
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    labels = []
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (150, 150))  # Resize images
            images.append(img)
            labels.append(img_name)  # Using the image name as label for display
    return images, labels

def display_images(images, labels, num_cols=3):
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols  # Calculate number of rows needed

    # Limit the number of images displayed to avoid large figure sizes
    max_images_to_display = min(num_images, 15)  # Display a maximum of 15 images
    num_rows = (max_images_to_display + num_cols - 1) // num_cols

    plt.figure(figsize=(5 * num_cols, 5 * num_rows))
    for i in range(max_images_to_display):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        plt.title(labels[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    dataset_path = input("Enter the path to your dataset (images folder): ")
    images, labels = load_images_from_folder(dataset_path)
    display_images(images, labels)
