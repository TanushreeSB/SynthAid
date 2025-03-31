from flask import Flask, request, render_template
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)

# Load the trained model and segmentation model
model = load_model('C:\Users\Tanushree\Downloads\oral cancer detection copy 2\models\simple_cnn.keras')
segmentation_model = load_model('C:\Users\Tanushree\Downloads\oral cancer detection copy 2\models\segmentation_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))  
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  
    return img

def load_metrics(filename='metrics.json'):
    with open(filename, 'r') as f:
        metrics = json.load(f)
    return metrics['accuracy'], metrics['precision'], metrics['recall']
def generate_overlay(image_path):
    """Generate an overlay with a green box around the cancerous region and magnify the part inside the box."""
    img = cv2.imread(image_path)
    original_size = img.shape[:2] 
    img_resized = cv2.resize(img, (150, 150)) 
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    
    mask = segmentation_model.predict(img_input)[0]
    mask_resized = cv2.resize(mask, original_size[::-1])  # Resize mask to original image size
    mask_binary = (mask_resized > 0.5).astype(np.uint8)  # Binarize mask to highlight cancer regions

    # Create overlay with green box around detected regions
    overlay_img = img.copy()  # Make a copy of the original image to work with
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:  # Check if any contours were detected
        for contour in contours:
            # Get bounding box for the cancerous region
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green box around cancerous region

            # Add padding to the bounding box to zoom into a larger area around the cancer
            padding = 40  # Padding to zoom into a slightly larger area around the cancer (adjust as necessary)
            x_padded = max(0, x - padding)
            y_padded = max(0, y - padding)
            w_padded = min(original_size[1], x + w + padding) - x_padded
            h_padded = min(original_size[0], y + h + padding) - y_padded

            # Crop the region around the cancer with padding
            cropped_region = img[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]

            # Magnify the cropped region (you can increase the magnification factor as needed)
            magnified_region = cv2.resize(cropped_region, (w_padded * 3, h_padded * 3))  # Magnify 3 times

            # Calculate target position for overlay
            magnified_overlay_y = y_padded
            magnified_overlay_x = x_padded

            # Ensure the magnified region fits within the bounds of the original image
            target_height = original_size[0] - magnified_overlay_y
            target_width = original_size[1] - magnified_overlay_x

            # Resize magnified region if it exceeds the available space
            magnified_height, magnified_width = magnified_region.shape[:2]
            if magnified_height > target_height or magnified_width > target_width:
                magnified_region = cv2.resize(magnified_region, (target_width, target_height))

            # Adjust magnified region shape to fit precisely within the overlay bounds
            overlay_img[magnified_overlay_y:magnified_overlay_y + magnified_region.shape[0], 
                        magnified_overlay_x:magnified_overlay_x + magnified_region.shape[1]] = magnified_region
            cv2.rectangle(overlay_img, 
                          (magnified_overlay_x, magnified_overlay_y), 
                          (magnified_overlay_x + magnified_region.shape[1], magnified_overlay_y + magnified_region.shape[0]), 
                          (0, 255, 0), 2)  # Green box around magnified area

    # Save overlay image
    overlay_filename = 'overlay_' + os.path.basename(image_path)
    overlay_path = os.path.join('/Users/anshulshukla/Desktop/oral cancer detection copy/web/static/images 2', overlay_filename)
    cv2.imwrite(overlay_path, overlay_img)

    return overlay_filename



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # Ensure the 'static/images 2' directory exists
        save_directory = '/Users/anshulshukla/Desktop/oral cancer detection copy/web/static/images 2'
        os.makedirs(save_directory, exist_ok=True)  # Create directory if it doesn't exist

        # Save the file to the 'images 2' folder
        file_path = os.path.join(save_directory, file.filename)
        file.save(file_path)
        
        # Preprocess the image
        img = preprocess_image(file_path)
        
        # Make predictions
        predictions = model.predict(img)
        
        # Flip the prediction logic
        if predictions[0][0] > 0.5:  # Assuming binary classification
            result = "Non-Cancer"  # Non-Cancer
            overlay_filename = None
        else:
            result = "Cancer"  # Cancer
            overlay_filename = generate_overlay(file_path)

        # Load accuracy, precision, and recall from metrics JSON
        accuracy, precision, recall = load_metrics()
        
        # Render the result template with prediction results and metrics
        return render_template('result.html', result=result, filename=file.filename, accuracy=accuracy, precision=precision, recall=recall, overlay_filename=overlay_filename)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

