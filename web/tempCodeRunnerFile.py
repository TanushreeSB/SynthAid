from flask import Flask, request, render_template
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)

# Load the trained model
model = load_model('/Users/anshulshukla/Desktop/oral cancer detection/models/cancer_detection_model.keras')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (150, 150))  # Resize to match model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def load_metrics(filename='metrics.json'):
    with open(filename, 'r') as f:
        metrics = json.load(f)
    return metrics['accuracy'], metrics['precision'], metrics['recall']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # Ensure the 'static/images' directory exists
        save_directory = 'static/images'
        os.makedirs(save_directory, exist_ok=True)  # Create directory if it doesn't exist

        # Save the file to the 'images' folder
        file_path = os.path.join(save_directory, file.filename)  # Save in static/images
        file.save(file_path)
        
        # Preprocess the image
        img = preprocess_image(file_path)
        
        # Make predictions
        predictions = model.predict(img)
        
        # Invert the prediction logic
        if predictions[0][0] > 0.5:  # Assuming binary classification
            result = "Non-Cancer"   # Flip the label for predictions above 0.5
        else:
            result = "Cancer"       # Flip the label for predictions below or equal to 0.5

        # Load accuracy, precision, and recall from metrics JSON
        accuracy, precision, recall = load_metrics()
        
        # Render the result template with prediction results and metrics
        return render_template('result.html', result=result, filename=file.filename, accuracy=accuracy, precision=precision, recall=recall)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
