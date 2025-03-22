import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import json
from dataset_loading import load_images_from_folder  # Ensure this is correct based on your file structure

def build_model(input_shape):
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))  # Use Input layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Use sigmoid for binary classification
    return model

def save_metrics_to_json(accuracy, precision, recall, filename='metrics.json'):
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
    with open(filename, 'w') as f:
        json.dump(metrics, f)

def calculate_precision_recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))  # True Positives
    tn = np.sum((y_pred == 0) & (y_true == 0))  # True Negatives
    fp = np.sum((y_pred == 1) & (y_true == 0))  # False Positives
    fn = np.sum((y_pred == 0) & (y_true == 1))  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

if __name__ == "__main__":
    dataset_path = input("Enter the path to your dataset (images folder): ")
    
    # Load dataset
    images, labels = load_images_from_folder(dataset_path)

    # Convert string labels to numerical values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)  # Automatically converts labels

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Define model
    input_shape = (150, 150, 3)  # Example input shape
    model = build_model(input_shape)

    # Compile model
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy:.2f}")

    # Make predictions to calculate precision and recall
    y_pred_probs = model.predict(X_val)
    y_pred = (y_pred_probs > 0.5).astype(int)  # Binarize predictions

    precision, recall = calculate_precision_recall(y_val, y_pred)

    # Save metrics to JSON
    save_metrics_to_json(val_accuracy, precision, recall)

    # Save the model in the recommended Keras format
    model.save('/Users/anshulshukla/Desktop/oral cancer detection/models/cancer_detection_model.keras')

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
