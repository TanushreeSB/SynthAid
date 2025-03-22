# metrics.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_true, y_pred):
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="Cancer", average="binary")
    recall = recall_score(y_true, y_pred, pos_label="Cancer", average="binary")
    f1 = f1_score(y_true, y_pred, pos_label="Cancer", average="binary")
    
    # Print the evaluation metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

if __name__ == "__main__":
    # Example usage
    # Replace these lists with your actual data
    y_true = ["Cancer", "Non-Cancer", "Cancer", "Non-Cancer", "Cancer"]
    y_pred = ["Cancer", "Non-Cancer", "Non-Cancer", "Non-Cancer", "Cancer"]
    
    evaluate_model(y_true, y_pred)
