import os
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import mlflow

# CONFIG
MODEL_PATH = "best_fruit_model_resnet50v2.h5"
IMAGES_FOLDER = "images/"
CONFIDENCE_THRESHOLD = 50.0  # percent
CONTINUOUS_MONITORING = False # run once for CI/CD, True for continuous monitoring
INTERVAL_SECONDS = 24 * 3600  # daily
EXPERIMENT_NAME = "Fruit_Classification_Production"

classes = [
    'ripe apple', 'ripe banana', 'ripe dragon', 'ripe grapes', 'ripe lemon',
    'ripe mango', 'ripe orange', 'ripe papaya', 'ripe pineapple', 'ripe pomegranate',
    'ripe strawberry', 'unripe apple', 'unripe banana', 'unripe dragon',
    'unripe grapes', 'unripe lemon', 'unripe mango', 'unripe orange',
    'unripe papaya', 'unripe pineapple', 'unripe pomegranate', 'unripe strawberry'
]

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Set experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# FUNCTIONS
def process_image(file_path):
    """Load and preprocess image"""
    img = Image.open(file_path)
    width, height = img.size
    img_array = np.array(img.resize((224, 224))) / 255.0
    return np.expand_dims(img_array, axis=0), (width, height)

def log_drift(image_path, pred, img_size):
    """Log prediction, image stats, and alerts to MLflow"""
    top_indices = pred[0].argsort()[-3:][::-1]
    top_classes = [classes[i] for i in top_indices]
    top_confidences = [float(pred[0][i])*100 for i in top_indices]

    # Image statistics
    img_array, _ = process_image(image_path)
    mean_color = np.mean(img_array, axis=(0,1,2))
    std_color = np.std(img_array, axis=(0,1,2))
    width, height = img_size

    # Start MLflow run
    with mlflow.start_run(run_name="drift_monitoring"):
        mlflow.log_param("model_version", "v1.0")
        mlflow.log_param("image_name", os.path.basename(image_path))
        mlflow.log_param("num_classes", len(classes))
        mlflow.log_param("image_width", width)
        mlflow.log_param("image_height", height)

        # Log top 3 predictions
        for i, cls in enumerate(top_classes):
            mlflow.log_metric(f"top{i+1}_confidence", top_confidences[i])
            mlflow.log_param(f"top{i+1}_class", cls)

        # Log color stats
        mlflow.log_metric("mean_red", mean_color[0])
        mlflow.log_metric("mean_green", mean_color[1])
        mlflow.log_metric("mean_blue", mean_color[2])
        mlflow.log_metric("std_red", std_color[0])
        mlflow.log_metric("std_green", std_color[1])
        mlflow.log_metric("std_blue", std_color[2])

        # Drift alert
        if top_confidences[0] < CONFIDENCE_THRESHOLD:
            alert_msg = f"ALERT: Low confidence ({top_confidences[0]:.2f}%) for {image_path}"
            print(alert_msg)
            mlflow.log_param("alert", alert_msg)

        print(f"Logged drift metrics for {image_path}")

def monitor_once():
    """Run drift monitoring once over all images"""
    class_counts = {cls: 0 for cls in classes}

    for file in os.listdir(IMAGES_FOLDER):
        file_path = os.path.join(IMAGES_FOLDER, file)
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_array, img_size = process_image(file_path)
            pred = model.predict(img_array)
            predicted_class = classes[int(np.argmax(pred))]
            class_counts[predicted_class] += 1
            log_drift(file_path, pred, img_size)

    # Log class distribution
    with mlflow.start_run(run_name="drift_distribution"):
        for cls, count in class_counts.items():
            mlflow.log_metric(f"class_count_{cls}", count)
    print("Logged class distribution for all images.")

def main():
    if CONTINUOUS_MONITORING:
        while True:
            print("Running drift monitoring...")
            monitor_once()
            print(f"Waiting {INTERVAL_SECONDS} seconds until next run...")
            time.sleep(INTERVAL_SECONDS)
    else:
        monitor_once()

if __name__ == "__main__":
    main()
