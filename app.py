from flask import Flask, request
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

app = Flask(__name__)

# Fruit classes
classes = [
    'ripe apple', 'ripe banana', 'ripe dragon', 'ripe grapes', 'ripe lemon',
    'ripe mango', 'ripe orange', 'ripe papaya', 'ripe pineapple', 'ripe pomegranate',
    'ripe strawberry', 'unripe apple', 'unripe banana', 'unripe dragon',
    'unripe grapes', 'unripe lemon', 'unripe mango', 'unripe orange',
    'unripe papaya', 'unripe pineapple', 'unripe pomegranate', 'unripe strawberry'
]

# Load model
model = tf.keras.models.load_model("best_fruit_model_resnet50v2.h5")

# HOME = Upload Form
@app.route("/", methods=["GET"])
def home():
    return """
    <h2>Upload a Fruit Image</h2>
    <form action="/predict" method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*">
        <button type="submit">Predict</button>
    </form>
    """

# Prediction Route (Top 3)
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(file).resize((224, 224))
    img_array = np.array(img)

    # Preprocess for ResNet50V2
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]  # 1D array of predictions
    top_indices = pred.argsort()[-3:][::-1]  # top 3 indices
    top_results = [(classes[i], float(pred[i])*100) for i in top_indices]  # class + confidence %

    # Build HTML to display top 3 predictions
    results_html = "<h2>Top 3 Predictions</h2><ol>"
    for cls, conf in top_results:
        results_html += f"<li><strong>{cls}</strong> — {conf:.2f}%</li>"
    results_html += "</ol><a href='/'>Predict another image</a>"

    return results_html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
