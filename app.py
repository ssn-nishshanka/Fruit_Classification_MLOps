from fastapi import FastAPI, UploadFile
import tensorflow as tf
from PIL import Image
import numpy as np

app = FastAPI()
model = tf.keras.models.load_model("best_fruit_model_resnet50v2.keras")

@app.get("/")
def home():
    return {"message": "Fruit Classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile):
    image = Image.open(file.file).resize((224, 224))
    img_array = np.expand_dims(np.array(image)/255.0, axis=0)
    prediction = np.argmax(model.predict(img_array), axis=1)
    return {"predicted_class": int(prediction[0])}
