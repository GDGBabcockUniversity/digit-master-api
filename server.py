from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import numpy as np

from PIL import Image
import io

app = FastAPI()

# Load the pre-trained TensorFlow model
model = tf.saved_model.load("model")

# Define the input data model
class PredictionRequest(BaseModel):
    data: List[float]


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read the file and convert it to a PIL image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")  # Convert to grayscale
        
        # Preprocess the image to match model input shape (e.g., 28x28 pixels for MNIST)
        image = image.resize((28, 28))
        image_array = np.array(image)
        image_array = image_array / 255.0  # Normalize to [0, 1]
        image_array = image_array.astype('float32')
        image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for model input
        
        # Perform prediction
        prediction = model(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Return the predicted class
        return {"predicted_class": int(predicted_class)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
