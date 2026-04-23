import os
import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
import io
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.neighbors import KNeighborsClassifier

app = FastAPI()

# Load KNN model and features
with open("models/knn_features.pkl", "rb") as f:
    X_train, y_train, X_val, y_val, classes = pickle.load(f)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Load CNN model
cnn_model = tf.keras.models.load_model("models/cnn_plant_disease.h5")
train_dir = "data/raw/small/train"
cnn_classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

# Feature extractor for KNN
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def extract_features_from_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224,224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    return features.flatten()

# SUGGESTIONS – update with your actual class names
suggestions = {
    # Example entries – replace with your own
    "Pepper__bell___Bacterial_spot": "Apply copper-based bactericide. Remove infected leaves.",
    "Pepper__bell___healthy": "No treatment needed. Maintain good practices.",
    "Potato___Early_blight": "Use copper fungicide. Rotate crops. Remove affected leaves.",
    "Potato___healthy": "No treatment needed. Ensure proper soil drainage.",
    # Add your fifth class here
}
default_suggestion = "Consult local agricultural extension office."

def get_suggestion(cls):
    return suggestions.get(cls, default_suggestion)

@app.post("/predict/knn")
async def predict_knn(file: UploadFile = File(...)):
    features = extract_features_from_image(await file.read())
    pred_idx = knn.predict([features])[0]
    disease = classes[pred_idx]
    return {"model": "KNN", "disease": disease, "suggestion": get_suggestion(disease)}

@app.post("/predict/cnn")
async def predict_cnn(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((128,128))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = cnn_model.predict(x)
    idx = np.argmax(preds[0])
    disease = cnn_classes[idx]
    confidence = float(preds[0][idx])
    return {"model": "CNN", "disease": disease, "confidence": confidence, "suggestion": get_suggestion(disease)}

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Plant Disease Detector</title>
    <style>body{font-family:Arial;text-align:center;margin-top:50px;}</style>
    </head>
    <body>
        <h1>🌿 Plant Disease Detection</h1>
        <h2>CNN Model</h2>
        <form action="/predict/cnn" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <button type="submit">Predict with CNN</button>
        </form>
        <h2>KNN Model</h2>
        <form action="/predict/knn" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*">
            <button type="submit">Predict with KNN</button>
        </form>
    </body>
    </html>
    """
