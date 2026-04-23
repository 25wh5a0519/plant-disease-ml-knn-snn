import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pickle

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def build_feature_dataset(data_dir, model):
    features = []
    labels = []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg','.jpeg','.png')):
                img_path = os.path.join(cls_dir, fname)
                feat = extract_features(img_path, model)
                features.append(feat)
                labels.append(class_to_idx[cls])
    return np.array(features), np.array(labels), class_names

if __name__ == "__main__":
    base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    train_dir = "data/raw/small/train"
    val_dir = "data/raw/small/val"
    
    X_train, y_train, classes = build_feature_dataset(train_dir, base_model)
    X_val, y_val, _ = build_feature_dataset(val_dir, base_model)
    
    os.makedirs("models", exist_ok=True)
    with open("models/knn_features.pkl", "wb") as f:
        pickle.dump((X_train, y_train, X_val, y_val, classes), f)
    print(f"✅ Features saved. Train: {len(X_train)}, Val: {len(X_val)}")
