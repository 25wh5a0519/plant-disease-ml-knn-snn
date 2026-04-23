import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split

mlflow.set_experiment("PlantDisease_CNN")

BASE_PATH = r"C:\Users\LENOVO\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage"
class_names = sorted([d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))])

# Create temporary train/val folders (we'll split the dataset)
train_dir = "data/raw/train_cnn"
val_dir = "data/raw/val_cnn"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Split images per class
for cls in class_names:
    cls_path = os.path.join(BASE_PATH, cls)
    images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg','.jpeg','.png'))]
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(train_dir, cls, img))
    for img in val_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(val_dir, cls, img))

print("Train/val split created")

img_size = (128,128)
batch_size = 32
epochs = 10

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
val_gen = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

with mlflow.start_run():
    mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "img_size": img_size})
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    
    for epoch in range(epochs):
        mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
    
    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_plant_disease.h5")
    mlflow.tensorflow.log_model(model, "cnn_model")
    print("✅ CNN training complete.")
