import pickle
import mlflow
import mlflow.sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("PlantDisease_KNN")

with mlflow.start_run():
    with open("models/knn_features.pkl", "rb") as f:
        X_train, y_train, X_val, y_val, classes = pickle.load(f)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    
    mlflow.log_param("n_neighbors", 5)
    mlflow.log_metric("val_accuracy", acc)
    mlflow.sklearn.log_model(knn, "knn_model")
    
    print(f"✅ KNN validation accuracy: {acc:.4f}")
