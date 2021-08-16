import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from exercise_assistant.models.train import ModelTrainer
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from joblib import load

class ModelLoader():

    def __init__(self):
        pass

    def load_knn_model(model_path):
        # Load KNN Model
        pose_classification_model = load(model_path)

        return pose_classification_model


if __name__ == '__main__':
    # Set path variables
    ROOT = os.path.join('exercise_assistant')
    MODEL_PATH = os.path.join(ROOT, 'models', 'classification', 'pose_classifier.joblib')
    ml = ModelLoader()
    # Load Model and Perform Validation (TODO)
    pose_classification_model = ml.load_knn_model(MODEL_PATH)
    print("Load Successful")
    # X_data = np.load(X_PATH)
    # y = np.load(Y_PATH)
    # X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.1, random_state=42)

    # # Predict
    # ypredicted = exercise_recognition_model.predict(X_test)

    # ytrue = np.argmax(y_test, axis=1)
    # ypredicted = np.argmax(ypredicted, axis=1)
    # print(ytrue)
    # print(ypredicted)

    # # Print performance metrics to terminal
    # confusion_matrix = multilabel_confusion_matrix(ytrue, ypredicted)
    # accuracy = accuracy_score(ytrue, ypredicted)

    # print(confusion_matrix)
    # print(accuracy)