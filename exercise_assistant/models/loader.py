import os
import  numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from train import build_lstm
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

ROOT = os.path.join('exercise_assistant')
X_PATH = os.path.join(ROOT, 'models', 'processed_data', 'X', 'features.npy')
Y_PATH = os.path.join(ROOT, 'models', 'processed_data', 'y', 'target.npy')

actions = np.array(['left_lunge', 'right_lunge', 'stand'])
keypoints_per_frame = 132
frames_per_video = 45

MODEL_PATH = os.path.join(ROOT, 'models', 'action_recognition', 'exercise_recognition_model.h5')

# Load Model Function
def load_model(x_path, y_path, model_path, actions, keypoints_per_frame, frames_per_video):
    X_data = np.load(x_path)
    y = np.load(y_path)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.1, random_state=42)
    exercise_recognition_model = build_lstm(frames_per_video, keypoints_per_frame, actions)

    exercise_recognition_model.load_weights(MODEL_PATH)

    return X_test, y_test, exercise_recognition_model



if __name__ == '__main__':

    X_test, y_test, exercise_recognition_model = load_model(X_PATH, Y_PATH, MODEL_PATH, actions,
                                                            keypoints_per_frame, frames_per_video)

    ypredicted = exercise_recognition_model.predict(X_test)

    ytrue = np.argmax(y_test, axis=1)
    ypredicted = np.argmax(ypredicted, axis=1)    


    print(ytrue)
    print(ypredicted)
    confusion_matrix = multilabel_confusion_matrix(ytrue, ypredicted)
    accuracy = accuracy_score(ytrue, ypredicted)

    print(confusion_matrix)
    print(accuracy)