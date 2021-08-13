import os
import  numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from train import build_lstm
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

ROOT = os.path.join('exercise_assistant')
x_path = os.path.join(ROOT, 'models', 'processed_data', 'X', 'features.npy')
y_path = os.path.join(ROOT, 'models', 'processed_data', 'y', 'target.npy')

actions = np.array(['left_lunge', 'right_lunge', 'stand'])
keypoints_per_frame = 132
frames_per_video = 45

X_data = np.load(x_path)
y = np.load(y_path)
print(X_data.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.1, random_state=42)

MODEL_PATH = os.path.join(ROOT, 'models', 'action_recognition', 'exercise_recognition_model.h5')

exercise_recognition_model = build_lstm(frames_per_video, keypoints_per_frame, actions)

exercise_recognition_model.load_weights(MODEL_PATH)

yhat = exercise_recognition_model.predict(X_test)

ytrue = np.argmax(y_test, axis=1)
yhat = np.argmax(yhat, axis=1)
print(ytrue)
print(yhat)
#confusion_matrix = multilabel_confusion_matrix(ytrue, yhat)
accuracy = accuracy_score(ytrue, yhat)

#print(confusion_matrix)
print(accuracy)
