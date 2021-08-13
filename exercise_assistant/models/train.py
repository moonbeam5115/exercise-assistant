import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


def build_lstm(frames_per_video, keypoints_per_frame, actions):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu',
                input_shape=(frames_per_video, keypoints_per_frame)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model

def train_model(model):
    ROOT = 'exercise_assistant'
    x_path = os.path.join(ROOT, 'models', 'processed_data', 'X', 'features.npy')
    y_path = os.path.join(ROOT, 'models', 'processed_data', 'y', 'target.npy')
    
    X_data = np.load(x_path)
    y = np.load(y_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.1, random_state=42)
    
    log_dir = os.path.join(ROOT, 'models', 'logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    model.fit(X_train, y_train, epochs=350, callbacks=[tb_callback])

    return model

def save_model(model, dest_directory):
    out_file_path = os.path.join(dest_directory, 'exercise_recognition_model.h5')
    model.save(out_file_path)

if __name__ == '__main__':

    keypoints_per_frame = 132
    frames_per_video = 45
    actions = np.array(['left_lunge', 'right_lunge', 'stand'])
    
    exercise_assistant = build_lstm(frames_per_video, keypoints_per_frame, actions)

    trained_assistant = train_model(exercise_assistant)

    MODEL_OUT_PATH = os.path.join('exercise_assistant', 'models', 'action_recognition')
    save_model(trained_assistant, MODEL_OUT_PATH)