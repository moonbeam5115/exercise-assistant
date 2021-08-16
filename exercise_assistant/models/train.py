import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

class ModelTrainer():

    def __init__(self, MODEL_PATH, pose_bank):
        self.MODEL_PATH = MODEL_PATH
        self.pose_bank = pose_bank
        self.videos = 30
        self.frames_per_video = 30
    
    def build_knn(frames_per_video, keypoints_per_frame, pose_bank):
        # Build Model Logic goes here
        Model = None
        return Model

    def train_knn(self, model):
        x_path = os.path.join(self.MODEL_PATH, 'processed_data', 'X', 'features.npy')
        y_path = os.path.join(self.MODEL_PATH, 'processed_data', 'y', 'target.npy')

        X_data = np.load(x_path)
        y = np.load(y_path)

        X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.1, random_state=42)

        # log_dir = os.path.join(self.MODEL_PATH, 'logs')
        # tb_callback = TensorBoard(log_dir=log_dir)
        model.fit(X_train, y_train)

        return model

    def save_model(self, model, dest_directory):
        out_file_path = os.path.join(dest_directory, 'pose_classification_model.h5')
        model.save(out_file_path)

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


if __name__ == '__main__':
    MODEL_PATH = os.path.join('exercise_assistant', 'models')
    keypoints_per_frame = 66
    frames_per_video = 30
    pose_bank = np.array(['stand', 'right_lunge', 'left_lunge', 'other'])

    model_manager = ModelTrainer(MODEL_PATH, pose_bank)
    exercise_assistant = model_manager.build_knn(keypoints_per_frame, pose_bank)
    
    print("Training Neural Network...")
    trained_assistant = model_manager.train_knn(exercise_assistant)
    print("Training Complete!")
    MODEL_OUT_PATH = os.path.join(MODEL_PATH, 'classification')
    model_manager.save_model(trained_assistant, MODEL_OUT_PATH)