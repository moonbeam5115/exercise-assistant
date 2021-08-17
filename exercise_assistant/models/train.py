import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from joblib import dump
import argparse

parser = argparse.ArgumentParser(description='Collect Some Data for Computer Vision Applications.')
parser.add_argument("--videos", default=30, help="Number of videos per pose to collect", type=int)
parser.add_argument("--fpv", default=30, help="Frames per video to collect", type=int)

args = parser.parse_args()

class ModelTrainer():

    def __init__(self, MODEL_PATH, pose_bank, videos, frames_per_video):
        self.MODEL_PATH = MODEL_PATH
        self.pose_bank = pose_bank
        self.videos = videos
        self.frames_per_video = frames_per_video
    
    def build_knn(self, num_neighbors=5):
        # Build Model Logic goes here
        Model = KNeighborsClassifier(num_neighbors)
        return Model

    def train_knn(self, model):
        x_path = os.path.join(self.MODEL_PATH, 'processed_data', 'X', 'features.npy')
        y_path = os.path.join(self.MODEL_PATH, 'processed_data', 'y', 'target.npy')

        X_data = np.load(x_path)
        y = np.load(y_path)
        print(X_data.shape)
        print(y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.1, random_state=42)
        dim1 = X_train.shape[0]
        dim2 = int(X_train.shape[1]*X_train.shape[2])
        X_train = np.reshape(X_train, (dim1, dim2))
        model.fit(X_train, y_train)

        return model

    def save_model(self, model, dest_directory):
        dump(model, dest_directory)

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
    #keypoints_per_frame = 66 Used for LSTM
    videos = args.videos
    frames_per_video = args.fpv
    pose_bank = np.array(['stand', 'right_lunge', 'left_lunge', 'other'])

    model_manager = ModelTrainer(MODEL_PATH, pose_bank, videos, frames_per_video)
    exercise_assistant = model_manager.build_knn()
    
    print("Training Neural Network...")
    trained_assistant = model_manager.train_knn(exercise_assistant)
    print("Training Complete!")
    MODEL_OUT_PATH = os.path.join(MODEL_PATH, 'classification', 'pose_classifier.joblib')
    model_manager.save_model(trained_assistant, MODEL_OUT_PATH)