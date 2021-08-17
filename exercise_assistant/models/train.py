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
import shutil

parser = argparse.ArgumentParser(description='Collect Some Data for Computer Vision Applications.')
parser.add_argument("--videos", default=30, help="Number of videos per pose to collect", type=int)
parser.add_argument("--fpv", default=30, help="Frames per video to collect", type=int)
parser.add_argument("--save", help="Save Results to DB", action="store_true")

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

        X_data = np.load(x_path, allow_pickle=True)
        y = np.load(y_path, allow_pickle=True)
        X_data = np.array(X_data)
        X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.1, random_state=42)
        print("X_TRAIN SHAPE: ", X_train.shape)
        print("y_TRAIN SHAPE: ", y_train.shape)

        nested_dim = X_train[1].shape
        dim1 = X_train.shape[0]
        dim2 = int(nested_dim[0]*nested_dim[1])
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
    PIPELINE_OUT_PATH = os.path.join('gDrive_Output') # Place output location here
    MODEL_PATH = os.path.join('exercise_assistant', 'models')
    DATA_PATH = os.path.join('exercise_assistant', 'data')
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
    save_condition = len(os.listdir(DATA_PATH))
    if args.save and (save_condition == len(pose_bank)):
        print("DATA SAVED!")
        x_path = os.path.join(MODEL_PATH, 'processed_data', 'X', 'features.npy')
        y_path = os.path.join(MODEL_PATH, 'processed_data', 'y', 'target.npy')
        # if --save: Save Model, Processed_Data (X, y) - Then Remove
        if os.path.exists(PIPELINE_OUT_PATH):
            model_file_path = os.path.join(PIPELINE_OUT_PATH, 'KNN_model.joblib')
            model_manager.save_model(trained_assistant, model_file_path)
            shutil.copy(x_path, PIPELINE_OUT_PATH)
            shutil.copy(y_path, PIPELINE_OUT_PATH)
            os.remove(MODEL_OUT_PATH)
            os.remove(x_path)
            os.remove(y_path)

            # Save Raw Data Folders
            if os.path.exists(DATA_PATH):
                # Save to Google Drive and Delete from local machine
                # TODO SAVE LOGIC GOES HERE
                for dir in os.listdir(DATA_PATH):
                    pose_folder_path = os.path.join(DATA_PATH, dir)
                    final_path = os.path.join(PIPELINE_OUT_PATH, 'Data')
                    if not os.path.exists(final_path):
                        shutil.copytree(pose_folder_path, final_path)
                    shutil.rmtree(pose_folder_path)
