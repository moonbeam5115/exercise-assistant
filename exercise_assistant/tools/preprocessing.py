from exercise_assistant.models.loader import MODEL_PATH
from exercise_assistant.models.train import MODEL_OUT_PATH
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

# A sequence is a group of frames representing an action
# Sequences are all the groups of frames representing all actions - our features AKA X (input)
# Labels will be our target variable (true value for action) - y (output)

class CreateTrainTestData():

    def __init__(self, DATA_PATH):
        self.DATA_PATH = DATA_PATH
        self.pose_bank = np.array(['stand', 'right_lunge', 'left_lunge', 'other'])
        self.videos = 30
        self.frames_per_video = 30
    
    def preprocess_lstm(self):
        pass

    def preprocess_knn(self, save=False):
        videos = self.videos
        frames_per_video = self.frames_per_video
        pose_bank = self.pose_bank

        label_map = {label:num for num, label in enumerate(pose_bank)}
        sequences, labels = [], []

        for exercise_pose in pose_bank:
            for vid_number in range(videos):
                pose_keypoints = []
                for frame_number in range(frames_per_video):
                    try:
                        frame_keypoints = np.load(self.DATA_PATH, exercise_pose, str(vid_number), "{}.npy".format(frame_number))
                        pose_keypoints.append(frame_keypoints)
                    except Exception as e:
                        print(e)
                sequences.append(pose_keypoints)
                labels.append(label_map[exercise_pose])
    
        # X_data is the features (List of keypoint positions + visibility) that describe the action
        # y is the target value (The ground truth for that action)
        X_data = np.array(sequences)
        y = to_categorical(labels).astype(int)

        if save:
            MODEL_OUT_PATH = os.path.join(MODEL_PATH, 'processed_data')
            print("Saving Preprocessed Results to {}".format(MODEL_OUT_PATH))
            X_npy_path = os.path.join(MODEL_OUT_PATH, 'X', 'features')
            y_npy_path = os.path.join(MODEL_OUT_PATH, 'y', 'target')
            np.save(X_npy_path, X_data)
            np.save(y_npy_path, y)

        return X_data, y

# def create_train_test_data():
#     ROOT = 'exercise_assistant'
#     DATA_PATH = 'data'
#     videos = 30
#     frames_per_video = 45
#     actions = np.array(['left_lunge', 'right_lunge', 'stand'])

#     label_map = {label:num for num, label in enumerate(actions)}
#     sequences, labels = [], []

#     for action in actions:
#         for vid_number in range(videos):
#             action_keypoints = []
#             for frame_number in range(frames_per_video):
#                 try:
#                     frame_keypoints = np.load(os.path.join(ROOT, DATA_PATH, action, str(vid_number), "{}.npy".format(frame_number)))
#                     action_keypoints.append(frame_keypoints)
#                 except Exception as e:
#                     print(e)
#             sequences.append(action_keypoints)
#             labels.append(label_map[action])
    
#     # X_data is the features (List of keypoint positions + visibility) that describe the action
#     # y is the target value (The ground truth for that action)
#     X_data = np.array(sequences)
#     y = to_categorical(labels).astype(int)

#     X_npy_path = os.path.join(ROOT, 'models', 'processed_data', 'X', 'features')
#     y_npy_path = os.path.join(ROOT, 'models', 'processed_data', 'y', 'target')
#     np.save(X_npy_path, X_data)
#     np.save(y_npy_path, y)

#     return X_data, y


if __name__ == '__main__':
    DATA_PATH = os.path.join('exercise_assistant', 'data')
    data_manager = CreateTrainTestData(DATA_PATH)
    X_data, y = data_manager.preprocess_knn(save=True)
    print(np.array(X_data).shape)
    print(np.array(y).shape)
    print(len(X_data))
    # (90, 30, 66)
    # (Videos, Frames_per_video, Keypoints_per_frame)