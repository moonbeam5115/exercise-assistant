from tensorflow.keras.utils import to_categorical
import numpy as np
import os
import argparse

# A sequence is a group of frames representing an action
# Sequences are all the groups of frames representing all actions - our features AKA X (input)
# Labels will be our target variable (true value for action) - y (output)

parser = argparse.ArgumentParser(description='Collect Some Data for Computer Vision Applications.')
parser.add_argument("--videos", default=30, help="Number of videos per pose to collect", type=int)
parser.add_argument("--fpv", default=30, help="Frames per video to collect", type=int)

args = parser.parse_args()

class CreateTrainTestData():

    def __init__(self, DATA_PATH, videos, frames_per_video):
        self.DATA_PATH = DATA_PATH
        self.model_path = os.path.join('exercise_assistant', 'models')
        self.pose_bank = np.array(['stand', 'right_lunge', 'left_lunge', 'other'])
        self.videos = videos
        self.frames_per_video = frames_per_video
    
    def preprocess_lstm(self):
        pass

    def preprocess_knn(self, save=False):
        videos = self.videos
        frames_per_video = self.frames_per_video
        pose_bank = self.pose_bank
        keypoints_per_frame = 66
        label_map = {label:num for num, label in enumerate(pose_bank)}
        total_poses, labels = [], []

        for exercise_pose in pose_bank:
            for vid_number in range(videos):
                for frame_number in range(frames_per_video):
                    try:
                        path_to_keypoint_data = os.path.join(self.DATA_PATH, '{}'.format(exercise_pose), str(vid_number), "{}.npy".format(frame_number))
                        frame_keypoints = np.load(path_to_keypoint_data, allow_pickle=True)
                        total_poses.append(frame_keypoints)
                        labels.append(label_map[exercise_pose])
                    except Exception as e:
                        print(e)
                        print("trouble accessing")
                
    
        # X_data is the features (List of keypoint positions + visibility) that describe the action
        # y is the target value (The ground truth for that action)

        X_data = np.array(total_poses)
        y = to_categorical(labels).astype(int)

        if save:
            MODEL_OUT_PATH = os.path.join(self.model_path, 'processed_data')
            print("Saving Preprocessed Results to {}".format(MODEL_OUT_PATH))
            X_npy_path = os.path.join(MODEL_OUT_PATH, 'X', 'features')
            y_npy_path = os.path.join(MODEL_OUT_PATH, 'y', 'target')
            np.save(X_npy_path, X_data, allow_pickle=True)
            np.save(y_npy_path, y, allow_pickle=True)

        return X_data, y


if __name__ == '__main__':
    DATA_PATH = os.path.join('exercise_assistant', 'data')
    videos = args.videos
    frames_per_video = args.fpv
    data_manager = CreateTrainTestData(DATA_PATH, videos, frames_per_video)
    X_data, y = data_manager.preprocess_knn(save=True)
    print("X DATA SHAPE: ", np.array(X_data).shape)
    print("y DATA SHAPE: ", np.array(y).shape)
    # (30, 30, 66)
    # (Videos, Frames_per_video, Keypoints_per_frame)