from tensorflow.keras.utils import to_categorical
import numpy as np
import os

ROOT = 'exercise_assistant'
DATA_PATH = 'data'
videos = 30
frames_per_video = 45
actions = np.array(['left_lunge', 'right_lunge', 'stand'])

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
# A sequence is a group of frames representing an action
# Sequences are all the groups of frames representing all actions - our features AKA X (input)
# Labels will be our target variable (true value for action) - y (output)
def create_train_test_data():
    
    for action in actions:
        for vid_number in range(videos):
            action_keypoints = []
            for frame_number in range(frames_per_video):
                try:
                    frame_keypoints = np.load(os.path.join(ROOT, DATA_PATH, action, str(vid_number), "{}.npy".format(frame_number)))
                    action_keypoints.append(frame_keypoints)
                except Exception as e:
                    print(e)
            sequences.append(action_keypoints)
            labels.append(label_map[action])
    
    X_data = np.array(sequences)
    y = to_categorical(labels).astype(int)

    X_npy_path = os.path.join(ROOT, 'models', 'processed_data', 'X', 'features')
    y_npy_path = os.path.join(ROOT, 'models', 'processed_data', 'y', 'target')
    np.save(X_npy_path, X_data)
    np.save(y_npy_path, y)


if __name__ == '__main__':
    create_train_test_data()
    print(np.array(sequences).shape)
    print(len(sequences))
    # (90, 45, 132)
    # (Videos, Frames_per_video, Keypoints_per_frame)