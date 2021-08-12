from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

ROOT = 'exercise_assistant'
DATA_PATH = 'data'
videos = 30
frames_per_video = 45
actions = np.array(['left_lunge', 'right_lunge', 'stand'])

label_map = {label:num for num, label in enumerate(actions)}

# A sequence is a group of frames representing an action
# Sequences are all the groups of frames representing all actions - our features AKA X (input)
# Labels will be our target variable (true value for action) - y (output)
sequences, labels = [], []
for action in actions:
    for vid_number in range(videos):
        action_keypoints = []
        for frame_number in range(frames_per_video):
            frame_keypoints = np.load(os.path.join(ROOT, DATA_PATH, action, str(vid_number), "{}.npy".format(frame_number)))
            action_keypoints.append(frame_keypoints)
        sequences.append(action_keypoints)
        labels.append(label_map[action])

print(np.array(sequences).shape)
print(len(sequences))
# (90, 45, 132)
# (Videos, Frames_per_video, Keypoints_per_frame)