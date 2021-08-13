import tools
import time
import numpy as np
import os
from matplotlib import pyplot as plt
from models.loader import load_model
# # Main Program
# def mediapipe_detection(image, model):

# Repetitions will be counted when going from {Exercise Pose} back to {Resting Pose}
def create_exercise_assistant():
    ROOT = os.path.join('exercise_assistant')
    X_PATH = os.path.join(ROOT, 'models', 'processed_data', 'X', 'features.npy')
    Y_PATH = os.path.join(ROOT, 'models', 'processed_data', 'y', 'target.npy')

    actions = np.array(['left_lunge', 'right_lunge', 'stand'])
    keypoints_per_frame = 132
    frames_per_video = 45

    MODEL_PATH = os.path.join(ROOT, 'models', 'action_recognition', 'exercise_recognition_model.h5')
    exercise_assistant = load_model(MODEL_PATH, actions,
                                        keypoints_per_frame, frames_per_video)

    return exercise_assistant


if __name__ == '__main__':
    exercise_assistant = create_exercise_assistant()
    poseEstimator = tools.Estimator()
    print("Begin Program")
    print("...")
    poseEstimator.estimation_loop()
    print("...")
    print("End Program")