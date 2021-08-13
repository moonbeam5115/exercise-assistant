from exercise_assistant import tools
import time
import numpy as np
import os
from matplotlib import pyplot as plt
from exercise_assistant.models.loader import load_model

# Repetitions will be counted when going from {Exercise Pose} back to {Resting Pose}

if __name__ == '__main__':
    poseEstimator = tools.Estimator()
    exercise_assistant = poseEstimator.create_exercise_assistant()
    print("Begin Program")
    print("...")
    poseEstimator.estimation_loop()
    print("...")
    print("End Program")