import tools
import time
import numpy as np
from matplotlib import pyplot as plt

# # Main Program
# def mediapipe_detection(image, model):

# Repetitions will be counted when going from {Exercise Pose} back to {Resting Pose}


if __name__ == '__main__':
    poseEstimator = tools.Estimator()
    print("Begin Program")
    print("...")
    poseEstimator.estimation_loop()
    print("...")
    print("End Program")