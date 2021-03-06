from exercise_assistant.tools.pose import Estimator
import time
import numpy as np
import os
from matplotlib import pyplot as plt

# Repetitions will be counted when going from {Exercise Pose} back to {Resting Pose}

if __name__ == '__main__':
    poseEstimator = Estimator()
    print("Begin Program")
    print("...")
    poseEstimator.estimation_loop()
    print("...")
    print("End Program")