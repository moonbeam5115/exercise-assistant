import cv2
import os
import mediapipe as mp
import numpy as np

# Media Pipe Pose Estimator
class Estimator():

  def __init__(self):
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_pose = mp.solutions.pose

  def draw_styled_landmarks(self, image, results):
    # Stylize the landmarks
    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
    self.mp_drawing.DrawingSpec(color=(50, 22, 210), thickness=2, circle_radius=1),
    self.mp_drawing.DrawingSpec(color=(120, 200, 21), thickness=2, circle_radius=1)
    )

  def transform_keypoints(self, results):
    # Flatten Array of Key Points to feed into NN Model
    flattened_pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return flattened_pose

  def estimation_loop(self):
    # For webcam input:
    webcam = cv2.VideoCapture(0)
    with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
      while webcam.isOpened():
        # get frame from webcam
        success, image = webcam.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.draw_styled_landmarks(image, results)

        cv2.imshow('Exercise Assistant', image)
        if cv2.waitKey(5) & 0xFF == 27:
          break
    webcam.release()
    cv2.destroyAllWindows()

  def calculate_angles(self, results):
    pose_angles = []

    return pose_angles

  def draw_angles(self, angles):
    pass
        


