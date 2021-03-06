import cv2
import os
import mediapipe as mp
import numpy as np
from exercise_assistant.models.loader import ModelLoader
from joblib import load


# Media Pipe Pose Estimator
class Estimator():

  def __init__(self):
    ROOT = os.path.join('exercise_assistant')
    self.mp_drawing = mp.solutions.drawing_utils
    self.mp_pose = mp.solutions.pose
    self.model_path = os.path.join(ROOT, 'models', 'classification', 'pose_classifier.joblib')
    self.pose_bank = np.array(['stand', 'right_lunge', 'left_lunge', 'other'])
    self.keypoints_per_frame = 132
    self.frames_per_video = 45

  def draw_styled_landmarks(self, image, results):
    # Stylize the landmarks
    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
    self.mp_drawing.DrawingSpec(color=(50, 22, 210), thickness=2, circle_radius=1),
    self.mp_drawing.DrawingSpec(color=(120, 200, 21), thickness=2, circle_radius=1)
    )

  def transform_keypoints(self, results, keypoint_values=66):
    # Flatten Array of Key Points to feed into NN Model
    # Only considering x, y coordinates on image for now
    flattened_pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(keypoint_values)
    return flattened_pose

  def estimation_loop(self):
    sequence = []
    action_history = []
    threshold = 0.4
    ai_coach = load('gDrive_Output/KNN_model.joblib')
    
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
        flattened_keypoints = self.transform_keypoints(results)
        print("FLATTENED KEYPOINT INFO: ")
        print(len(flattened_keypoints), flattened_keypoints.reshape(1, -1))
        action_predicted = ai_coach.predict(flattened_keypoints.reshape(1, -1))
        print(action_predicted)
        action_predicted = self.pose_bank[np.argmax(action_predicted)]
        print(action_predicted)
        color = (120, 150, 0)
        text_thickness = 2
        cv2.putText(image, '{}'.format(action_predicted), (120, 80),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness, cv2.LINE_AA)
        
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
        


