import os
import shutil
import numpy as np
import cv2
import time
import mediapipe as mp
import pyttsx3
from tools.pose import Estimator


# Global Variables
DATA_PATH = os.path.join('exercise_assistant/data')

# Actions to detect
actions = np.array(['left_lunge', 'right_lunge', 'stand'])

# Number of Videos worth of data
videos = 30

# Number of frames in each video
frames_per_video = 45

# Pose Model
mp_pose = mp.solutions.pose
text_thickness = 2
color = (0, 0, 255)


def create_action_directories(actions):
    for action in actions:
        for vid_number in range(videos):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(vid_number)))
            except Exception as e:
                pass

def export_keypoint_data(src_data_path, dest_data_path):
    # TODO: Write function to export data to Google Drive:
    # https://developers.google.com/drive/api/v3/quickstart/python
    for action_folder in os.listdir(src_data_path):
        pass


def process_image(image, pose_model, estimator):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose_model.process(image)
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    estimator.draw_styled_landmarks(image, results)

    return image, results

def split_lines(image, text):
    y0, dy = 50, 50
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        cv2.putText(image, line, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), text_thickness, cv2.LINE_AA)


def collect_keypoint_pose_data(actions):
    estimator = Estimator()
    # For webcam input:
    webcam = cv2.VideoCapture(0)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        
        for action in actions:
            # Loop through actions
            print('Collecting Key Point Data for {}'.format(action))
            for vid_number in range(videos):
                # Loop through videos (aka sequences)
                for frame_number in range(frames_per_video):
                    # Loop through each frame
                    # Capture frame from webcam
                    success, image = webcam.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        # If loading a video, use 'break' instead of 'continue'.
                        continue
                    scale_percent = 130
                    width = int(image.shape[1]*scale_percent/100)
                    height = int(image.shape[0]*scale_percent/100)
                    dsize = (width, height)
                    image = cv2.resize(image, dsize)
                    image, result = process_image(image, pose, estimator)


                    # Apply collection logic
                    if frame_number == 0:
                        # Restarts and Begins CountDown
                        # Countdown 3
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness, cv2.LINE_AA)
                        text = 'Collecting frames for {} \n Video Number {}'.format(action, vid_number+1)
                        cv2.putText(image, '3', (420, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness, cv2.LINE_AA)
                        split_lines(image, text)
                        cv2.imshow('Exercise Assistant', image)
                        cv2.waitKey(1000)
                        pyttsx3.speak('3')

                        # Countdown 2
                        success, image = webcam.read()
                        image = cv2.resize(image, dsize)
                        image, result = process_image(image, pose, estimator)
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness, cv2.LINE_AA)
                        cv2.putText(image, '2', (420, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness, cv2.LINE_AA)
                        split_lines(image, text)
                        cv2.imshow('Exercise Assistant', image)
                        cv2.waitKey(1000)
                        pyttsx3.speak('2')

                        # Countdown 1
                        success, image = webcam.read()
                        image = cv2.resize(image, dsize)
                        image, result = process_image(image, pose, estimator)
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness, cv2.LINE_AA)
                        cv2.putText(image, '1', (420, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness, cv2.LINE_AA)
                        split_lines(image, text)
                        cv2.imshow('Exercise Assistant', image)
                        cv2.waitKey(1000)
                        pyttsx3.speak('1')
                    else:
                        text = 'Collecting frames for {} \n Video Number {}'.format(action, vid_number+1)
                        # This Collects the Data
                        
                        split_lines(image, text)
                        #process_image(image, pose)
                        cv2.imshow('Exercise Assistant', image)
                    
                    # Export Keypoints
                    keypoints = estimator.transform_keypoints(result)
                    npy_path = os.path.join(DATA_PATH, action, str(vid_number), str(frame_number))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Create directories for Actions
    if not os.path.isdir('exercise_assistant/data/{}'.format(actions[0])):
        print('Creating Action Folders and Video_Number Subfolders...')
        create_action_directories(actions)
        
    print('Collecting data for training...')    
    collect_keypoint_pose_data(actions)