import os
import shutil
import numpy as np
import cv2
import time
import mediapipe as mp
import pyttsx3
from exercise_assistant.tools.pose import  Estimator
import argparse

parser = argparse.ArgumentParser(description='Collect Some Data for Computer Vision Applications.')
parser.add_argument("--videos", default=30, help="Number of videos per pose to collect", type=int)
parser.add_argument("--fpv", default=30, help="Frames per video to collect", type=int)

args = parser.parse_args()

def create_pose_directories(pose_bank, DATA_PATH, num_vids):
    # Create directory for each action and appropriate subdirectories
    for exercise_pose in pose_bank:
        for vid_number in range(num_vids):
            try:
                os.makedirs(os.path.join(DATA_PATH, exercise_pose, str(vid_number)))
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
    text_thickness = 2
    # Create a second Line of Text in openCV
    y0, dy = 50, 50
    for i, line in enumerate(text.split('\n')):
        y = y0 + i*dy
        cv2.putText(image, line, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), text_thickness, cv2.LINE_AA)


def collect_keypoint_pose_data(pose_bank, videos, frames_per_video):
    # Path Variable
    DATA_PATH = os.path.join('exercise_assistant', 'data')

    # Pose Model
    mp_pose = mp.solutions.pose
    color = (0, 0, 255)
    text_thickness = 2
    estimator = Estimator()
    # For webcam input:
    webcam = cv2.VideoCapture(0)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose_model:

        for exercise_pose in pose_bank:
            # Loop through actions
            print('Collecting Key Point Data for {}'.format(exercise_pose))
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
                    frame_width = int(image.shape[1])
                    frame_height = int(image.shape[0])
                    desired_size = (int(frame_width*scale_percent/100), int(frame_height*scale_percent/100))
                    image = cv2.resize(image, desired_size)
                    image, result = process_image(image, pose_model, estimator)

                    # Apply collection logic
                    if frame_number == 0:
                        # Restarts and Begins CountDown
                        # Countdown 3
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness, cv2.LINE_AA)
                        text = 'Collecting frames for {} \n Video Number {}'.format(exercise_pose, vid_number+1)
                        cv2.putText(image, '3', (420, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness, cv2.LINE_AA)
                        split_lines(image, text)
                        cv2.imshow('Exercise Assistant', image)
                        cv2.waitKey(1000)
                        pyttsx3.speak('3')

                        # Countdown 2
                        success, image = webcam.read()
                        image = cv2.resize(image, desired_size)
                        image, result = process_image(image, pose_model, estimator)
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
                        image = cv2.resize(image, desired_size)
                        image, result = process_image(image, pose_model, estimator)
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness, cv2.LINE_AA)
                        cv2.putText(image, '1', (420, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, text_thickness, cv2.LINE_AA)
                        split_lines(image, text)
                        cv2.imshow('Exercise Assistant', image)
                        cv2.waitKey(1000)
                        pyttsx3.speak('1')
                    else:
                        # Show real time feedback for user
                        text = 'Collecting frames for {} \n Video Number {}'.format(exercise_pose, vid_number+1)
                        split_lines(image, text)
                        cv2.imshow('Exercise Assistant', image)
                    # This Collects the Data 
                    # Transform (Flatten) and Export Keypoints
                    keypoints = estimator.transform_keypoints(result)
                    npy_path = os.path.join(DATA_PATH, exercise_pose, str(vid_number), str(frame_number))
                    np.save(npy_path, keypoints)

                    if cv2.waitKey(5) & 0xFF == 27:
                        break
                
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Number of videos to collect data for
    videos = args.videos
    # Number of frames for each video
    frames_per_video = args.fpv
    DATA_PATH = os.path.join('exercise_assistant/data')
    # Exercise Poses to detect
    pose_bank = np.array(['right_lunge', 'left_lunge', 'stand', 'other'])

    # Create directories for exercise_poses - if they don't exist yet
    if not os.path.isdir('exercise_assistant/data/{}'.format(pose_bank[0])):
        print('Creating Exercise Pose Folders and Video_Number Subfolders...')
        create_pose_directories(pose_bank, DATA_PATH, videos)
    
    # Begin data collection process
    print('Collecting Training Data for {} Videos \n @ {} Frames per Video'.format(videos, frames_per_video))    
    collect_keypoint_pose_data(pose_bank, videos, frames_per_video)