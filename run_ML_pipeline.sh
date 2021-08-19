export vNum=30 # Number of Videos to Capture
export FPV=30 # Number of Frames per video
python exercise_assistant\\tools\\collect_data.py --videos $vNum --fpv $FPV
python exercise_assistant\\tools\\preprocessing.py --videos $vNum --fpv $FPV
python exercise_assistant\\models\\train.py --videos $vNum --fpv $FPV --save