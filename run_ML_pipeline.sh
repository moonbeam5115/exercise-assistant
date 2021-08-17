export vNum=2
export FPV=10
#python exercise_assistant\\tools\\collect_data.py --videos $vNum --fpv $FPV
python exercise_assistant\\tools\\preprocessing.py --videos $vNum --fpv $FPV
python exercise_assistant\\models\\train.py --videos $vNum --fpv $FPV