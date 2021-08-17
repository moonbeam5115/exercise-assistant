from setuptools import setup, find_packages

setup(name='exercise_assistant',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'opencv-python==4.5.3.56',
        'tensorflow==2.5.0',
        'scikit-learn==0.24.2',
        'mediapipe==0.8.3',
        'pyttsx3==2.90',
        'pydrive==1.3.1'
    ]
)