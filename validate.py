from joblib import load
import numpy as np
import os
from sklearn.model_selection import train_test_split

pose_classifier = load('gDrive_Output/KNN_model.joblib')
ROOT = 'gDrive_Output'
x_path = os.path.join(ROOT, 'features.npy')
y_path = os.path.join(ROOT, 'target.npy')
X_data = np.load(x_path, allow_pickle=True)
y = np.load(y_path, allow_pickle=True)
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.1, random_state=42)
print(pose_classifier.score(X_test, y_test))