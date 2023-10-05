import cv2
import joblib
from sklearn.metrics import roc_curve, auc, classification_report
from scipy.cluster.vq import kmeans, vq
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scikitplot as skplt

train_path = "C:/Users/27820/Desktop/Campus Stuff/MASTERS/dataset/train"
training_names = os.listdir(train_path)

image_paths = []
image_classes = []
class_id = 0


def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imglist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

des_list = []

surf = cv2.xfeatures2d.SURF_create(500)
surf.setUpright(True)
surf.setExtended(True)
print(surf.getHessianThreshold())

for image_path in image_paths:
    im = cv2.imread(image_path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(equ, (5, 5), 0)
    kpts, des = surf.detectAndCompute(blur, None)
    des_list.append((image_path, des))

descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

descriptors_float = descriptors.astype(float)

k = 80  # k is the number of clusters we want to divide into
voc, variance = kmeans(descriptors_float, k, 1)

# histogram of features, represent as a vector
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1

stdSlr = StandardScaler().fit(im_features)  # dont want the histogram to be scewed by outliers so we normalize it
im_features = stdSlr.transform(im_features)

X_train, X_test, y_train, y_test = train_test_split(im_features, np.array(image_classes), test_size=0.2)

clf = OneVsRestClassifier(SVC(kernel='rbf', gamma=0.001, C=100, probability=True))
clf.fit(im_features, np.array(image_classes))


joblib.dump((clf, training_names, stdSlr, k, voc), "C:/Users/27820/Desktop/Campus Stuff/MASTERS/dataset/bovw.pkl",
            compress=3)

######### GRID SEARCH ###########

# param_grid = {'C': [0.1, 1, 10, 100, 1000],
# 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
# 'kernel': ['rbf']}

# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

# fitting the model for grid search
# grid.fit(X_train, y_train)
