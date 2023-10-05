import joblib
from scipy.cluster.vq import kmeans, vq
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

train_path = "C:/Users/27820/Desktop/Campus Stuff/MASTERS/trial/train"
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

k = 80
voc, variance = kmeans(descriptors_float, k, 1)

im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1

stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# X_train, X_test, y_train, y_test = train_test_split(
# im_features, np.array(image_classes),
# test_size = 0.30)

# param_grid = {'n_neighbors': [1, 3, 5],
# 'weights': ['uniform', 'distance'],
# 'metric': ['euclidean', 'manhattan']}

# grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, verbose=3)

# fitting the model for grid search
# grid.fit(X_train, y_train)

# print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
# print(grid.best_estimator_)
clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=1, metric="euclidean", weights='distance'))

clf.fit(im_features, np.array(image_classes))

joblib.dump((clf, training_names, stdSlr, k, voc), "C:/Users/27820/Desktop/Campus Stuff/MASTERS/trial/bovwknn.pkl",
            compress=3)
