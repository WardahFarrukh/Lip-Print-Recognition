import itertools
import numpy as np
import cv2
import os
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, \
    precision_recall_fscore_support, roc_curve, plot_confusion_matrix
import joblib
import matplotlib.pyplot as plt
import scikitplot as skplt
from scipy.cluster.vq import vq

# Load the classifier, class names, scaler, number of clusters and vocabulary
clf, classes_names, stdSlr, k, voc = joblib.load("C:/Users/27820/Desktop/Campus Stuff/MASTERS/dataset/bovw.pkl")

# path of the testing image and store them in a list
test_path = 'C:/Users/27820/Desktop/Campus Stuff/MASTERS/dataset/test'

testing_names = os.listdir(test_path)

# path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0


# list all file names in a directory
def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


# Fill empty lists with image path, classes, and add class ID number
for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = imglist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

# Create feature extraction and keypoint detector objects using sift
# List to store descriptors
des_list = []

# setting hessian threshold to 500
surf = cv2.xfeatures2d.SURF_create(500)
# upright orientation
surf.setUpright(True)
# 128-bit descriptor
surf.setExtended(True)

# pre-processing
for image_path in image_paths:
    im = cv2.imread(image_path)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(equ, (5, 5), 0)
    kpts, des = surf.detectAndCompute(blur, None)
    des_list.append((image_path, des))

# Stack all the descriptors  in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor))

# Calculate histogram of features
test_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        test_features[i][w] += 1

# Scale the features
test_features = stdSlr.transform(test_features)

# true class names so they can be compared with predicted classes
true_class = [classes_names[i] for i in image_classes]
# predictions and report predicted class names.
predictions = [classes_names[i] for i in clf.predict(test_features)]

# Print the true class and Predictions
print("true_class =" + str(true_class))
print("prediction =" + str(predictions))


def showconfusionmatrix(cm):
    plt.imshow(cm)
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.show()


accuracy = accuracy_score(true_class, predictions)
print("accuracy = ", accuracy)
cm = confusion_matrix(true_class, predictions)
showconfusionmatrix(cm)

probabilities = clf.predict_proba(test_features)

########################## ROC CURVE##############################


skplt.metrics.plot_roc(true_class, probabilities, title="(BoVW SVM) ROC CURVE", classes_to_plot=[0, "cold"])
skplt.metrics.plot_precision_recall(true_class, probabilities, title="((BoVW) SVM Precision-Recall CURVE",
                                    classes_to_plot=[0, "cold"])
plt.show()

print("accuracy = ", accuracy)
pmacro = precision_recall_fscore_support(true_class, predictions, average='macro')
pmicro = precision_recall_fscore_support(true_class, predictions, average='micro')

print(pmacro)
print(pmicro)


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print('True positive = ', cm[0][0])
    print('False positive = ', cm[0][1])
    print('False negative = ', cm[1][0])
    print('True negative = ', cm[1][1])

print_confusion_matrix(true_class, predictions)