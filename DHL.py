import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.manifold import TSNE
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import shap

# print(os.listdir("C:/Users/27820/Desktop/Campus Stuff/MASTERS/trial/"))
from tensorflow.python.keras.models import load_model, Model

img_width = 224  # Resize images
img_height = 260

# Capture training data and labels into respective lists
train_images = []
train_labels = []

for directory_path in glob.glob("C:/Users/27820/Desktop/Campus Stuff/MASTERS/trial/train/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_width, img_height))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

# Convert lists to arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Capture test/validation data and labels into respective lists

test_images = []
test_labels = []
for directory_path in glob.glob("C:/Users/27820/Desktop/Campus Stuff/MASTERS/trial/test/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_width, img_height))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)

# Convert lists to arrays
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Encode labels from text to integers.
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

# Split data into test and train datasets (already split but assigning to meaningful convention)
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

###################################################################
# Normalize pixel values to between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0


model = load_model('model-finetuned.h5')

#############################
# Load model wothout classifier/fully connected layers
RESNET_model = Model(inputs=model.input, outputs=model.get_layer('dense').output)

RESNET_model.summary()  # Trainable parameters will be 0

# Now, let us use features from convolutional network for RF
feature_extractor = RESNET_model.predict(x_train)

# SVM

RF_model = OneVsRestClassifier(SVC(kernel='rbf', C=1000, gamma=0.1, probability=True))

# Train the model on training data
RF_model.fit(feature_extractor, y_train)  # For sklearn no one hot encoding

# Send test data through same feature extractor process
X_test_feature = RESNET_model.predict(x_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

# Now predict using the trained model.
prediction_RF = RF_model.predict(X_test_features)
prediction_RF = le.inverse_transform(prediction_RF)
prediction = RF_model.predict(X_test_features)


tsne = TSNE(n_components=2).fit_transform(feature_extractor)


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)
# initialize a matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.show()

# Print overall accuracy
from sklearn import metrics

print("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))

probabilities = RF_model.predict_proba(X_test_features)

print(test_labels.shape)
print(prediction_RF.shape)

# Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

cm = confusion_matrix(test_labels, prediction_RF)
print(cm)


# classification_metrics = metrics.classification_report(test_labels, prediction_RF)
# print(classification_metrics)

def showconfusionmatrix(cm):
    plt.imshow(cm)
    plt.title("Confusion matrix")
    plt.colorbar()
    plt.show()


showconfusionmatrix(cm)

pmacro = precision_recall_fscore_support(test_labels, prediction_RF, average='macro')
pmicro = precision_recall_fscore_support(test_labels, prediction_RF, average='micro')

print(pmacro)
print(pmicro)

skplt.metrics.plot_roc(test_labels, probabilities, title="(resnet50 + svm) ROC CURVE", classes_to_plot=[0, "cold"])
skplt.metrics.plot_precision_recall(test_labels, probabilities, title="(resnet50 + svm) Precision-Recall CURVE",
                                    classes_to_plot=[0, "cold"])
plt.show()
