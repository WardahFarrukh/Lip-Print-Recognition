import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.inspection import permutation_importance
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

model = load_model('Resnet_model')

#############################
# Load model wothout classifier/fully connected layers
RESNET_model = Model(inputs=model.input, outputs=model.get_layer('dense').output)

RESNET_model.summary()  # Trainable parameters will be 0

x_subset = x_train[0:200]
y_subset = y_train[0:200]
# Now, let us use features from convolutional network for RF
feature_extractor = RESNET_model.predict(x_subset)

import seaborn as sns
import matplotlib.patheffects as PathEffects

sns.set_style('whitegrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.1,
                rc={"lines.linewidth": 1.5})

RS = 123


def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(16, 16))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=10, c=palette[colors.astype(int)], s=40)
    ax.axis('on')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):
        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=10)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=3, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts

import pandas as pd

# tsne = TSNE(n_components=2, random_state=0, learning_rate=10)
# tsne_out = tsne.fit_transform(feature_extractor)
# tsne_frame = pd.DataFrame(np.row_stack(tsne_out), columns=['x', 'y'])
# tsne_frame['label'] = y_subset  # baseline diagnosis labels
# tsne_framesub = tsne_frame.copy()
# sns.set_context("notebook", font_scale=1.1)
# sns.set_style("ticks")
# sns.lmplot(x='x', y='y',
#            data=tsne_framesub,
#            fit_reg=False, legend=True,
#            height=8,
#            hue='label',
#            scatter_kws={"s": 50, "alpha": 0.2})

tsne = TSNE(n_components=2).fit_transform(feature_extractor)

fashion_scatter(tsne, y_subset)
plt.show()
