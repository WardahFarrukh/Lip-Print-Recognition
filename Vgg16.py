import tensorflow as tf
from keras import metrics
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from vit_keras import vit, utils
import scikitplot as skplt
import os.path
import math
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

# loading up our datasets
from tensorflow.python.keras.utils.np_utils import to_categorical

TRAINING_DATA_DIR = "C:/Users/27820/Desktop/Campus Stuff/MASTERS/trial/train"
TEST_DATA_DIR = "C:/Users/27820/Desktop/Campus Stuff/MASTERS/trial/test"
VAL_DATA_DIR = "C:/Users/27820/Desktop/Campus Stuff/MASTERS/trial/validation"

TRAIN_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TRAINING_DATA_DIR)])
TEST_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(TEST_DATA_DIR)])
VAL_SAMPLES = sum([len(filenames) for dirpath, dirnames, filenames in os.walk(VAL_DATA_DIR)])

print("Number of training samples : {}".format(TRAIN_SAMPLES))
print("Number of test samples : {}".format(TEST_SAMPLES))

NUM_CLASSES = len(next(os.walk(TRAINING_DATA_DIR))[1])
print("Number of classes: {}".format(NUM_CLASSES))

IMG_WIDTH, IMG_HEIGHT = 260, 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
                                   # horizontal_flip=True,
                                   # rotation_range=15)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(TRAINING_DATA_DIR,
                                                    target_size=(IMG_WIDTH,
                                                                 IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    # Training pictures are shuffled to introduce more randomness
                                                    # during the training process
                                                    class_mode='categorical')

test_generator = val_datagen.flow_from_directory(TEST_DATA_DIR,
                                                 target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False,
                                                 class_mode='categorical')

labels = train_generator.class_indices
# print(labels)

test_labels = test_generator.classes

base_model = vit.vit_b16(
    image_size=224,
    activation='softmax',
    pretrained=True,
    include_top=True,
    pretrained_top=False
)
#
# # Freeze all layers except for the last 8 layers that will be retrained
for layer in base_model.layers:
    layer.trainable = False
#
custom_model = GlobalAveragePooling2D()(base_model.output)
custom_model = Dense(1024, activation='relu')(custom_model)
custom_model = Dense(512, activation='relu')(custom_model)
#
# # Final layer : the number of neurons is equal to the number of classes we want to predict
# # Since we have more than two classes, we choose 'softmax' as the activation function.
custom_model = Dense(NUM_CLASSES, activation='softmax')(custom_model)
#
model = Model(inputs=base_model.input, outputs=custom_model)
#
model.summary()
#
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['acc'])
#
history = model.fit_generator(train_generator,
                              steps_per_epoch=math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE),
                              validation_data=test_generator,
                              validation_steps=math.ceil(float(TEST_SAMPLES) / BATCH_SIZE),
                              epochs=100)
#
model.save('ViT.h5', save_format='h5')
#
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(len(acc))
plt.plot(epochs, acc, "r", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "r", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.show()

# model = load_model('modelvgg16-finetuned.h5')
#
# score = model.evaluate_generator(test_generator, math.ceil(float(TEST_SAMPLES) / BATCH_SIZE))
# print("Test accuracy : {:.9f} %".format(score[1] * 100))
#
# num_classes = len(test_generator.class_indices)
# test_labels = to_categorical(test_labels, num_classes=num_classes)
#
# predictions = np.round(model.predict(test_generator, math.ceil(float(TEST_SAMPLES) / BATCH_SIZE)), 0)
# classification_metrics = metrics.classification_report(test_labels, predictions)
# print(classification_metrics)
#
# categorical_test_labels = pd.DataFrame(test_labels).idxmax(axis=1)
# categorical_preds = pd.DataFrame(predictions).idxmax(axis=1)
#
#
# # To get better visual of the confusion matrix:
# def showconfusionmatrix(cm):
#     plt.imshow(cm)
#     plt.title("Confusion matrix")
#     plt.colorbar()
#     plt.show()
#
#
# cm = confusion_matrix(categorical_test_labels, categorical_preds)
# showconfusionmatrix(cm)
#
# pmacro = precision_recall_fscore_support(categorical_test_labels, categorical_preds, average='macro')
# pmicro = precision_recall_fscore_support(categorical_test_labels, categorical_preds, average='micro')
#
# print(pmacro)
# print(pmicro)
#
# testlabel = test_generator.classes
#
# print(test_generator.classes.shape)
# print(predictions.shape)
# skplt.metrics.plot_roc(testlabel, predictions, title="(resnet50 + svm) ROC CURVE", classes_to_plot=[0, "cold"])
# skplt.metrics.plot_precision_recall(testlabel, predictions, title="(vgg16) Precision-Recall CURVE",
#                                     classes_to_plot=[0, "cold"])
# plt.show()
#
# fpr, tpr, threshold = roc_curve(categorical_test_labels, categorical_preds, pos_label=1)
# fnr = 1-tpr
# eer = fpr[np.nanargmin(np.absolute((fnr) - fpr))]
# print(eer)
