# USAGE
# python finetune.py

# import the necessary packages
from helpers.clr_callback import CyclicLR
from helpers import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.python.util import deprecation
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from imutils import paths
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import pandas as pd
import numpy as np
import os

# logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# cpu backend
tf.config.set_visible_devices([], 'GPU')  # Hide GPU devices
tf.config.set_visible_devices(tf.config.list_physical_devices('CPU'), 'CPU')


def plot_training(H, N, plotPath):
    # plot the training history and save it to disk
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)


# turn off the deprecation warnings and logs to keep the
# console clean for convenience
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# define the number of classes and batch size
NUM_CLASSES = len(config.CLASSES.items())
BATCH_SIZE = 32

# load the training dataframe, extract the label column, and
# binarize the labels
trainDF = pd.read_csv(config.TRAIN_DF)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainDF["label"])

# account for skew (class imbalance) in the data
classTotals = trainY.sum(axis=0)
classWeight = {}
for i, total in enumerate(classTotals):
    classWeight[i] = classTotals.max() / total

# construct the paths to the training, validation, and testing
# data directories
trainPath = os.path.sep.join([config.IMAGE_PATH_FILTERED, "training"])
valPath = os.path.sep.join([config.IMAGE_PATH_FILTERED, "validation"])
testPath = os.path.sep.join([config.IMAGE_PATH_FILTERED, "testing"])

# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
totalTest = len(list(paths.list_images(testPath)))

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# initialize the validation/testing data augmentation object
valAug = ImageDataGenerator()

# set the mean subtraction value for each of the data augmentation objects so that mean
# subtraction is handled as the augmented images are generated
trainAug.mean = config.MEAN
valAug.mean = config.MEAN

# initialize the training generator
trainGen = trainAug.flow_from_directory(
    trainPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=True,
    batch_size=BATCH_SIZE)

# initialize the validation generator
valGen = valAug.flow_from_directory(
    valPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE)

# initialize the testing generator
testGen = valAug.flow_from_directory(
    testPath,
    class_mode="categorical",
    target_size=(224, 224),
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE)

# load MobileNetV2 without the head layers
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# construct the head of the model that will be placed on top of the base model with a
# fully connected head and softmax classifier
headModel = baseModel.output
headModel = GlobalAveragePooling2D()(headModel)
headModel = Dense(NUM_CLASSES, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# freeze the base network; through the process of fine-tuning we'll be training the head
# of the network
baseModel.trainable = False

# compile our model after we've set our setting layers to be non-trainable; use the Adam
# learning rate optimizer which adapts the learning rate for each parameter individually;
# we use categorical crossentropy because this is a multi-class problem
logging.info("[INFO] compiling model...")
opt = Adam(lr=config.MIN_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# define a learning rate range to train over, so compute the step size, and initialize
# the cyclic learning rate, a method that helps the model converge faster
stepSize = config.STEP_SIZE * (totalTrain // BATCH_SIZE)
clr = CyclicLR(
    mode=config.CLR_METHOD,
    base_lr=1e-05,
    max_lr=1e-03,
    step_size=stepSize)

# train the network (note the callback for cyclical learning rate which will run at each
# epoch)
logging.info("[INFO] training network...")
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // BATCH_SIZE,
    validation_data=valGen,
    validation_steps=totalVal // BATCH_SIZE,
    callbacks=[clr],
    epochs=3,
    class_weight=classWeight)

# reset the testing generator and evaluate the network after
# fine-tuning just the network head
logging.info("[INFO] evaluating after fine-tuning network head...")
testGen.reset()
predIdxs = model.predict_generator(testGen, steps=(totalTest // BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
logging.info(classification_report(testGen.classes, predIdxs,
                                   target_names=config.CLASSES.keys()))
plot_training(H, 3, config.WARMUP_PLOT_PATH)

# reset our data generators
trainGen.reset()
valGen.reset()

# set the base model's trainable parameters to true
baseModel.trainable = True

# freeze all the layers before the 100th layer
for layer in baseModel.layers[:100]:
    layer.trainable = False

# loop over each layer in the model and show
# if it is trainable or not
for layer in baseModel.layers:
    logging.info("{}: {}".format(layer, layer.trainable))

# for the changes to the model to take effect we need to recompile the model, this time
# using Adam with a *very* small learning rate
logging.info("[INFO] re-compiling model...")
opt = Adam(config.MIN_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model again, this time fine-tuning *both* the final set
# of CONV layers along with our set of FC layers
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // BATCH_SIZE,
    validation_data=valGen,
    validation_steps=totalVal // BATCH_SIZE,
    class_weight=classWeight,
    epochs=10)

# reset the testing generator and then use our trained model to
# make predictions on the data
logging.info("[INFO] evaluating after fine-tuning network...")
testGen.reset()
predIdxs = model.predict_generator(testGen, steps=(totalTest // BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
logging.info(classification_report(testGen.classes, predIdxs,
                                   target_names=config.CLASSES.keys()))
plot_training(H, 10, config.UNFROZEN_PLOT_PATH)

# serialize the model to disk
logging.info("[INFO] serializing network...")
model.save(config.MODEL_PATH)
