# USAGE
# python fo_load_dataset.py

# imports
from tensorflow.keras.models import load_model
from helpers import config
import fiftyone as fo
import pandas as pd
import numpy as np
import logging
import uuid
import cv2
import os

# logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# load the model
logging.info("loading the model...")
model = load_model(os.path.join('model', 'insect_pest_model.h5'))


def add_samples(annotations: dict, name: str) -> list:
    # initialize the samples list
    s = []

    # num images
    num_images = len(annotations.items())

    # iterate over each annotation and add the sample
    for i, (filename, label) in enumerate(annotations.items()):
        # create a sample
        image_path = os.path.join(config.IMAGE_PATH, filename)
        sample = fo.Sample(filepath=image_path)

        # set the sample name
        sample[name] = fo.Classification(label=label)

        # training and validation are 100% confident (unless we find any outliers later)
        if name == 'train' or name == 'val':
            sample['confidence'] = 1.0

        # perform inference on testing samples
        if name == 'test':
            # load the input image and clone it for annotation
            image = cv2.imread(image_path)
            orig = image.copy()

            # the model was trained on RGB ordered images but OpenCV represents
            # images in BGR order, so swap the channels, and then resize to
            # the image size we determined
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))

            # add batch dimension to the image to make it compatible, convert the
            # image to a floating point data type, and perform mean subtraction
            image = np.expand_dims(image, axis=0)
            image = image.astype("float32")
            mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")
            image -= mean

            # make a prediction
            prediction = model.predict([image], batch_size=1)

            # grab the class label index and class label
            predicted_label_index = prediction.argmax(axis=1)[0]
            predicted_label = list(config.CLASSES_MAP.items())[predicted_label_index][1]

            # grab the probability of the predicted class and set the confidence and label
            # fields
            predicted_probability = prediction[0][predicted_label_index]
            sample['confidence'] = predicted_probability
            sample['label'] = predicted_label

        # add the sample
        logging.info(f"{name} - processed {i+1} of {num_images}")
        s.append(sample)

    return s


# use pandas to read the class labels where each filename and class label looks like
# 00036.jpg 0
# the delimiter is a *space* and not a *comma*
logging.info("reading label txt files")
trainDF = pd.read_csv(config.TRAIN,
                      sep=" ",
                      header=None, names=["image_filename", "label"])
valDF = pd.read_csv(config.VALID,
                    sep=" ",
                    header=None, names=["image_filename", "label"])
testDF = pd.read_csv(config.TEST,
                     sep=" ",
                     header=None, names=["image_filename", "label"])

# change the labels from 0-indexed to 1-indexed since
# these were lagged by 1 when they were loaded above
trainDF["label"] = trainDF["label"] + 1
valDF["label"] = valDF["label"] + 1
testDF["label"] = testDF["label"] + 1

# create a dict from the dataframe
logging.info("creating annotation dictionaries")
train_annotations = trainDF.set_index('image_filename')['label'].to_dict()
val_annotations = valDF.set_index('image_filename')['label'].to_dict()
test_annotations = testDF.set_index('image_filename')['label'].to_dict()

# remap integer labels to string labels from the config
logging.info("remapping integer labels to string labels from the config")
train_annotations = {p: config.CLASSES_MAP[c] for p, c in train_annotations.items()
                     if c in config.CLASSES_MAP}
val_annotations = {p: config.CLASSES_MAP[c] for p, c in val_annotations.items()
                   if c in config.CLASSES_MAP}
test_annotations = {p: config.CLASSES_MAP[c] for p, c in test_annotations.items()
                    if c in config.CLASSES_MAP}

# create the FiftyOne dataset
logging.info("creating the FiftyOne dataset")
dataset = fo.Dataset(f"ip102-classification-{uuid.uuid4()}")

# add a field to the dataset
dataset.add_sample_field("confidence", fo.FloatField)

# create data samples for FiftyOne
logging.info("adding FiftyOne dataset samples")
samples = add_samples(train_annotations, 'train') + \
          add_samples(val_annotations, 'val') + \
          add_samples(test_annotations, 'test')
dataset.add_samples(samples)

# launch the app while specifying the custom dataset and keep the session open
logging.info("launching FiftyOne!")
session = fo.launch_app(dataset, port=6161)
session.wait()
logging.info("session closed")
