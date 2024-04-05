# USAGE
# python fo_load_dataset.py

# imports
from helpers import config
import fiftyone as fo
import fiftyone.brain as fob
import pandas as pd
import logging
import os


# logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


def add_samples(annotations: dict, name: str) -> list:
    # initialize the samples list
    s = []

    # iterate over each annotation and add the sample
    for filename, label in annotations.items():
        sample = fo.Sample(filepath=os.path.join(config.IMAGE_PATH, filename))
        sample[name] = fo.Classification(label=label)
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

# create data samples for FiftyOne
logging.info("adding FiftyOne dataset samples")
samples = add_samples(train_annotations, 'train') + \
          add_samples(val_annotations, 'val') + \
          add_samples(test_annotations, 'test')

# create the FiftyOne dataset
logging.info("creating the FiftyOne dataset")
dataset = fo.Dataset("ip102-classification-5")
dataset.add_samples(samples)

# add image viz
# fob.compute_visualization(dataset, brain_key="img_viz")

# launch the app while specifying the custom dataset and keep the session open
logging.info("launching FiftyOne!")
session = fo.launch_app(dataset, port=6161)
session.wait()
logging.info("session closed")
