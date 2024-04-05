# USAGE
# python fo_load_dataset_w_confidence_batch.py

# imports
from tensorflow.keras.models import load_model
from helpers import config
import fiftyone as fo
import fiftyone.brain as fob
import pandas as pd
import numpy as np
import logging
import uuid
import cv2
import os

# logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# define the batch size
BATCH_SIZE = 16

# load the model
logging.info("loading the model...")
model = load_model(os.path.join('model', 'insect_pest_model.h5'))


def add_samples(annotations: dict, name: str) -> list:
    def batcher(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    # initialize the samples list
    s = []

    # num images
    num_images = len(annotations.items())

    # convert the dictionary intio a list of tuples for batching
    annotation_list = list(annotations.items())

    # iterate over each annotation batch and add the sample(s)
    for batch_idx, batch in enumerate(batcher(annotation_list, BATCH_SIZE)):
        for i, (filename, label) in enumerate(batch):
            # handle the case when the last batch is smaller than BATCH_SIZE
            if filename is None or label is None:
                break

            # (unless we find any outliers later)
            if name == 'train' or name == 'val':
                # create a sample
                image_path = os.path.join(config.IMAGE_PATH, filename)
                sample = fo.Sample(filepath=image_path)

                # set the sample name
                sample[name] = fo.Classification(label=label)

                # set the confidence to one since this is training or validation
                sample['confidence'] = 1.0
                s.append(sample)

            # perform inference on testing samples
            elif name == 'test':
                images = []
                for filename, _ in batch:
                    image_path = os.path.join(config.IMAGE_PATH, filename)
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (224, 224))
                    images.append(image)

                images = np.array(images)
                images = images.astype("float32")
                mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")
                images -= mean

                # make predictions for the batch of images
                predictions = model.predict(images, batch_size=BATCH_SIZE)

                for prediction, (filename, _) in zip(predictions, batch):
                    # grab the class label index and class label
                    predicted_label_index = prediction.argmax(axis=0)
                    predicted_label = list(
                        config.CLASSES_MAP.items())[predicted_label_index][1]

                    # create the sample and set the name
                    sample = fo.Sample(filepath=os.path.join(config.IMAGE_PATH, filename))
                    sample[name] = fo.Classification(label=predicted_label)

                    # calculate the prediction probability and set the confidence field
                    predicted_probability = prediction[predicted_label_index]
                    sample['confidence'] = predicted_probability

                    # set the label and add the sample
                    sample['label'] = predicted_label
                    s.append(sample)

            # logging message
            # todo improvement use the *progressbar2* library
            if i+1 == BATCH_SIZE:
                logging.info(f"{name} - processed batch {batch_idx + 1} of "
                             f"{num_images // BATCH_SIZE}")

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

# compute uniqueness
fob.compute_uniqueness(dataset)

# create a view of similar images (least unique first)
similar_view = dataset.sort_by("uniqueness")

# launch the app while specifying the custom dataset and keep the session open
logging.info("launching FiftyOne!")
session = fo.launch_app(dataset, port=5151)

# load the similar view
session.view = similar_view

# wait until app closed
session.wait()
logging.info("session closed")
