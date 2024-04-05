# USAGE
# python classify_image.py --model model/insect_pest_model.h5 --image sample_data/71.png

# import the necessary packages
from tensorflow.keras.models import load_model
from helpers import config
import tensorflow as tf
import numpy as np
import argparse
import logging
import time
import cv2

# logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# construct the argument parser and parse the arguments
logging.info("parsing command line arguments")
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path to the tensorflow model")
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
args = vars(ap.parse_args())

# define output image margin and spacing
x_margin = 10
y_spacing = 30

# load the model
logging.info("loading the model...")
model = load_model(args["model"])

# load the input image and clone it for annotation
logging.info("loading the image")
image = cv2.imread(args["image"])
orig = image.copy()
 
# the model was trained on RGB ordered images but OpenCV represents
# images in BGR order, so swap the channels, and then resize to
# the image size we determined
logging.info("preprocessing - swapping color channels and resizing")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))

# add batch dimension to the image to make it compatible, convert the
# image to a floating point data type, and perform mean subtraction
logging.info("preprocessing - adding a batch dimension, convering to float32, and "
             "conducting mean subtraction")
image = np.expand_dims(image, axis=0)
image = image.astype("float32")
mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")
image -= mean

# make a prediction
logging.info("making a prediction")
start = time.time()
prediction = model.predict([image], batch_size=1)
end = time.time()
logging.info("classification took {:.4f} seconds...".format(end - start))

# grab the class label index and class label
logging.info("analyzing results")
predicted_label_index = prediction.argmax(axis=1)[0]
predicted_label = list(config.CLASSES_MAP.items())[predicted_label_index][1]

# grab the probability of the predicted class
predicted_probability = prediction[0][predicted_label_index]

# grab the top 5 probabilities
top5_indices = prediction[0].argsort()[-5:][::-1]
top5_probabilities = prediction[0][top5_indices]

# print and display the top 5 results
logging.info("Top-5 results:")
for i, (idx, proba) in enumerate(zip(top5_indices, top5_probabilities), 1):
    # grab the class label and generate output text
    label = list(config.CLASSES_MAP.items())[idx][1]
    text = "Label: {}, {:.2f}%".format(label, proba * 100)

    # write the predictions on the output image
    cv2.putText(orig, text, (x_margin, y_spacing * i), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    # display the result on the terminal
    logging.info("RESULT {}. {}".format(i, text))

# show the output image
cv2.imshow("Output", orig)
cv2.waitKey(0)
