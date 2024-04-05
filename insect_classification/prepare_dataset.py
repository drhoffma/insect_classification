# USAGE
# python prepare_dataset.py

# import the necessary packages
from helpers import config
import logging
import pandas as pd
import shutil
import os

# logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


def get_top_classes(df, N=5, labels=None):
    # if no labels are provided explicitly extract the top-5 labels
    # from the pandas data frame and store them in a list
    if labels is None:
        top5Labels = pd.value_counts(df["label"], sort=True).head(N).index.tolist()

        # filter out the data points corresponding to these labels
        modifiedDF = df[df["label"].isin(top5Labels)]
    
    # if labels are provided, find all data points that 
    # belong to the supplied labels
    else:
        modifiedDF = df[df["label"].isin(labels)]

    # return the modified data frame
    return modifiedDF


# load the .txt files in pandas data frame providing the
# appropriate column names
logging.info("loading data files...")
trainDF = pd.read_csv(config.TRAIN,
                      sep=" ",
                      header=None, names=["image_path", "label"])
valDF = pd.read_csv(config.VALID,
                    sep=" ",
                    header=None, names=["image_path", "label"])
testDF = pd.read_csv(config.TEST,
                     sep=" ",
                     header=None, names=["image_path", "label"])

# change the labels from 0-indexed to 1-indexed since 
# these were lagged by 1 when they were loaded above 
trainDF["label"] = trainDF["label"] + 1
valDF["label"] = valDF["label"] + 1
testDF["label"] = testDF["label"] + 1

# grab the top 5 labels from the training data frame to filter the
# *data points* accordingly
logging.info("preparing data frames...")
trainDFTop = get_top_classes(trainDF)

# extract the unique labels from trainDFTop, and
# convert the numpy array to a list
topTrainLabels = trainDFTop["label"].unique().tolist()

# filter top-5 labels from validation and testing data
valDFTop = get_top_classes(valDF, labels=topTrainLabels)
testDFTop = get_top_classes(testDF, labels=topTrainLabels)

# create 3-tuple of the 2-tuple dataset split 
# name vs. data frame so that we can later loop over them
splits = (
    ("training", trainDFTop),
    ("testing", testDFTop),
    ("validation", valDFTop)
)

# loop over the data splits
for (split, df) in splits:
    # loop over the rows of the data frame
    logging.info("processing '{} split'...".format(split))
    for (_, row) in df.iterrows():
        # grab the image path, label, add the original base path, and
        # construct output image and output directory path
        imagePath = os.path.sep.join([config.IMAGE_PATH, row.values[0]])
        label = str(row.values[1])
        dirPath = os.path.sep.join([config.IMAGE_PATH_FILTERED, split, label])

        # if the output directory does not exist, create it
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        # construct the path to the output image file and copy it
        p = os.path.sep.join([dirPath, row.values[0]])
        shutil.copy2(imagePath, p)
 
# serialize the filtered out data frames 
logging.info("serializing data frames...")
trainDFTop.to_csv(config.TRAIN_DF, index=False)
testDFTop.to_csv(config.TEST_DF, index=False)
valDFTop.to_csv(config.VALID_DF, index=False)

# calculate the split percentages
logging.info("Images are split according to the following percentages")
numTrainImages = len(trainDFTop)
numTestImages = len(testDFTop)
numValImages = len(valDFTop)
totalImages = numTrainImages + numValImages + numTestImages
logging.info(f"Training [{(numTrainImages/totalImages)*100:.0f}]")
logging.info(f"Testing [{(numTestImages/totalImages)*100:.0f}]")
logging.info(f"Validation [{(numValImages/totalImages)*100:.0f}]")
