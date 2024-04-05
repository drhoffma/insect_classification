# import the necessary packages
import numpy as np
import os

# define the ImageNet mean subtraction (in RGB order)
MEAN = np.array([123.68, 116.779, 103.939], dtype="float32")

# initialize the dictionary to map the class names to integers
CLASSES = {"Cicadellidae": 102,
           "Aphids": 25,
           "Blister beetle": 52,
           "Lycorma delicatula": 68,
           "Miridae": 71
}
CLASSES_MAP = {v: k for k, v in CLASSES.items()}

# path to the IP102 insects dataset
DATASET_PATH = os.path.join("..", "dataset", "ip102_v1.1")

# specify the paths of the text files containing the image paths
# and their labels
TRAIN = os.path.join(DATASET_PATH, "train.txt")
VALID = os.path.join(DATASET_PATH, "val.txt")
TEST = os.path.join(DATASET_PATH, "test.txt")

# specify the paths of csv files of the filtered out dataframes
TRAIN_DF = os.path.join(DATASET_PATH, "train_df.csv")
VALID_DF = os.path.join(DATASET_PATH, "valid_df.csv")
TEST_DF = os.path.join(DATASET_PATH, "test_df.csv")

# specify the path where original and pruned images reside
IMAGE_PATH = os.path.join(DATASET_PATH, "images")
IMAGE_PATH_FILTERED = os.path.join(DATASET_PATH, "data")

# define the minimum learning rate, maximum learning rate,
# step size, and Cyclical Learning Rate (CLR) method
MIN_LR = 1e-5
MAX_LR = 1e-2
STEP_SIZE = 8
CLR_METHOD = "triangular"

# set the path to the serialized model after training
MODEL_PATH = os.path.join("model", "insect_pest_model.h5")

# define the path to the output training history plots
UNFROZEN_PLOT_PATH = os.path.join("output", "unfrozen.png")
WARMUP_PLOT_PATH = os.path.join("output", "warmup.png")
