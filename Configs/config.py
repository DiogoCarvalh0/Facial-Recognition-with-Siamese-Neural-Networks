import time

# Path to the images, based on the 'main.py' file
POSITIVE_LABEL_PATH = './Data/positive/'
NEGATIVE_LABEL_PATH = './Data/negative/'
ANCHOR_LABEL_PATH = './Data/anchor/'

# Number of images to use to train
NR_IMAGES_TO_USE = 300

# Number of images to use as anchor for predictions (after training)
ANCHOR_NR_IMAGES_FOR_PREDICTION = 300

# Shape of input
INPUT_IMAGE_SIZE = (105, 105)

# Batch size and number of epochs
BATCH_SIZE = 16
EPOCHS = 5

# Model name to use when saving
MODEL_NAME = f'siamese-model-{int(time.time())}'

########## For Prediction ########
# Metric above which a prediction is considered positive
DETECTION_THRESHOLD = 0.5
# Propotion of positive predictions needed for a positive label
VERIFICATION_THRESHOLD = 0.75

