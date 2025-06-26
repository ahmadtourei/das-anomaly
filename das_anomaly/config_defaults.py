"""
Global configuration for the DAS-anomaly demo pipeline
------------------------------------------------------


DO NOT EDIT. Edit the config_user.py script instead.
"""

# Data unit - should be either velocity or strain_rate
DATA_UNIT = "velocity"
# Path to the DAS data (main spool)
DATA_PATH = "/path/to/the/das/data"
# Path to the background noise data examples
BN_DATA_PATH = "/path/to/the/das/background_noise_data"
# Start and end time for the data spool
T_1 = "2022-12-01 00:00:00"
T_2 = "2022-12-08 00:00:00"

# Size of the input/output images
SIZE = 128
# Batch size for the train generator
BATCH_SIZE = 64
# Desired density threshold based on density score
# of background noise data and known anomolous data
DENSITY_THRESHOLD = 5_000
# Empirically choose 95th-percentile amplitude for a background noise
# (anomaly-free) PSD. Used as vmax in imshow to keep colour scaling
# consistent across plots.
CLIP_VALUE_MAX = 1e-6

# Set parameters for preprocessing the data
MIN_FREQ = 0
MAX_FREQ = 250
STEP_MULTIPLE = 2  # gauge length to channel spacing ratio
START_CHANNEL = 0
END_CHANNEL = 800
TIME_WINDOW = 2  # sec.
TIME_OVERLAP = 1  # sec.
DPI = 300  # saved image quality

# Define the path to power spectral density (PSD) plots
PSD_PATH = "/path/to/PSD/plots"

# Define number of train+test and the ratio of the test to train
NUM_IMAGE = 1000
RATIO = 0.2

# Define the directory path for training, testing, and known anomalous PSD images
TRAIN_IMAGES_PATH = "/path/to/training/PSD/plots/"
TEST_IMAGES_PATH = "/path/to/testing/PSD/plots/"
ANOMALY_IMAGES_PATH = "/path/to/PSD/plots/with/anomalies/"

# Number of epochs for training
NUM_EPOCH = 500

# Define the path to the trained model
TRAINED_PATH = "/path/to/saved/results/trained/model/"

# Define the path to the detected anomalies results
RESULTS_PATH = "/path/to/saved/results/from/detect_anomalies/"
RESULTS_FOLDER_NAME = "results_folder"
