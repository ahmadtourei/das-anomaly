# Path to the data 
DATA_PATH = '/path/to/the/DAS/data'
# Start and end time for the data spool
T_1 = "2023-01-23 00:00:00"
T_2 = "2023-02-00 00:00:00"

# Size of the input images 
SIZE = 512
# Desired density threshold based on density score of background noise data and known anomolous data
DENSITY_THRESHOLD = 20_485.340556576524
# Batch size for the train generator
BATCH_SIZE = 64

# Set parameters for preprocessing the data
STEP_MULTIPLE = 2  # gauge length to channel spacing ratio
START_CHANNEL = 0
END_CHANNEL = 800
TIME_WINDOW = 2  # sec.
TIME_OVERLAP = 1  # sec.
DPI = 300  # saved image quality

# Define the path to power spectral density (PSD) plots 
PSD_PATH = '/path/to/PSD/plots'

# Define number of train+test and the ratio of the test to train
NUM_IMAGE = 1000
RATIO = 0.2

# Define the directory path for training, testing, and known anomalous PSD images
TRAIN_IMAGES_PATH = './data/training_dataset/'
TEST_IMAGES_PATH = './data/training_dataset/'
ANOMALY_IMAGES_PATH = './data/anomalous_dataset/'

# Number of epochs for training
NUM_EPOCH = 250

# Define the path to the trained model
TRAINED_PATH = './data/'

# Define the path to the detected anomalies results 
RESULTS_PATH = '/path/to/saved/results/from/detect_anomalies/'
RESULTS_FOLDER_NAME = 'results_folder'
