"""
Use the trained model to detect anomalies (potential seismic events.)
"""
import glob
import json
import numpy as np
import os
import shutil

from keras.models import load_model
from mpi4py import MPI
from sklearn.neighbors import KernelDensity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from das_anomaly import check_if_anomaly
from das_anomaly.settings import SETTINGS


# Size of the input images 
size = SETTINGS.SIZE

# Define the path to power spectral density (PSD) plots 
psd_dir = SETTINGS.PSD_DIR

# Define the path to the results 
results_path = SETTINGS.RESULTS_PATH
model_path = results_path + f'model_{size}.h5'
loaded_model = load_model(model_path)
# Define the destination directory
destination_dir = os.path.join(results_path, "copied_detected_anomalies")

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Define the desired threshold for density score (from validate_and_plot_density step)
density_threshold = SETTINGS.DENSITY_THRESHOLD

# Define generators for training, validation and, anomaly data.
batch_size = SETTINGS.BATCH_SIZE
datagen = ImageDataGenerator(rescale=1./255)

# Create the train generator (with same parameters as for the trained model)
train_images_path = SETTINGS.TRAIN_IMAGES_PATH 
train_generator = datagen.flow_from_directory(
    train_images_path,
    target_size=(size, size),
    batch_size=batch_size,
    class_mode='input'
    )

# Read the history of the trained model 
with open(os.path.join(results_path, f'history_{size}.json'), 'r') as json_file:
    history_dict = json.load(json_file)

# Extract the encoder network, with trained weights
encoder_model = Sequential()
# Add the convolutional layer without weights
encoder_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(size, size, 3)))
# Set the weights from the corresponding layer of the loaded model
encoder_model.layers[-1].set_weights(loaded_model.layers[0].get_weights())
encoder_model.add(MaxPooling2D((2, 2), padding='same'))
encoder_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
encoder_model.layers[-1].set_weights(loaded_model.layers[2].get_weights())
encoder_model.add(MaxPooling2D((2, 2), padding='same'))
encoder_model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
encoder_model.layers[-1].set_weights(loaded_model.layers[4].get_weights())
encoder_model.add(MaxPooling2D((2, 2), padding='same'))

# Calculate KDE of latent space using sklearn and determine if PSD is an anomaly
encoded_images = encoder_model.predict(train_generator, verbose=0)
encoder_output_shape = encoder_model.output_shape 
out_vector_shape = encoder_output_shape[1]*encoder_output_shape[2]*encoder_output_shape[3]

encoded_images_vector = [np.reshape(img, (out_vector_shape)) for img in encoded_images]

# Fit KDE to the image latent data
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(encoded_images_vector)

# Initiate MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_mpi = comm.Get_size()

splits = len(os.listdir(psd_dir))

# Write whether the image is an anomaly in a text file
for i in range(rank, splits, size_mpi):
    folder_name = os.listdir(psd_dir)[i]
    folder_path = os.path.join(psd_dir, folder_name)
    if os.path.isdir(folder_path):
        # Construct the glob pattern for PSD file paths in the current folder
        spectrum_file_pattern = os.path.join(folder_path, '*')
        spectrum_file_paths = glob.glob(spectrum_file_pattern)
    # Save the the results in a text file 
    with open(f"{results_path}/{folder_name}_output_model_{size}_anomaly.txt", 'w') as file:
        for j in range(len(spectrum_file_paths)):       
            anomaly_flag = check_if_anomaly(
                encoder_model=encoder_model,
                size=size,
                img_path=spectrum_file_paths[j],
                density_threshold=density_threshold,
                kde=kde
            )
            print(f"Line {j}, image {spectrum_file_paths[j]}: {anomaly_flag}", file=file)
            
            # copy the detected anomaly to anomaly folder
            if anomaly_flag == "The image is an anomaly":
                shutil.copy(spectrum_file_paths[j], destination_dir)
