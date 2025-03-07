"""
Use the trained model to detect anomalies (potential seismic events.)
"""
import glob
import json
import numpy as np
import os
import sys

from keras.models import load_model
from sklearn.neighbors import KernelDensity
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from das_anomaly import check_if_anomaly


# Size of the input images 
size = 128

# Define the path to the results 
results_path = '/globalscratch/ahmad9/caserm/spectrum_analysis/results/'
model_path = results_path + f'model_1_{size}.h5'
loaded_model = load_model(model_path)

# Define the path to power spectral density (PSD) plots 
data_spool = 'UTC-YMD20220617-HMS155316.989'
root_dir = f'/globalscratch/ahmad9/caserm/spectrum_analysis/spectrum_plots/{data_spool}/'

# Define your desired threshold for density score
density_threshold = 15934.15

# Define generators for training, validation and, anomaly data.
batch_size = 64
datagen = ImageDataGenerator(rescale=1./255)

# Create the train generator (with same parameters as for the trained model)
train_generator = datagen.flow_from_directory(
    "",
    target_size=(size, size),
    batch_size=batch_size,
    class_mode='input'
    )

# Define the autoencoder
# Encoder
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(size, size, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

# Decoder
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))

model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Read the history of the trained model 
with open(os.path.join(results_path, 'history_1_128.json'), 'r') as json_file:
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

# Write whether the image is an anomaly in a text file
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        # Construct the glob pattern for PSD file paths in the current folder
        spectrum_file_pattern = os.path.join(folder_path, '*')
        spectrum_file_paths = glob.glob(spectrum_file_pattern)
    # SSave the the results in a text file 
    with open(results_path + data_spool + '/' + str(folder_name) + '_output_model_1_128_anomaly.txt', 'w') as file:
        for i in range (0,len(spectrum_file_paths)):       
            print(
                f"Line {i}, image {spectrum_file_paths[i]}: 
                  {check_if_anomaly(encoder_model=encoder_model, size=size, img_path=spectrum_file_paths[i], density_threshold=density_threshold, kde=kde)}", 
                  file=file
                  )
