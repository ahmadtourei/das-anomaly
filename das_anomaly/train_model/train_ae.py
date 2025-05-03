"""
This script uses unsupervised autoencoders to train a deep learning model for anomaly detection in images.

For microseismic event detection, images can be the power spectral density (PSD) plots.

We will consider the bottleneck layer output from our autoencoder as the latent space.
Using the reconstruction error and kernel density estimation (KDE) based on the vectors in the latent space, anomalies can be detected.
"""
import json
import os

from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from das_anomaly import plot_train_test_loss, encoder, decoder
from das_anomaly.settings import SETTINGS


# Specify path to save model results 
trained_path = SETTINGS.TRAINED_PATH

# Size of the input images and number of epoches for training
size = SETTINGS.SIZE
num_epoch = SETTINGS.NUM_EPOCH

# Define generators for training, validation, and anomaly data.
batch_size = SETTINGS.BATCH_SIZE
datagen = ImageDataGenerator(rescale=1.0 / 255)

# Path to training PSD plots (seen data)
train_path = SETTINGS.TRAIN_IMAGES_PATH
num_train_data = sum(len(files) for _, _, files in os.walk(train_path))
train_generator = datagen.flow_from_directory(
    train_path, target_size=(size, size), batch_size=batch_size, class_mode="input"
)

# Path to testing PSD plots (unseen data)
test_path = SETTINGS.TEST_IMAGES_PATH
num_test_data = sum(len(files) for _, _, files in os.walk(test_path))
validation_generator = datagen.flow_from_directory(
    test_path, target_size=(size, size), batch_size=batch_size, class_mode="input"
)

# Define the autoencoder.
# Encoder
model = encoder(size)
# Save the encoder in TF's SavedModel format
model.save(os.path.join(trained_path, f"encoder_model_{size}"), save_format="tf")
# Decoder
model = decoder(model)

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

model.summary()

# Fit the model
history = model.fit(
    train_generator,
    steps_per_epoch=num_train_data // batch_size,
    epochs=num_epoch,
    validation_data=validation_generator,
    validation_steps=num_test_data // batch_size,
    shuffle=True,
)

# Save the model in h5 format
model.save(trained_path + f"model_{size}.h5")

# Save the history as well
history_dict = history.history
history_json = json.dumps(history_dict)
with open(trained_path + f"history_{size}.json", "w") as json_file:
    json_file.write(history_json)

# Plot the training and validation accuracy and loss at each epoch
plot_train_test_loss(history, trained_path)
