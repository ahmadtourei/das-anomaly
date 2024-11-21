"""
This script uses unsupervised autoencoders to train a deep learning model for anomaly detection in images.

For microseismic event detection, images can be the power spectral density (PSD) plots. 

We will consider the bottleneck layer output from our autoencoder as the latent space. 
Using the reconstruction error and kernel density estimation (KDE) based on the vectors in the latent space, anomalies can be detected.  
"""
import json
import os
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Add the parent directory to the system path to enable importing modules from it.
current_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(current_dir, '..')
sys.path.append(source_dir)
from utils import plot_train_test_loss


# Specify path to save results (model and plots)
results_path = '/u/pa/nb/tourei/scratch/sits/ae_anomaly_detection/train/dec22/first_week/'

# Size of the input images and number of epoches for training
size = 512
num_epoch = 500

# Define generators for training, validation, and anomaly data.
batch_size = 64
datagen = ImageDataGenerator(rescale=1./255)

# Path to training PSD plots (seen data)
train_path = '/u/pa/nb/tourei/scratch/sits/ae_anomaly_detection/train/dec22/first_week/plots/train/'
num_train_data = 768
train_generator = datagen.flow_from_directory(
    train_path,
    target_size=(size, size),
    batch_size=batch_size,
    class_mode='input'
    )

# path to testing PSD plots (unseen data)
test_path = '/u/pa/nb/tourei/scratch/sits/ae_anomaly_detection/train/dec22/first_week/plots/test/'
num_test_data = 192
validation_generator = datagen.flow_from_directory(
    test_path,
    target_size=(size, size),
    batch_size=batch_size,
    class_mode='input'
    )

# path to known seismic events
events_path = '/u/pa/nb/tourei/scratch/sits/ae_anomaly_detection/train/dec22/first_week/plots/anomaly/'
anomaly_generator = datagen.flow_from_directory(
    events_path,
    target_size=(size, size),
    batch_size=batch_size,
    class_mode='input'
    )

# Define the autoencoder. 
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
model.summary()

# Fit the model. 
history = model.fit(
        train_generator,
        steps_per_epoch= num_train_data // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_test_data // batch_size,
        shuffle = True)

# Save the model in h5 format
model.save(results_path + f'model_1_{size}.h5')  

# Save the history as well
history_dict = history.history
history_json = json.dumps(history_dict)
with open(results_path + f'history_1_{size}.json', 'w') as json_file:
    json_file.write(history_json)
    
# Plot the training and validation accuracy and loss at each epoch
plot_train_test_loss(history, results_path)
