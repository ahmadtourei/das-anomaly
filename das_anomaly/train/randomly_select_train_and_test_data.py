"""
Randomly select train and test images for training the autoencoder.
"""
import os
import random
import shutil

from das_anomaly.settings import SETTINGS


# Define PSD plots path
psd_dir = SETTINGS.PSD_DIR
# Define the target directory for training and testing (PSDs to be copied to)
train_images_path = SETTINGS.TRAIN_IMAGES_PATH
test_images_path = SETTINGS.TEST_IMAGES_PATH

# Define number of train+test and the ratio
num_img = SETTINGS.NUM_IMAGE
ratio = SETTINGS.RATIO

# Create output directories if it doesn't exist
os.makedirs(train_images_path, exist_ok=True)
os.makedirs(test_images_path, exist_ok=True)

# Collect all PNG files from the subfolders
all_png_files = []
for subdir, _, files in os.walk(psd_dir):
    for file in files:
        if file.endswith(".png"):
            all_png_files.append(os.path.join(subdir, file))

# Check if there are at least num_selected_img files
if len(all_png_files) < num_img:
    raise ValueError(f"Only {len(all_png_files)} PNG files found, which is less than the required {num_selected_img}.")

# Randomly select num_selected_img PNG files
selected_files = random.sample(all_png_files, num_img)
print(f"Selected {num_img} PNG files. Working on copying the images...")

# Calculate the number of test files
num_test = int(num_img * ratio)
num_train = num_img - num_test

# Split selected files into train and test sets
train_files = selected_files[:num_train]
test_files = selected_files[num_train:]

# Copy train files to train directory
for file_path in train_files:
    shutil.copy(file_path, train_images_path)

# Copy test files to test directory
for file_path in test_files:
    shutil.copy(file_path, test_images_path)

print(f"Copied {num_train} files to {train_images_path}.")
print(f"Copied {num_test} files to {test_images_path}.")
