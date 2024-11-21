"""
Randomly select train and test images for training the autoencoder.
"""
import os
import random
import shutil


# Define the root directory containing the subfolders
psd_dir = '/u/pa/nb/tourei/scratch/sits/ae_anomaly_detection/spectrum_plots/dec22/first_week/'
output_dir_train = '/u/pa/nb/tourei/scratch/sits/ae_anomaly_detection/train/dec22/first_week/plots/train/'
output_dir_test = '/u/pa/nb/tourei/scratch/sits/ae_anomaly_detection/train/dec22/first_week/plots/test/'

# Define number of train+test and the ratio
num_selected_img = 1000
test_to_train_ratio = 0.2

# Create output directories if it doesn't exist
os.makedirs(output_dir_train, exist_ok=True)
os.makedirs(output_dir_test, exist_ok=True)

# Collect all PNG files from the subfolders
all_png_files = []
for subdir, _, files in os.walk(psd_dir):
    for file in files:
        if file.endswith(".png"):
            all_png_files.append(os.path.join(subdir, file))

# Check if there are at least num_selected_img files
if len(all_png_files) < num_selected_img:
    raise ValueError(f"Only {len(all_png_files)} PNG files found, which is less than the required {num_selected_img}.")

# Randomly select num_selected_img PNG files
selected_files = random.sample(all_png_files, num_selected_img)
print(f"Selected {num_selected_img} PNG files. 
      Working on copying the images...")

# Calculate the number of test files
num_test = int(num_selected_img * test_to_train_ratio)
num_train = num_selected_img - num_test

# Split selected files into train and test sets
train_files = selected_files[:num_train]
test_files = selected_files[num_train:]

# Copy train files to train directory
for file_path in train_files:
    shutil.copy(file_path, output_dir_train)

# Copy test files to test directory
for file_path in test_files:
    shutil.copy(file_path, output_dir_test)

print(f"Copied {num_train} files to {output_dir_train}.")
print(f"Copied {num_test} files to {output_dir_test}.")     
