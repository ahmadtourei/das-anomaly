"""
Utility functions for anomaly detection in DAS datasets using autoencoders.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.fftpack as ft

from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D


def calculate_percentile(data, percentile):
    """Calculate the given percentile of a list of data."""
    size = len(data)
    if size == 0:
        return None  # Return None if data is empty

    # Sort the data
    data.sort()

    # Calculate the percentile position
    pos = (size + 1) * percentile / 100.0 - 1
    if pos.is_integer():
        # If pos is an integer, return the data at pos
        return data[int(pos)]
    else:
        # If pos is not an integer, interpolate between adjacent data points
        lower_index = int(pos)
        upper_index = min(lower_index + 1, size - 1)  # Ensure upper_index is within bounds
        interpolation = pos - lower_index
        return data[lower_index] * (1 - interpolation) + data[upper_index] * interpolation


def check_if_anomaly(encoder_model, size, img_path, density_threshold, kde):
    """Check whether the image is an anomaly"""
    # Flatten the encoder output because KDE from sklearn takes 1D vectors as input
    encoder_output_shape = encoder_model.output_shape
    out_vector_shape = encoder_output_shape[1] * encoder_output_shape[2] * encoder_output_shape[3]

    img = Image.open(img_path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img = np.array(img.resize((size, size), Image.Resampling.LANCZOS))
    img = img / 255.0
    img = img[np.newaxis, :, :, :]
    encoded_img = encoder_model.predict([[img]], verbose=0)
    encoded_img = [np.reshape(img, (out_vector_shape)) for img in encoded_img]
    density = kde.score_samples(encoded_img)[0]

    if density < density_threshold:
        out = "The image is an anomaly"
    else:
        out = "The image is normal"
    return out


def decoder(model):
    """Imports decoder model."""
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))

    return model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))


def density(encoder_model, batch_images, kde):
    """Caulculate the density score."""
    # Flatten the encoder output because KDE from sklearn takes 1D vectors as input
    encoder_output_shape = encoder_model.output_shape
    out_vector_shape = encoder_output_shape[1] * encoder_output_shape[2] * encoder_output_shape[3]

    density_list = []
    for im in range(0, batch_images.shape[0]):
        img = batch_images[im]
        img = img[np.newaxis, :, :, :]
        encoded_img = encoder_model.predict([[img]], verbose=0)  # Create a compressed version of the image using the encoder
        encoded_img = [np.reshape(img, (out_vector_shape)) for img in encoded_img]  # Flatten the compressed image
        density = kde.score_samples(encoded_img)[0]  # get a density score for the new image
        density_list.append(density)

    return np.array(density_list)


def encoder(size):
    """Imports the encoder model."""
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(size, size, 3)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    return model


def plot_spec(
    patch_strain,
    min_freq,
    max_freq,
    sampling_rate,
    title,
    output_rank,
    fig_path,
    dpi,
):
    """Save the spectrum of all channels stacked (Channel-Frequency-Amplitude plot)."""
    # Get the data
    strain_rate = patch_strain.transpose("time", "distance").data
    # Get coords info
    dist_coord = patch_strain.coords.get_coord("distance")
    dist_min = dist_coord.min()
    dist_max = dist_coord.max()
    # Check for valid inputs (note - these checks aren't exhaustive)
    if max_freq <= min_freq:
        print("Error in plot_spec inputs: minFrq " + str(min_freq) + " >= maxFrq " + str(max_freq))
        return
    # Calculate the amplitude spectrum (not amplitude symmetry for +/- frequencies)
    spect = ft.fft(strain_rate, axis=0) 
    nFrqBins = int(spect.shape[0] / 2)  # number of frequency bins
    amplitudeSpec = np.absolute(spect[:nFrqBins, :])
    # Calculate indices corresponding to the frequencies of interest
    nyquist_frq = sampling_rate / 2.0  # the Nyquist frequency
    # Make sure maxFrq doesn't exceed Nyquist frequency
    if max_freq > nyquist_frq:
        print(
            "Error in plot_spec inputs: maxFrq "
            + str(max_freq)
            + " >= Nyquist frequency "
            + str(nyquist_frq)
            + " indicated by sampleRate "
            + str(sampling_rate)
        )
    # convert frequencies to an index in the array
    HzPerBin = nyquist_frq / float(nFrqBins)
    minFrqIdx = int(min_freq / HzPerBin)
    maxFrqIdx = int(max_freq / HzPerBin)
    # Plot
    _, ax = plt.subplots(figsize=(12, 12))
    #clipValMax = np.percentile(amplitudeSpec[minFrqIdx:maxFrqIdx, :], 95)
    clipValMax = 7e-6 # this value should be calculated based on the above line for a typical background noise (anomaly-free PSD)
    clipValMin = 0
    # Define the colors in RGB
    colors = [
        (1, 0, 0),  # Red
        (0, 1, 0),  # Green
        (0, 0, 1),  # Blue
    ]  
    # Create the colormap
    n_bins = 100  # Increase for smoother transitions
    cmap_name = "rgb_custom_cmap"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    _ = ax.imshow(
        amplitudeSpec[minFrqIdx:maxFrqIdx, :],
        aspect="auto",
        interpolation="none",
        cmap=cm,
        extent=(dist_min, dist_max, max_freq, min_freq),
        vmin=clipValMin,
        vmax=clipValMax,
    )
    # Hide the axes
    ax.axis("off")
    # Hide the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    fig_path_ranks = os.path.join(fig_path, "rank_" + output_rank)
    # Check if the directory does not exist
    if not os.path.exists(fig_path_ranks):
        # Create the directory
        os.makedirs(fig_path_ranks)
    plt.savefig(os.path.join(fig_path_ranks, f"{title}.png"), dpi=dpi)
    plt.close("all")


def plot_train_test_loss(history, path):
    """Plot the training and validation accuracy at each epoch"""
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "y", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    title = "Training_and_validation_accuracy_and_loss"
    plt.savefig(os.path.join(path, title + ".png"))


def search_keyword_in_files(directory, keyword):
    """Function to search for a keyword in all text results within a directory"""
    keyword_count = 0
    lines_with_keyword = []

    # Walk through all files in the specified directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file is a text file
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, encoding="utf-8") as f:
                    for line in f:
                        if keyword in line:
                            keyword_count += line.count(keyword)
                            # Strip removes leading/trailing whitespace
                            lines_with_keyword.append(line.strip())

    return keyword_count, lines_with_keyword
