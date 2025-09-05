"""
Utility functions for anomaly detection in DAS datasets using autoencoders.
"""

from __future__ import annotations

import os
import string

import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack as ft
import tensorflow as tf
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

from das_anomaly.settings import SETTINGS


@staticmethod
def _compile(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"],  # mean absolute error
    )
    return model


def _unique_name(model, base):
    used = {l.name for l in model.layers}
    if base not in used:
        return base
    i = 1
    while f"{base}_{i}" in used:
        i += 1
    return f"{base}_{i}"


def check_if_anomaly(
    encoder_model, size, img_path, kde, density_threshold=None, mse_threshold=None
):
    """Check whether the image is an anomaly"""
    # Flatten the encoder output because KDE from sklearn takes 1D vectors as input
    encoder_output_shape = encoder_model.output_shape
    out_vector_shape = (
        encoder_output_shape[1] * encoder_output_shape[2] * encoder_output_shape[3]
    )

    img = Image.open(img_path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img = np.array(img.resize((size, size), Image.Resampling.LANCZOS))
    img = img / 255.0
    img = img[np.newaxis, :, :, :]
    encoded_img = encoder_model.predict([[img]], verbose=0)
    encoded_img = [np.reshape(img, (out_vector_shape)) for img in encoded_img]
    log_density = kde.score_samples(encoded_img)[0]

    if mse_threshold is not None:
        cloned_enc = tf.keras.models.clone_model(encoder_model)
        cloned_enc.build(encoder_model.input_shape)
        cloned_enc.set_weights(encoder_model.get_weights())
        ae = decoder(cloned_enc)
        ae_model_compiled = _compile(ae)

        img_mse = np.asarray(img, dtype=np.float64)
        if img_mse.ndim == 3:  # (H,W,C) -> (1,H,W,C)
            img_mse = img_mse[None, ...]
        assert img_mse.ndim == 4, f"Expected 4D (B,H,W,C), got {img_mse.shape}"

        mse = ae_model_compiled.evaluate(img_mse, img_mse, batch_size=1)[0]
    # --- criteria (ignore any None thresholds) ---
    density_is_anom = (density_threshold is None) or (log_density < density_threshold)
    mse_is_anom = (mse_threshold is None) or (mse > mse_threshold)

    is_anomaly = density_is_anom and mse_is_anom
    return "The image is an anomaly" if is_anomaly else "The image is normal"


def decoder(
    model: Sequential,
    *,
    output_channels: int = 3,
    conv_activation: str = "relu",
    final_activation: str = "sigmoid",
) -> Sequential:
    """
    Extend `encoder` with a decoder that mirrors its Conv2D + MaxPooling blocks.

    Parameters
    ----------
    model : keras.Sequential (the encoder)
        A stack of Conv2D → MaxPooling2D blocks (built with `encoder()`).
        The model at the end of encoding stage.
    output_channels : int, default=3
        Number of channels in the reconstructed image (e.g. 3 for RGB).
    conv_activation : str, default="relu"
        Activation used in decoder Conv2D layers.
    final_activation : str, default="sigmoid"
        Activation of the final reconstruction layer.

    Returns
    -------
    autoencoder : keras.Sequential
        The same object you passed in, now containing the full encoder-decoder.
    """
    # Collect the filter sizes from the encoder’s Conv2D layers
    conv_filters = [
        layer.filters for layer in model.layers if isinstance(layer, Conv2D)
    ]

    if not conv_filters:
        raise ValueError("Encoder must contain Conv2D layers.")

    # Mirror them (e.g. [64, 32, 16]  →  [16, 32, 64])
    for filt in conv_filters[::-1]:
        model.add(
            Conv2D(
                filt,
                (3, 3),
                activation=conv_activation,
                padding="same",
                name=_unique_name(model, "dec_conv"),
            )
        )
        model.add(UpSampling2D((2, 2), name=_unique_name(model, "dec_up")))

    # Final pixel-space reconstruction layer
    model.add(
        Conv2D(
            output_channels,
            (3, 3),
            activation=final_activation,
            padding="same",
            name=_unique_name(model, "dec_out"),
        )
    )
    return model


def density(encoder_model, batch_images, kde):
    """Caulculate the density score."""
    # Flatten the encoder output because KDE from sklearn takes 1D vectors as input
    encoder_output_shape = encoder_model.output_shape
    out_vector_shape = (
        encoder_output_shape[1] * encoder_output_shape[2] * encoder_output_shape[3]
    )

    density_list = []
    for im in range(0, batch_images.shape[0]):
        img = batch_images[im]
        img = img[np.newaxis, :, :, :]
        encoded_img = encoder_model.predict(
            [[img]], verbose=0
        )  # Create a compressed version of the image using the encoder
        encoded_img = [
            np.reshape(img, (out_vector_shape)) for img in encoded_img
        ]  # Flatten the compressed image
        density = kde.score_samples(encoded_img)[
            0
        ]  # get a density score for the new image
        density_list.append(density)

    return np.array(density_list)


def encoder(
    size: int,
    num_layers: int = 5,
    start_filters: int = 64,
    filters_halve_every_layer: bool = True,
    activation: str = "relu",
) -> Sequential:
    """
    Build a simple CNN encoder.

    Parameters
    ----------
    size : int
        Input (height = width) in pixels. Input shape = (size, size, 3).
    num_layers : int, default=3
        Number of Conv2D + MaxPooling2D blocks.
    start_filters : int, default=64
        Filters in the first Conv2D layer.
    filters_halve_every_layer : bool, default=True
        If True, filters //= 2 each block (e.g., 64→32→16…).
        If False, every block uses `start_filters`.
    activation : str, default="relu"
        Activation function for Conv2D layers.

    Returns
    -------
    model : keras.Sequential
        The assembled encoder.
    """
    if start_filters / num_layers < 4:
        raise ValueError(
            "start_filters/num_layers must be greater than 4 (the number of filters for the last layer). Consider increasing number of filters or decreasing number of layers."
        )

    model = Sequential()

    filters = start_filters
    # First block (needs input_shape)
    model.add(
        Conv2D(
            filters,
            (3, 3),
            activation=activation,
            padding="same",
            input_shape=(size, size, 3),
        )
    )
    model.add(MaxPooling2D((2, 2), padding="same"))

    # Remaining blocks
    for _ in range(1, num_layers):
        if filters_halve_every_layer and filters > 4:
            filters //= 2
        model.add(Conv2D(filters, (3, 3), activation=activation, padding="same"))
        model.add(MaxPooling2D((2, 2), padding="same"))

    return model


def get_psd_max_clip(patch_strain, min_freq, max_freq, sampling_rate, percentile):
    """
    Return the given percentile of the data in frequency domain.
    This value serve as a the upper bound of the colorbar for PSD
    plotting, which will be fixed for all PSDs.
    """
    strain_rate = patch_strain.transpose("time", "distance").data  # pragma: no cover
    # Calculate the amplitude spectrum (not amplitude symmetry for +/- frequencies)
    spect = ft.fft(strain_rate, axis=0)
    n_frq_bins = int(spect.shape[0] / 2)  # number of frequency bins
    amplitude_spec = np.absolute(spect[:n_frq_bins, :])
    # Calculate indices corresponding to the frequencies of interest
    nyquist_frq = sampling_rate / 2.0  # the Nyquist frequency
    # convert frequencies to an index in the array
    hz_per_bin = nyquist_frq / float(n_frq_bins)
    min_frq_idx = int(min_freq / hz_per_bin)
    max_frq_idx = int(max_freq / hz_per_bin)

    clip_val_max = np.percentile(amplitude_spec[min_frq_idx:max_frq_idx, :], percentile)
    return clip_val_max


def plot_spec(
    patch_strain,
    min_freq,
    max_freq,
    sampling_rate,
    title,
    output_rank,
    fig_path,
    dpi,
    hide_axes=True,
    save_fig=True,
):
    """Save the power spectral density (Channel-Frequency-Amplitude) plot."""
    # Get the data
    strain_rate = patch_strain.transpose("time", "distance").data  # pragma: no cover
    # Get coords info
    dist_coord = patch_strain.coords.get_coord("distance")
    dist_min = dist_coord.min()
    dist_max = dist_coord.max()
    # Check for valid inputs (note - these checks aren't exhaustive)
    if max_freq <= min_freq:
        raise ValueError(
            f"`min_freq` {min_freq} must be less than " f"`max_freq` {max_freq}."
        )
    # Calculate the amplitude spectrum (not amplitude symmetry for +/- frequencies)
    spect = ft.rfft(strain_rate, axis=0)
    n_frq_bins = spect.shape[0]  # number of frequency bins
    amplitude_spec = np.absolute(spect)
    # Calculate indices corresponding to the frequencies of interest
    nyquist_frq = sampling_rate / 2.0  # the Nyquist frequency
    # Make sure maxFrq doesn't exceed Nyquist frequency
    if max_freq > nyquist_frq:
        raise ValueError(
            f"`max_freq` {max_freq} must be less than "
            f"`nyquist frequency` {nyquist_frq}."
        )
    # convert frequencies to an index in the array
    hz_per_bin = nyquist_frq / float(n_frq_bins)
    min_frq_idx = int(min_freq / hz_per_bin)
    max_frq_idx = int(max_freq / hz_per_bin)
    # Plot
    clip_val_max = SETTINGS.CLIP_VALUE_MAX  # pragma: no cover
    clip_val_min = 0
    colors = [
        (1, 0, 0),  # Red
        (0, 1, 0),  # Green
        (0, 0, 1),  # Blue
    ]
    # Create the colormap
    n_bins = 100  # Increase for smoother transitions
    cmap_name = "rgb_custom_cmap"
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    if hide_axes:
        _, ax = plt.subplots(figsize=(12, 12))
        # Define the colors in RGB
        _ = ax.imshow(
            amplitude_spec[min_frq_idx:max_frq_idx, :],
            aspect="auto",
            interpolation="none",
            cmap=cm,
            extent=(dist_min, dist_max, max_freq, min_freq),
            vmin=clip_val_min,
            vmax=clip_val_max,
        )
        # make y axis (freq) increasing
        ax.invert_yaxis()
        # Hide the axes
        ax.axis("off")  # pragma: no cover
        # Hide the ticks
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        fig = plt.figure(figsize=(14, 12))
        spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[12, 1], figure=fig)

        # Create main axis for imshow with exact 12x12 aspect
        ax = fig.add_subplot(spec[0])
        img = ax.imshow(
            amplitude_spec[min_frq_idx:max_frq_idx, :],
            aspect="auto",
            interpolation="none",
            cmap=cm,
            extent=(dist_min, dist_max, max_freq, min_freq),
            vmin=clip_val_min,
            vmax=clip_val_max,
        )
        # make y axis (freq) increasing
        ax.invert_yaxis()
        # Format main plot
        ax.set_xlabel("Distance (m)", fontsize=20)
        ax.set_ylabel("Frequency (Hz)", fontsize=20)

        # Increase tick label size and thickness
        ax.tick_params(axis="both", which="major", labelsize=18, width=2.5, length=8)
        ax.tick_params(axis="both", which="minor", labelsize=16, width=2, length=5)

        # Increase thickness of axis lines
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)

        # Create a separate axis for colorbar to keep imshow part exactly 12x12
        cbar_ax = fig.add_subplot(spec[1])
        cbar = plt.colorbar(img, cax=cbar_ax)
        cbar.set_label("Amplitude", fontsize=18)
        cbar.ax.tick_params(labelsize=16)
        # Ensure colorbar offset text (if using scientific notation) is also large
        cbar.formatter.set_powerlimits((-2, 2))  # Force scientific notation if needed
        cbar.ax.yaxis.get_offset_text().set_fontsize(
            16
        )  # Correct way to set offset text size
        # Add large axis titles
        ax.set_xlabel("Distance (m)", fontsize=32)
        ax.set_ylabel("Frequency (Hz)", fontsize=32)

        # Increase tick label size and thickness
        ax.tick_params(axis="both", which="major", labelsize=22, width=2.5, length=8)
        ax.tick_params(axis="both", which="minor", labelsize=22, width=2, length=5)

        # Increase thickness of axis lines
        ax.spines["top"].set_linewidth(2.5)
        ax.spines["bottom"].set_linewidth(2.5)
        ax.spines["left"].set_linewidth(2.5)
        ax.spines["right"].set_linewidth(2.5)

    if save_fig:
        # save figure
        fig_path_ranks = os.path.join(fig_path, "rank_" + str(output_rank))
        # Check if the directory does not exist
        if not os.path.exists(fig_path_ranks):
            # Create the directory
            os.makedirs(fig_path_ranks)
        plt.savefig(os.path.join(fig_path_ranks, f"{title}.png"), dpi=dpi)
    else:
        plt.show()
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
    matching_lines: list[str] = []
    trailing_punct = str.maketrans("", "", string.punctuation)

    # iterate over files in the first level only
    for file_path in directory.iterdir():
        if file_path.suffix != ".txt" or not file_path.is_file():
            continue

        with file_path.open(encoding="utf-8") as fh:
            for raw in fh:
                line = raw.rstrip()
                if not line:
                    continue

                last_token = line.split()[-1].translate(trailing_punct)
                if last_token == keyword:
                    keyword_count += 1
                    matching_lines.append(line)

    return keyword_count, matching_lines
