# das-anomaly
[![DOI](https://zenodo.org/badge/823391484.svg)](https://zenodo.org/doi/10.5281/zenodo.12747212)
[![Licence](https://www.gnu.org/graphics/lgplv3-88x31.png)](https://www.gnu.org/licenses/lgpl.html)
<!--
[![codecov](https://codecov.io/gh/ahmadtourei/das-anomaly/branch/main/graph/badge.svg)](https://codecov.io/gh/ahmadtourei/das-anomaly)
-->

_das-anomaly_ is an open-source Python package for anomaly detection in distributed acoustic sensing (DAS) datasets using an autoencoder-based deep learning algorithm. It is being developed by Ahmad Tourei under the supervision of Dr. Eileen R. Martin at Colorado School of Mines. 

If you use _das-anomaly_ in your work, please cite the following:

> Ahmad Tourei. (2025). ahmadtourei/das-anomaly: latest (Concept). Zenodo. http://doi.org/10.5281/zenodo.14927429


## Installation
### Prerequisites
- Python >= 3.10
- `pip`
### Install Required Dependencies Only
For clean dependency management, use a virtual environment or a fresh Conda environment.
To install the package in editable mode with the required dependencies, run the following after cloning the repository and navigating to the repo directory:

```bash
pip install -e .
```
### Install All Dependencies 

To install the package in editable mode with all optional dependencies, run:

```bash
pip install -e '.[all]'
```

### Uninstall 
To uninstall the package, run:

```bash
pip uninstall das_anomaly
```

## Instructions
The main steps for using the package are as follows:
1. Define constants: Using the _user_defaults_ script in the das_anomaly directory, define the constants and directory paths (for data, PSD images, detected anomaly results, etc.)
2. Generate PSD plots: Using the _plot_psd_ scripts, create power spectral density (PSD) plots in RGB format. We average the energy over a desired time window and stack all channels together to create a PSD with channels on the X-axis and frequency on the Y-axis. We create PSD of anomaly-free images (usually background noise) and known seismic events. We can use Open MPI to distribute plotting PSDs over CPUs. 
3. Train: Using the _train_model_ scripts, randomly select train and test PSD images and train the model on anomaly-free PSD images. 
4. Test and set a threshold: Using the _validate_and_plot_density_ jupyter notebook in the examples directory, validate the trained model and find an appropriate density score as a threshold for anomaly detection.
5. Run the trained model: Using the _detect_anomalies_ scripts, detect anomalies in PSD images via the trained model and write their information (single processor or use Open MPI). Then using the _count_anomalies_ scripts, count the number of detected anomalies.

## Package's Dependencies matplotlib
- [matplotlib](https://matplotlib.org/)
- [DASCore](https://dascore.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [TesorFlow](https://www.tensorflow.org/install)

Optional:
- [MPI4Py](https://mpi4py.readthedocs.io/en/stable/install.html)

Installation and loding of [Open MPI](https://www.open-mpi.org/) module is required prior to `MPI4Py` installation. Ensure proper installation using a [helloworld example](https://mpi4py.readthedocs.io/en/3.1.4/install.html#testing).

## Note
Still under development. Use with caution.

## Contact
Ahmad Tourei, Colorado School of Mines
tourei@mines.edu | ahmadtourei@gmail.com
