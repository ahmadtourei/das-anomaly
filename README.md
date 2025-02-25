# das-anomaly
[![DOI](https://zenodo.org/badge/823391484.svg)](https://zenodo.org/doi/10.5281/zenodo.12747212)
[![Licence](https://www.gnu.org/graphics/lgplv3-88x31.png)](https://www.gnu.org/licenses/lgpl.html)

A Python package for anomaly detection in distributed acoustic sensing (DAS) datasets using an autoencoder-based deep learning algorithm.

## Installation
### Prerequisites
- Python >= 3.10
- `pip`
### Install Required Dependencies Only
It would be great to use a virtual environment or a fresh conda environment for better dependency management.
To install the package in editable mode with the required dependencies, run:

```bash
pip install -e .
```
### Install All Dependencies 

To install the package in editable mode with all optional dependencies, run:

```bash
pip install -e '.[all]'
```

## How to Use the Package?
The main steps are as follows:
1. Using the _plot_psd_ scripts, create power spectral density (PSD) plots in RGB format. We average the energy over a desired time window and stack all channels together to create a PSD with channels on the X-axis and frequency on the Y-axis. We create PSD of normal images (images without any anomaly or seismic event) and known seismic events. We can use MPI to distribute plotting PSDs over CPUs. 
2. Using the _train_model_ scripts, randomly select train and test PSD images and train the model on normal PSD images. 
3. Using the _validate_and_plot_density_ jupyter notebook, validate the trained model and find an appropriate density score as a threshold for anomaly detection.
4. Using the _detect_anomalies_ scripts, detect anomalies in PSD images with the trained model and write their information.
5. Using the _count_anomalies_ scripts, count the number of detected anomalies.

## Package's dependencies
- [DASCore](https://dascore.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Tesorflow](https://www.tensorflow.org/install)

Optional:
- [MPI4Py](https://mpi4py.readthedocs.io/en/stable/install.html)

## Contact
Ahmad Tourei, Colorado School of Mines
tourei@mines.edu | ahmadtourei@gmail.com
