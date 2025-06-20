# das-anomaly
[![DOI](https://zenodo.org/badge/823391484.svg)](https://zenodo.org/doi/10.5281/zenodo.12747212)
[![Licence](https://www.gnu.org/graphics/lgplv3-88x31.png)](https://www.gnu.org/licenses/lgpl.html)
[![codecov](https://codecov.io/gh/ahmadtourei/das-anomaly/branch/main/graph/badge.svg)](https://codecov.io/gh/ahmadtourei/das-anomaly)

_das-anomaly_ is an open-source Python package for unsupervised anomaly detection in distributed acoustic sensing (DAS) datasets using an autoencoder-based deep learning algorithm. It is being developed by Ahmad Tourei under the supervision of Dr. Eileen R. Martin at Colorado School of Mines. 

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
1. Define constants: 
Using the _user_defaults_ script in the das_anomaly directory, define the constants and directory paths (for data, PSD images, detected anomaly results, etc.)
2. Get a fixed value for the upper bound of PSD amplitudes:
To ensure that all the PSD images have the same colorbar range, we need to get an appropriate value for CLIP_VALUE_MAX in the _user_defaults_ script. To do so, we need to use the `get_psd_max_clip` function to calculate this value from a portion (TIME_WINDOW) of the data with no anomalies (i.e., background noise data).
### Example
```python
from das_anomaly.psd import PSDConfig, PSDGenerator
from das_anomaly.settings import SETTINGS
from das_anomaly.utils import get_psd_max_clip

# path to one or a few background noise data 
bn_data_path = SETTINGS.BN_DATA_PATH
cfg = PSDConfig(data_path=bn_data_path)
clip_val = gen.run_get_psd_val()
print(f"Mean 95-percentile amplitude across all patches: {clip_val:.3e}")
```
3. Generate PSD plots: 
Use the `das_anomaly.psd` module and create power spectral density (PSD) plots in RGB format. First, create a spool of DAS data and transform it to strain rate and apply a detrend function. Then, average the energy over a desired time window and stack all channels together to create a spatial PSD with channels on the X-axis and frequency on the Y-axis. Finally, create PSDs of anomaly-free images (usually background noise) and known anomalies. You can use MPI to distribute plotting PSDs over CPUs. 
### Example
```python
from das_anomaly.psd import PSDConfig, PSDGenerator
from das_anomaly.settings import SETTINGS

data_path = SETTINGS.DATA_PATH
cfg = PSDConfig(data_path=data_path)
# serial processing with single processor:
PSDGenerator(cfg).run()
# parallel processing with multiple processors using MPI:
PSDGenerator(cfg).run_parallel()
```
4. Train: 
The `das_anomaly.train` module helps with randomly selecting train and test PSD images and training the model on anomaly-free PSD images. 
### Example
```python
from das_anomaly.settings import SETTINGS
from das_anomaly.train import TrainAEConfig, AutoencoderTrainer, TrainSplitConfig, ImageSplitter

# select and copy train and test datasets from PSD
cfg = TrainSplitConfig()
ImageSplitter(cfg).run()

# train the autoencoder model
cfg = TrainAEConfig()
AutoencoderTrainer(cfg).run()
```
5. Test and set a threshold: 
Using the _validate_and_plot_density_ jupyter notebook in the examples directory, validate the trained model and find an appropriate density score as a threshold for anomaly detection. Make sure to modify the DENSITY_THRESHOLD parameter in the _user_defaults_ script. 

6. Run the trained model: 
The `das_anomaly.detect` module applies the trained model to the data, detects anomalies in the PSD images, and writes their information. MPI can be used to distribute PSDs over CPUs. Then, using the `das_anomaly.count` module, count the number of detected anomalies.
### Example
```python
from das_anomaly.count.counter import CounterConfig, AnomalyCounter
from das_anomaly.detect import DetectConfig, AnomalyDetector

cfg = DetectConfig()
# serial processing with single processor:
AnomalyDetector(cfg).run()
# parallel processing with multiple processors using MPI:
AnomalyDetector(cfg).run_parallel()

# count number of anomalies
cfg = CounterConfig(keyword="anomaly")
total = AnomalyCounter(cfg).run()
num = len(total)
print(f'Total number of detected anomalies: {num}')
```

## Package's Dependencies
- [DASCore](https://dascore.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [TesorFlow](https://www.tensorflow.org/install)

Optional:
- [MPI4Py](https://mpi4py.readthedocs.io/en/stable/install.html)

Installation and loading of [Open MPI](https://www.open-mpi.org/) is required prior to `MPI4Py` installation. Ensure proper installation using a [helloworld example](https://mpi4py.readthedocs.io/en/3.1.4/install.html#testing).

## Note
Still under development. Use with caution.

## Contact
Ahmad Tourei, Colorado School of Mines
tourei@mines.edu | ahmadtourei@gmail.com
