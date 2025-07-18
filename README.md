# das-anomaly
[![DOI](https://zenodo.org/badge/823391484.svg)](https://doi.org/10.5281/zenodo.12747212)
[![Licence](https://www.gnu.org/graphics/lgplv3-88x31.png)](https://www.gnu.org/licenses/lgpl.html)
[![codecov](https://codecov.io/gh/ahmadtourei/das-anomaly/branch/main/graph/badge.svg)](https://codecov.io/gh/ahmadtourei/das-anomaly)

_das-anomaly_ is an open-source Python package for unsupervised anomaly detection in distributed acoustic sensing (DAS) datasets using an autoencoder-based deep learning algorithm. It is being developed by Ahmad Tourei under the supervision of Dr. Eileen R. Martin at Colorado School of Mines. 

If you use _das-anomaly_ in your work, please cite the following:

> Ahmad Tourei. (2025). ahmadtourei/das-anomaly: latest (Concept). Zenodo. http://doi.org/10.5281/zenodo.12747212


## Installation
### Prerequisites
- Python = 3.10, 3.11, 3.12
- pip

### Dependencies
- [DASCore](https://dascore.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [TensorFlow](https://www.tensorflow.org/install)

Optional:
- [MPI4Py](https://mpi4py.readthedocs.io/en/stable/install.html)

Dependency notes:
1. Installation and loading of [Open MPI](https://www.open-mpi.org/) is required prior to `MPI4Py` installation. Ensure proper installation using a [helloworld example](https://mpi4py.readthedocs.io/en/3.1.4/install.html#testing).

2. If you'd like to train the model on GPU, make sure you install TensorFlow with GPU setup in your environment. More information can be found [here](https://www.tensorflow.org/install/pip#:~:text=4.-,Install%20TensorFlow,-TensorFlow%20requires%20a).

3. Currently waiting on `TesorFlow` to support Python 3.13 before we can support it as well.

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
1. Define constants and create a Spool of data: 
Using the _config_user_ script in the das_anomaly directory, define the constants and directory paths for data, power spectral density (PSD) images, detected anomaly results, etc. You would complete adding the values as you go over the steps mentioned below.

Then, using DASCore, create an index file for the [spool](https://dascore.org/tutorial/spool.html) of data first time reading the DAS data directory:
### Example
```python
import dascore as dc
from das_anomaly.settings import SETTINGS

data_path = SETTINGS.DATA_PATH

# Update will create an index of the contents for fast querying/access
spool = dc.spool(directory_path).update()
``` 
Note: Creating the spool for the first time may take some time if your directory contains hundreds of gigabytes or terabytes of DAS data. However, DASCore creates an index file, allowing it to quickly query the directory on subsequent accesses.

2. Set a consistent upper bound for PSD amplitude values:
To ensure all PSD images share the same colorbar scale, determine an appropriate CLIP_VALUE_MAX in the _config_user_ script. This can be done using the `get_psd_max_clip` function, which computes the mean value of maximum amplitude from TIME_WINDOWs of the data which does not include drastic anomalies (therefore, a quick exploratory data analysis is needed here.)
### Example
```python
from das_anomaly.psd import PSDConfig, PSDGenerator
from das_anomaly.settings import SETTINGS

# path to one or a few background noise data 
bn_data_path = SETTINGS.BN_DATA_PATH
cfg = PSDConfig(data_path=bn_data_path)
gen = PSDGenerator(cfg)
percentile = 90
clip_val = gen.run_get_psd_val(percentile=percentile)
print(f"Mean {percentile}-percentile amplitude across all patches: {clip_val:.3e}")
```
3. Generate PSD plots: 
Use the `das_anomaly.psd` module and create PSD plots in RGB format and in plain mode (with no axes or colorbar). The `das_anomaly.psd.PSDGenerator reads DAS data, creates a spool using DASCore library, applies a detrend function to each patch of the chunked spool, and then average the energy over a desired time window and stack all channels together to create a spatial PSD with channels on the X-axis and frequency on the Y-axis. You can use MPI to distribute reading data and plotting PSDs over CPUs. 
### Example
```python
from das_anomaly.psd import PSDConfig, PSDGenerator

cfg = PSDConfig()
# serial processing with single processor:
PSDGenerator(cfg).run()
# parallel processing with multiple processors using MPI:
PSDGenerator(cfg).run_parallel()
```
Note: If you'd like to use PSDs for purposes other than training the model, the `hide_axes=False` will plot the PSD with axes and colorbar (default is True).
### Example
```python
from das_anomaly.psd import PSDConfig, PSDGenerator

cfg = PSDConfig(hide_axes=False)
# serial processing with single processor:
PSDGenerator(cfg).run()
# parallel processing with multiple processors using MPI:
PSDGenerator(cfg).run_parallel()
```
4. Select and copy known anomaly PSD plots:
From the generated PSD plots, identify and copy examples of known anomalies to the ANOMALY_IMAGES_PATH specified in the _config_user_ script. These anomalies can include events such as earthquakes from an existing catalog, seismic activity, instrument noise, anthropogenic disturbances, etc. Including these examples helps improve the accuracy of thresholding during the anomaly detection process.

5. Train: 
The `das_anomaly.train` module helps with randomly selecting train and test PSD images and training the model (with CPU or GPU) on anomaly-free PSD images. If you need to change model's architecture, you'll need to modify the `encoder` and `decoder` functions in the [utils.py](das_anomaly/utils.py). 
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
Note: Since the `TrainSplitConfig()` function randomly selects PSD images from the generated plots, you must ensure the training and testing datasets do not include anomalies. If you have an excel sheet with time stamp of anomalies, use "exclude_known_events_from_training" in examples directory to exclude them. Or, manually inspect both the training and testing sets to ensure they do not contain apparent anomalies. Review their time- and frequency-domain representations, and remove any suspicious samples to maintain the quality of training.

6. Test and set a threshold: 
Using the _validate_and_plot_density_ jupyter notebook in the examples directory, validate the trained model and find an appropriate density score as a threshold for anomaly detection. Then, make sure to modify the DENSITY_THRESHOLD parameter in the _config_user_ script. 

7. Run the trained model: 
The `das_anomaly.detect` module applies the trained model to the data to detect anomalies in the PSD images and writes their information. It also copies the detected anomaly to the RESULTS_PATH. MPI can be used to distribute PSDs over CPUs. Then, using the `das_anomaly.count` module, count the number of detected anomalies and display their details and file paths.
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
AnomalyCounter(cfg).run() # prints info on number of anomalies and path to them
```

## Note
Still under development. Use with caution.

## Contact
Ahmad Tourei, Colorado School of Mines
tourei@mines.edu | ahmadtourei@gmail.com
