# das-anomaly
Python scripts for an autoencoder-based algorithm to detect anomalies in distributed acoustic sensing (DAS) datasets.

The main steps are as follows:
1. Plot power spectral density (PSD) plots in RGB format. We average the energy over a desired time window and stack all channels together to create a PSD with channels on the X-axis and frequency on the Y-axis. We create PSD of normal images (images without any anomaly or seismic event) and known seismic events. We can use MPI to distribute plotting PSDs over CPUs. 
2. Train the model on normal PSD images. 
3. Use the trained model to detect anomalies in PSDs.
4. Count number of detected anomalies (if needed).

Contact: Ahmad Tourei, Colorado School of Mines
tourei@mines.edu | ahmadtourei@gmail.com
