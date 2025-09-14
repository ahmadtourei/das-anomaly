from das_anomaly.psd import PSDConfig, PSDGenerator

cfg = PSDConfig()
# parallel processing with multiple processors using MPI:
PSDGenerator(cfg).run_parallel()
