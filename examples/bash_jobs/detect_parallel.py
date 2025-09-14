from das_anomaly.detect import AnomalyDetector, DetectConfig

cfg = DetectConfig()
AnomalyDetector(cfg).run_parallel()
