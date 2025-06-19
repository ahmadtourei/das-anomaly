import pytest

from das_anomaly.detect import AnomalyDetector, DetectConfig


def test_missing_model_raises(tmp_path):
    """No model file → FileNotFoundError during __init__."""
    cfg = DetectConfig(
        psd_path=tmp_path, results_path=tmp_path, train_images_path=tmp_path
    )
    with pytest.raises(FileNotFoundError):
        AnomalyDetector(cfg)
