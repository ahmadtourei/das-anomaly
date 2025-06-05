from unittest.mock import MagicMock

import pytest
from das_anomaly.train import TrainAEConfig, AutoencoderTrainer


@pytest.fixture
def dummy_cfg(tmp_path):
    return TrainAEConfig(
        train_dir=tmp_path / "train",
        test_dir=tmp_path / "test",
        out_dir=tmp_path / "out",
        img_size=32,
        batch_size=2,
        num_epoch=1,
    )


def test_trainer_build_and_save(monkeypatch, dummy_cfg):
    # ------------------------------------------------------------------ #
    # Patch heavy TF objects with cheap stubs
    # ------------------------------------------------------------------ #
    fake_model = MagicMock(name="keras.Model")
    fake_model.fit.return_value.history = {"loss": [0.1], "val_loss": [0.1]}
    fake_model.layers = [MagicMock(name="encoder")]
    fake_model.layers[0].save = MagicMock()

    monkeypatch.setattr(
        "das_anomaly.train.autoencoder.encoder", lambda size: fake_model
    )
    monkeypatch.setattr(
        "das_anomaly.train.autoencoder.decoder", lambda model: model
    )
    monkeypatch.setattr(
        "das_anomaly.train.autoencoder.ImageDataGenerator",
        MagicMock(return_value=MagicMock(
            flow_from_directory=MagicMock(return_value=MagicMock(samples=4))
        )),
    )

    # run
    trainer = AutoencoderTrainer(dummy_cfg)
    trainer.run()

    # ensure fit + save were called
    fake_model.fit.assert_called()
    fake_model.save.assert_called()
    fake_model.layers[0].save.assert_called()
