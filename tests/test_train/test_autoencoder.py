from unittest.mock import MagicMock

import pytest

from das_anomaly.train import AutoencoderTrainer, TrainAEConfig


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
    monkeypatch.setattr("das_anomaly.train.autoencoder.decoder", lambda model: model)
    monkeypatch.setattr(
        "das_anomaly.train.autoencoder.ImageDataGenerator",
        MagicMock(
            return_value=MagicMock(
                flow_from_directory=MagicMock(return_value=MagicMock(samples=4))
            )
        ),
    )

    # run
    trainer = AutoencoderTrainer(dummy_cfg)
    trainer.run()

    # ensure fit + save were called
    fake_model.fit.assert_called()
    fake_model.save.assert_called()
    fake_model.layers[0].save.assert_called()


def test_saved_files(tmp_path, monkeypatch):
    cfg = TrainAEConfig(
        train_dir=tmp_path,
        test_dir=tmp_path,
        out_dir=tmp_path / "out",
        img_size=16,
        batch_size=1,
        num_epoch=1,
    )

    # fake out heavy deps
    fake_hist = {"loss": [0.1], "val_loss": [0.1]}
    keras_model = MagicMock(name="Model")
    keras_model.fit.return_value.history = fake_hist
    keras_model.layers = [MagicMock(name="Encoder")]
    monkeypatch.setattr("das_anomaly.train.autoencoder.encoder", lambda s: keras_model)
    monkeypatch.setattr("das_anomaly.train.autoencoder.decoder", lambda m: m)
    monkeypatch.setattr(
        "das_anomaly.train.autoencoder.ImageDataGenerator",
        MagicMock(
            return_value=MagicMock(
                flow_from_directory=MagicMock(return_value=MagicMock(samples=2))
            )
        ),
    )
    monkeypatch.setattr(
        "das_anomaly.train.autoencoder.plot_train_test_loss", lambda h, p: None
    )

    AutoencoderTrainer(cfg).run()

    # artifacts really exist
    # model + encoder were asked to save to the expected locations
    keras_model.save.assert_called_with(
        cfg.out_dir / "model_16.h5", include_optimizer=False
    )
    keras_model.layers[0].save.assert_called_with(
        cfg.out_dir / "encoder_model_16", save_format="tf"
    )

    # history file written
    assert (cfg.out_dir / "history_16.json").exists()
