import pytest
import torch

from src.data.tinyshakespeare_datamodule import TinyShakespeareDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_tiny_shakespeare_datamodule(batch_size: int) -> None:
    """Tests `Tiny_Shakespeare`.

    to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"
    block_size = 128

    dm = TinyShakespeareDataModule(
        data_dir=data_dir, batch_size=batch_size, block_size=block_size
    )

    # before prepare_data, splits should be None
    dm.prepare_data()
    assert dm.data_train is None
    assert dm.data_val is None
    assert dm.data_test is None

    # after setup, splits should be populated
    dm.setup()
    assert dm.data_train is not None
    assert dm.data_val is not None
    assert dm.data_test is not None

    # dataloaders should be creatable
    assert dm.train_dataloader() is not None
    assert dm.val_dataloader() is not None
    assert dm.test_dataloader() is not None

    # check batch structure
    batch = next(iter(dm.train_dataloader()))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    # check batch size
    assert batch["input_ids"].shape[0] == batch_size

    # check sequence length matches block_size
    assert batch["input_ids"].shape[1] == block_size

    # check dtypes
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].dtype == torch.long
    assert batch["labels"].dtype == torch.long

    # check all splits have data
    assert len(dm.data_train) > 0
    assert len(dm.data_val) > 0
    assert len(dm.data_test) > 0
