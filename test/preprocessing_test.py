from src.preprocessing import FakeDetectorPreprocessor
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder

current_file_path = Path(__file__)
project_root = current_file_path.parent.parent.parent
DATA_PATH = project_root / "data"

DATA_PATH = str(DATA_PATH)


def test_fake_detector_preprocessor_prepare_data():
    """
    Test the `prepare_data` method of the `FakeDetectorPreprocessor` class:
        1. Check the type of the `train_loader` attribute.
        2. Check the type of the `val_loader` attribute.
        3. Check the type of the `test_loader` attribute.
    """
    preprocessor = FakeDetectorPreprocessor(DATA_PATH)
    preprocessor.prepare_data()

    assert type(preprocessor.train_loader) == torch.utils.data.dataloader.DataLoader
    assert type(preprocessor.val_loader) == torch.utils.data.dataloader.DataLoader
    assert type(preprocessor.test_loader) == torch.utils.data.dataloader.DataLoader


def test_fake_detector_splits_size():
    """
    Test the size of the splits in the `FakeDetectorPreprocessor` class:
        1. Check the size of the training split.
        2. Check the size of the validation split.
        3. Check the size of the test split.
    """
    preprocessor = FakeDetectorPreprocessor(DATA_PATH)
    preprocessor.prepare_data()

    # placeholder values (0.7, 0.15, 0.15)
    assert len(preprocessor.train_loader.dataset) == (0.7 * ImageFolder(root=DATA_PATH, transform=preprocessor.train_transforms))
    assert len(preprocessor.val_loader.dataset) == (0.15 * ImageFolder(root=DATA_PATH, transform=preprocessor.test_transforms))
    assert len(preprocessor.test_loader.dataset) == (0.15 * ImageFolder(root=DATA_PATH, transform=preprocessor.test_transforms))


