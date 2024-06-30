import torch
from src.preprocessing import CVPreprocessor
from src.models.resnet50 import ResNet50
from pathlib import Path

current_file_path = Path(__file__)
project_root = current_file_path.parent.parent.parent
DATA_PATH = project_root / "data"

DATA_PATH = str(DATA_PATH)


preprocessor = CVPreprocessor(DATA_PATH)
preprocessor.prepare_data()

test_batch_images, test_batch_labels = next(iter(preprocessor.train_loader))

n_classes = 1


def test_resnet50_forward_pass():
    """
    Test the forward pass of the ResNet50 model:
        1. Check the output type of the forward pass in evaluation.
        2. Check the output shape of the forward pass in evaluation.
        3. Check the output type of the forward pass in training.
        4. Check the output shape of the forward pass in training.
    """

    model = ResNet50(1)
    model.eval()
    with torch.no_grad():
        output = model(test_batch_images)
    
    assert type(output) == torch.Tensor
    assert output.shape == (test_batch_images.shape[0], n_classes)

    model.train()
    output = model(test_batch_images)
    assert type(output) == torch.Tensor
    assert output.shape == (test_batch_images.shape[0], n_classes)
