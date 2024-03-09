import torch
from src.preprocessing import FakeDetectorPreprocessor
from src.models.inception import Inception, BinaryInceptionLoss
from pathlib import Path

current_file_path = Path(__file__)
project_root = current_file_path.parent.parent.parent
DATA_PATH = project_root / "data"

DATA_PATH = str(DATA_PATH)

preprocessor = FakeDetectorPreprocessor(DATA_PATH)
preprocessor.prepare_data()

test_batch_images, test_batch_labels = next(iter(preprocessor.train_loader))

n_classes = 1


def test_inception_forward_pass():
    """
    Test the forward pass of the Inception module:
        1. Check the output type of the forward pass in evaluation.
        2. Check the output shape of the forward pass in evaluation.
        3. Check the output type of the forward pass in training.
        4. Check the output tuple length of the forward pass in training.
        5. Check the output type of each element in the tuple of the forward pass in training.
        6. Check the output shape of each element in the tuple of the forward pass in training.
    """

    model = Inception(1)
    model.eval()
    with torch.no_grad():
        output = model(test_batch_images)
    
    assert type(output) == torch.Tensor
    assert output.shape == (test_batch_images.shape[0], n_classes)

    model.train()
    output = model(test_batch_images)
    assert type(output) == tuple
    assert len(output) == 3
    for o in output:
        assert type(o) == torch.Tensor
        assert o.shape == (test_batch_images.shape[0], n_classes)

def test_binary_inception_loss():
    """
    Test the binary inception loss:
        1. Check the output type of the loss.
        2. Check if the loss is raising a ValueError when we use the loss in evaluation.
    """

    model = Inception(1)
    criterion = BinaryInceptionLoss()

    model.train()
    output = model(test_batch_images)
    loss = criterion(output, test_batch_labels.unsqueeze(1).type(torch.float32))
    assert type(loss) == torch.Tensor

    model.eval()
    with torch.no_grad():
        output = model(test_batch_images)
        
    try:
        loss = criterion(output, test_batch_labels.unsqueeze(1).type(torch.float32))
    except ValueError:
        pass
    else:
        assert False

