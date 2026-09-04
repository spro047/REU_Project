import numpy as np
import torch

from scripts.deep_geometry import TinyUNet, mask_to_tensor


def test_model_forward_shape() -> None:
    model = TinyUNet()
    batch = torch.zeros(2, 3, 96, 96)
    output = model(batch)
    assert output.shape == (2, 4, 96, 96)


def test_mask_to_tensor_preserves_classes() -> None:
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[:20, :20] = 1
    mask[40:60, 40:60] = 2
    mask[70:90, 10:30] = 3
    tensor = mask_to_tensor(mask)
    assert tensor.shape == (96, 96)
    assert {int(x) for x in tensor.unique()} == {0, 1, 2, 3}


def test_training_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = TinyUNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    batch = torch.rand(4, 3, 96, 96)
    target = torch.zeros(4, 96, 96, dtype=torch.long)
    target[:, 10:50, 10:50] = 1
    optimizer.zero_grad()
    before = float(loss_fn(model(batch), target).detach())
    loss = loss_fn(model(batch), target)
    loss.backward()
    optimizer.step()
    after = float(loss_fn(model(batch), target).detach())
    assert after < before
