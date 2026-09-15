import cv2
import numpy as np
import torch

from scripts.deep_geometry import clahe_frame, copy_paste, symmetric_cross_entropy


def test_sce_reduces_to_ce_when_beta_zero() -> None:
    logits = torch.randn(2, 4, 8, 8)
    targets = torch.zeros(2, 8, 8, dtype=torch.long)
    sce = symmetric_cross_entropy(logits, targets, alpha=1.0, beta=0.0)
    ce = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(sce, ce, atol=1e-6)


def test_sce_penalizes_mismatched_logits() -> None:
    targets = torch.zeros(1, 8, 8, dtype=torch.long)
    confident = torch.zeros(1, 4, 8, 8)
    confident[:, 0] = 5.0
    wrong = torch.zeros(1, 4, 8, 8)
    wrong[:, 2] = 5.0
    assert symmetric_cross_entropy(wrong, targets) > symmetric_cross_entropy(
        confident, targets
    )


def test_clahe_increases_l_channel_contrast() -> None:
    gradient = np.tile(np.linspace(60, 70, 64, dtype=np.uint8), (64, 1))
    low_contrast = np.stack([gradient, gradient, gradient], axis=-1)

    def l_std(image: np.ndarray) -> float:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return float(lab[:, :, 0].std())

    assert l_std(clahe_frame(low_contrast)) > l_std(low_contrast)


def test_copy_paste_adds_foreground_class() -> None:
    rng = np.random.default_rng(0)
    source = torch.zeros(3, 16, 16)
    source[1, 4:8, 4:8] = 1.0
    source_mask = torch.zeros(16, 16, dtype=torch.long)
    source_mask[4:8, 4:8] = 2
    target = torch.zeros(3, 16, 16)
    target[2, 10:14, 10:14] = 1.0
    target_mask = torch.zeros(16, 16, dtype=torch.long)
    target_mask[10:14, 10:14] = 3
    before = int((source_mask > 0).sum() + (target_mask > 0).sum())
    _, masks = copy_paste(
        torch.stack([source, target]),
        torch.stack([source_mask, target_mask]),
        rng,
        probability=1.0,
    )
    after = int((masks > 0).sum())
    assert after > before
