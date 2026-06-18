import torch

from tac_sie.losses import key_orthogonality_loss


def test_key_orthogonality_loss_penalizes_collapsed_keys():
    collapsed = torch.ones(2, 3, 4)
    separated = torch.eye(3, 4).unsqueeze(0).repeat(2, 1, 1)
    used = torch.ones(2, 3)

    assert key_orthogonality_loss(collapsed, used) > key_orthogonality_loss(separated, used)
    assert key_orthogonality_loss(separated, used).item() == 0.0

