import torch

from tac_sie.losses import query_key_alignment_loss


def test_query_key_alignment_loss_prefers_correct_slot():
    memory_keys = torch.eye(3, 3).unsqueeze(0).repeat(2, 1, 1)
    correct_query = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    wrong_query = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    correct_slot = torch.tensor([1, 2])

    assert query_key_alignment_loss(correct_query, memory_keys, correct_slot) < query_key_alignment_loss(
        wrong_query, memory_keys, correct_slot
    )
