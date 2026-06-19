import torch


def modular_add_targets(x_digit: torch.Tensor, offset_id: torch.Tensor, n_digits: int = 10) -> torch.Tensor:
    return (x_digit + offset_id) % n_digits
