from dataclasses import dataclass

import torch


@dataclass
class IdentityState:
    memory_keys: torch.Tensor
    memory_values: torch.Tensor
    slot_used: torch.Tensor

    @staticmethod
    def init(batch_size: int, n_slots: int, d_key: int, d_value: int, device: str) -> "IdentityState":
        return IdentityState(
            memory_keys=torch.zeros(batch_size, n_slots, d_key, device=device),
            memory_values=torch.zeros(batch_size, n_slots, d_value, device=device),
            slot_used=torch.zeros(batch_size, n_slots, device=device),
        )

