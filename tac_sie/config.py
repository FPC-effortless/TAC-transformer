from dataclasses import dataclass

import torch


@dataclass
class TACSIEConfig:
    vocab_size: int = 64
    n_digits: int = 10
    n_rules: int = 8
    n_offsets: int = 5
    d_model: int = 32
    d_key: int = 32
    d_value: int = 32
    d_hidden: int = 64
    n_memory_slots: int = 8
    lambda_key_orth: float = 0.1
    lambda_query_align: float = 0.1
    read_temperature: float = 1.0
    lambda_offset_vec: float = 0.5
    device: str = "cuda"

    def resolved_device(self) -> str:
        if self.device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return self.device
