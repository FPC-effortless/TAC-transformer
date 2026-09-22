from dataclasses import dataclass

@dataclass(frozen=True)
class TACOSMConfig:
    input_dim: int = 32
    hidden_dim: int = 64
    state_slots: int = 16
    top_k: int = 4
    structure_dim: int = 32
    num_compute_modules: int = 4
    decision_dim: int = 8
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.state_slots < 1:
            raise ValueError("state_slots must be >= 1")
        if not 1 <= self.top_k <= self.state_slots:
            raise ValueError("top_k must be in [1, state_slots]")
        if self.num_compute_modules < 1:
            raise ValueError("num_compute_modules must be >= 1")
        if min(self.input_dim, self.hidden_dim, self.structure_dim, self.decision_dim) < 1:
            raise ValueError("dimensions must be positive")
