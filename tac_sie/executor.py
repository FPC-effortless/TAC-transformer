import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditionExecutor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.digit_embedding = nn.Embedding(cfg.n_digits, cfg.d_model)
        self.offset_embedding = nn.Embedding(cfg.n_offsets, cfg.d_model)
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model * 2, cfg.d_hidden),
            nn.ReLU(),
            nn.Linear(cfg.d_hidden, cfg.d_hidden),
            nn.ReLU(),
            nn.Linear(cfg.d_hidden, cfg.n_digits),
        )

    def forward(self, x_digit: torch.Tensor, offset_id: torch.Tensor | None = None, offset_vec: torch.Tensor | None = None):
        x_emb = self.digit_embedding(x_digit)
        if offset_vec is None:
            if offset_id is None:
                raise ValueError("offset_id is required when offset_vec is not supplied")
            k_emb = self.offset_embedding(offset_id)
        else:
            k_emb = offset_vec
        return self.net(torch.cat([x_emb, k_emb], dim=-1))


def addition_table(cfg):
    device = cfg.resolved_device()
    xs, ks, ys = [], [], []
    for x in range(cfg.n_digits):
        for k in range(cfg.n_offsets):
            xs.append(x)
            ks.append(k)
            ys.append((x + k) % cfg.n_digits)
    return (
        torch.tensor(xs, device=device, dtype=torch.long),
        torch.tensor(ks, device=device, dtype=torch.long),
        torch.tensor(ys, device=device, dtype=torch.long),
    )


def pretrain_executor(executor: AdditionExecutor, cfg, epochs: int = 1000, lr: float = 1e-3) -> float:
    device = cfg.resolved_device()
    executor.to(device)
    opt = torch.optim.AdamW(executor.parameters(), lr=lr)
    x, k, y = addition_table(cfg)

    for _ in range(epochs):
        logits = executor(x, offset_id=k)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        acc = (executor(x, offset_id=k).argmax(-1) == y).float().mean().item()
    return acc


def freeze_executor(executor: AdditionExecutor) -> AdditionExecutor:
    for param in executor.parameters():
        param.requires_grad = False
    executor.eval()
    return executor
