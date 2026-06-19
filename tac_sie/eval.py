import torch
import torch.nn.functional as F


def avg_key_cosine(memory_keys: torch.Tensor, slot_used: torch.Tensor | None = None) -> float:
    k = F.normalize(memory_keys, dim=-1)
    sim = torch.einsum("bnd,bmd->bnm", k, k)
    n = sim.size(1)
    eye = torch.eye(n, device=sim.device).unsqueeze(0)
    mask = 1.0 - eye
    if slot_used is not None:
        mask = mask * torch.einsum("bn,bm->bnm", slot_used, slot_used)
    denom = mask.sum().clamp_min(1.0)
    return (sim.abs() * mask).sum().div(denom).item()


def attention_diagnostics(attn: torch.Tensor, correct_slot: torch.Tensor) -> dict[str, float]:
    batch_idx = torch.arange(attn.size(0), device=attn.device)
    correct = attn[batch_idx, correct_slot]
    masked = attn.clone()
    masked[batch_idx, correct_slot] = -1.0
    margin = correct - masked.max(dim=-1).values
    entropy = -(attn.clamp_min(1e-9) * attn.clamp_min(1e-9).log()).sum(dim=-1)
    return {
        "correct_slot_attention": correct.mean().item(),
        "correct_slot_margin": margin.mean().item(),
        "read_attention_entropy": entropy.mean().item(),
    }
