import torch
import torch.nn.functional as F


def key_orthogonality_loss(memory_keys: torch.Tensor, slot_used: torch.Tensor | None = None) -> torch.Tensor:
    k = F.normalize(memory_keys, dim=-1)
    sim = torch.einsum("bnd,bmd->bnm", k, k)
    n = sim.size(1)
    eye = torch.eye(n, device=sim.device).unsqueeze(0)
    off_diag = sim * (1.0 - eye)

    if slot_used is not None:
        mask = torch.einsum("bn,bm->bnm", slot_used, slot_used) * (1.0 - eye)
        off_diag = off_diag * mask
        denom = mask.sum().clamp_min(1.0)
    else:
        denom = max(off_diag.numel() - sim.size(0) * n, 1)

    return (off_diag**2).sum() / denom


def query_key_alignment_loss(query_keys: torch.Tensor, memory_keys: torch.Tensor, correct_slot: torch.Tensor) -> torch.Tensor:
    q = F.normalize(query_keys, dim=-1)
    k = F.normalize(memory_keys, dim=-1)
    logits = torch.einsum("bd,bnd->bn", q, k)
    return F.cross_entropy(logits, correct_slot)


def offset_vector_distillation_loss(pred_offset_vec: torch.Tensor, target_offset_vec: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_offset_vec, target_offset_vec.detach())

