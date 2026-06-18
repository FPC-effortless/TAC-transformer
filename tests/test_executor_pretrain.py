import torch

from tac_sie.config import TACSIEConfig
from tac_sie.executor import AdditionExecutor, pretrain_executor


def test_executor_pretrain_reaches_full_addition_accuracy():
    torch.manual_seed(7)
    cfg = TACSIEConfig(device="cpu", d_hidden=64)
    executor = AdditionExecutor(cfg)

    acc = pretrain_executor(executor, cfg, epochs=350, lr=3e-3)

    assert acc >= 0.98

