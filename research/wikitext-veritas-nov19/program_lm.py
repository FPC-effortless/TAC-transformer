"""Causal state mixer with token-conditioned retrieval from a learned program bank."""
import torch
from torch import nn
from torch.nn import functional as F
from tac_model import TACConfig, CausalStateMixer


class ProgramLM(nn.Module):
    def __init__(self, vocab, width, depth, programs, kernel):
        super().__init__()
        self.embed = nn.Embedding(vocab,width)
        cfg = TACConfig(vocab_size=vocab,d_model=width,state_mixer_kernel_size=kernel)
        self.mixers = nn.ModuleList(CausalStateMixer(cfg) for _ in range(depth))
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(depth))
        self.routers = nn.ModuleList(nn.Linear(width, programs) for _ in range(depth))
        self.banks = nn.ParameterList(nn.Parameter(torch.randn(programs,width)*.02) for _ in range(depth))
        self.readout = nn.Sequential(nn.LayerNorm(width), nn.Linear(width,vocab))

    def forward(self,x):
        h = self.embed(x)
        for mix,norm,route,bank in zip(self.mixers,self.norms,self.routers,self.banks):
            state = norm(h)
            h = h + mix(state)
            h = h + route(h).softmax(-1) @ bank
        return self.readout(h)


class WindowProgramLM(nn.Module):
    """Route a short causal window to learned program vectors before prediction."""
    def __init__(self,vocab,width,depth,programs,kernel):
        super().__init__()
        self.kernel = kernel
        self.embed = nn.Embedding(vocab,width)
        self.compress = nn.Linear(width*kernel,width)
        self.route = nn.Linear(width,programs)
        self.bank = nn.Parameter(torch.randn(programs,width)*.02)
        self.readout = nn.Linear(width,vocab)

    def forward(self,x):
        h = self.embed(x)
        window = F.pad(h,(0,0,self.kernel-1,0)).unfold(1,self.kernel,1)
        h = h + F.gelu(self.compress(window.flatten(2)))
        return self.readout(h + self.route(h).softmax(-1) @ self.bank)
