"""Tiny causal operational world for TAC-OSM experiments.

Actions are discrete one-hot vectors. The environment exposes the transition
oracle explicitly so causal access is controlled and measurable.
"""
from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass(frozen=True)
class WorldConfig:
    state_dim:int=8
    action_dim:int=4
    noise_std:float=0.02

class SyntheticOperationalWorld:
    def __init__(self,cfg=WorldConfig()):
        self.cfg=cfg
        if cfg.state_dim<4 or cfg.action_dim<2: raise ValueError("state_dim/action_dim too small")
        self.effect=torch.zeros(cfg.action_dim,cfg.state_dim)
        self.effect[0,0]=0.8; self.effect[0,1]=-0.5
        self.effect[1,0]=-0.6; self.effect[1,2]=0.7
        self.effect[2,1]=0.5; self.effect[2,3]=-0.6
        self.effect[3,2]=-0.5; self.effect[3,3]=0.8
        self.dynamics=torch.eye(cfg.state_dim)*0.88
        self.bias=torch.zeros(cfg.state_dim)
    def transition(self,state,action_index,generator=None):
        if state.dim()!=2: raise ValueError("state must be [batch,state_dim]")
        action=torch.nn.functional.one_hot(action_index,num_classes=self.cfg.action_dim).float()
        noise=torch.randn(state.shape,device=state.device,generator=generator)*self.cfg.noise_std
        return state@self.dynamics.T + action@self.effect + self.bias + noise
    def objective(self,state):
        # Lower pressure/backlog is better; stability is preferred.
        return -(state[:,0].abs()+0.7*state[:,1].abs()+0.4*state[:,2].abs()+0.2*state[:,3].abs())
    def optimal_action(self,state,generator=None):
        scores=[]
        for a in range(self.cfg.action_dim):
            scores.append(self.objective(self.transition(state,torch.full((state.size(0),),a,device=state.device,dtype=torch.long),generator)))
        return torch.stack(scores,1).argmax(1), torch.stack(scores,1)
    def sample(self,batch_size,generator=None,device=None):
        s=torch.randn(batch_size,self.cfg.state_dim,device=device,generator=generator)
        target,scores=self.optimal_action(s,generator)
        return s,target,scores
