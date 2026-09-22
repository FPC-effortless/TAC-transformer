from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor, nn
from .config import TACOSMConfig

@dataclass
class PersistentStructureState:
    slots: Tensor
    validity: Tensor
    def detach(self): return PersistentStructureState(self.slots.detach(), self.validity.detach())
    def clone(self): return PersistentStructureState(self.slots.clone(), self.validity.clone())

@dataclass
class TACOSMOutput:
    decision_logits: Tensor
    verifier_logit: Tensor
    route_weights: Tensor
    selected_indices: Tensor
    active_compute: Tensor
    state: PersistentStructureState
    action_logits: Optional[Tensor]=None
    forecast: Optional[Tensor]=None
    action_value: Optional[Tensor]=None

class StructuralCompressor(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.value=nn.Sequential(nn.Linear(cfg.hidden_dim,cfg.structure_dim),nn.LayerNorm(cfg.structure_dim)); self.write_gate=nn.Linear(cfg.hidden_dim,cfg.state_slots)
    def forward(self,hidden,state):
        gate=torch.sigmoid(self.write_gate(hidden)); value=self.value(hidden).unsqueeze(1); slots=state.slots+gate.unsqueeze(-1)*(value-state.slots); validity=torch.maximum(state.validity,gate).clamp(0.,1.); return PersistentStructureState(slots,validity)

class StructureRouter(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.query=nn.Linear(cfg.hidden_dim,cfg.structure_dim); self.scale=cfg.structure_dim**-0.5; self.top_k=cfg.top_k
    def forward(self,hidden,state):
        q=self.query(hidden).unsqueeze(1); scores=(q*state.slots).sum(-1)*self.scale; scores=scores.masked_fill(state.validity<=1e-6,-1e9); vals,idx=torch.topk(scores,k=min(self.top_k,state.slots.size(1)),dim=-1); weights=torch.softmax(vals,-1); gathered=state.slots.gather(1,idx.unsqueeze(-1).expand(-1,-1,state.slots.size(-1))); return (gathered*weights.unsqueeze(-1)).sum(1),weights,idx

class ConditionalExecutor(nn.Module):
    def __init__(self,cfg):
        super().__init__(); joined=cfg.hidden_dim+cfg.structure_dim; self.modules_bank=nn.ModuleList([nn.Sequential(nn.Linear(joined,cfg.hidden_dim),nn.GELU(),nn.Linear(cfg.hidden_dim,cfg.hidden_dim)) for _ in range(cfg.num_compute_modules)]); self.gate=nn.Linear(joined,cfg.num_compute_modules); self.dropout=nn.Dropout(cfg.dropout)
    def forward(self,hidden,structure):
        joined=torch.cat([hidden,structure],-1); gate=torch.softmax(self.gate(joined),-1); outputs=torch.stack([m(joined) for m in self.modules_bank],1); return self.dropout((gate.unsqueeze(-1)*outputs).sum(1)),gate

class ActionConditionedForecaster(nn.Module):
    """Latent transition model for P(S_future | S_t, A)."""
    def __init__(self,cfg):
        super().__init__(); self.action_encoder=nn.Sequential(nn.Linear(cfg.action_dim,cfg.hidden_dim),nn.GELU(),nn.Linear(cfg.hidden_dim,cfg.hidden_dim)); self.transition=nn.Sequential(nn.Linear(cfg.hidden_dim*2+cfg.structure_dim,cfg.hidden_dim),nn.GELU(),nn.Linear(cfg.hidden_dim,cfg.hidden_dim)); self.state_head=nn.Linear(cfg.hidden_dim,cfg.input_dim)
    def step(self,latent,structure,action): return self.transition(torch.cat([latent,structure,self.action_encoder(action)],-1))
    def forward(self,latent,structure,actions):
        if actions.dim()!=3: raise ValueError("actions must have shape [batch,horizon,action_dim]")
        current=latent; predictions=[]
        for t in range(actions.size(1)): current=self.step(current,structure,actions[:,t]); predictions.append(self.state_head(current))
        return torch.stack(predictions,1)

class ActionDecisionHead(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.score=nn.Sequential(nn.Linear(cfg.input_dim,cfg.hidden_dim),nn.GELU(),nn.Linear(cfg.hidden_dim,1))
    def forward(self,forecast): return self.score(forecast[:,:,-1]).squeeze(-1)

class TACOSM(nn.Module):
    """Persistent Operational Structure Model v0.3 research substrate."""
    def __init__(self,cfg:Optional[TACOSMConfig]=None):
        super().__init__(); self.cfg=cfg or TACOSMConfig(); self.encoder=nn.Sequential(nn.Linear(self.cfg.input_dim,self.cfg.hidden_dim),nn.GELU(),nn.LayerNorm(self.cfg.hidden_dim)); self.compressor=StructuralCompressor(self.cfg); self.router=StructureRouter(self.cfg); self.executor=ConditionalExecutor(self.cfg); self.decision_head=nn.Linear(self.cfg.hidden_dim,self.cfg.decision_dim); self.verifier_head=nn.Linear(self.cfg.hidden_dim+self.cfg.structure_dim,1); self.forecaster=ActionConditionedForecaster(self.cfg); self.action_decision=ActionDecisionHead(self.cfg); self.action_head=nn.Linear(self.cfg.hidden_dim,self.cfg.action_dim)
    def initial_state(self,batch_size,*,device=None): return PersistentStructureState(torch.zeros(batch_size,self.cfg.state_slots,self.cfg.structure_dim,device=device),torch.zeros(batch_size,self.cfg.state_slots,device=device))
    def reset_state(self,state): return PersistentStructureState(torch.zeros_like(state.slots),torch.zeros_like(state.validity))
    def forward(self,observation,state=None,*,write=True,candidate_actions=None):
        if observation.dim()!=2 or observation.size(-1)!=self.cfg.input_dim: raise ValueError(f"observation must have shape [batch,{self.cfg.input_dim}]")
        if state is None: state=self.initial_state(observation.size(0),device=observation.device)
        hidden=self.encoder(observation); new_state=self.compressor(hidden,state) if write else state; structure,rw,idx=self.router(hidden,new_state); active,cg=self.executor(hidden,structure); decision=self.decision_head(active); verifier=self.verifier_head(torch.cat([active,structure],-1)).squeeze(-1); action_logits=self.action_head(active); forecast=action_value=None
        if candidate_actions is not None:
            if candidate_actions.dim()!=4: raise ValueError("candidate_actions must have shape [batch,candidates,horizon,action_dim]")
            b,n,h,a=candidate_actions.shape
            if b!=observation.size(0) or a!=self.cfg.action_dim: raise ValueError("candidate_actions has incompatible batch/action dimensions")
            fa=candidate_actions.reshape(b*n,h,a); fl=active.unsqueeze(1).expand(-1,n,-1).reshape(b*n,-1); fs=structure.unsqueeze(1).expand(-1,n,-1).reshape(b*n,-1); forecast=self.forecaster(fl,fs,fa).reshape(b,n,h,self.cfg.input_dim); action_value=self.action_decision(forecast)
        return TACOSMOutput(decision,verifier,rw,idx,cg,new_state,action_logits,forecast,action_value)
    def parameter_count(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
