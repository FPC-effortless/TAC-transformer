"""Deterministic smoke/evaluation runner for the synthetic operational world."""
import argparse, json, random
import numpy as np
import torch
from tac_osm import TACOSM, TACOSMConfig
from .environment import SyntheticOperationalWorld, WorldConfig

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def evaluate(model,world,steps=8,batch=128,seed=0):
    g=torch.Generator().manual_seed(seed)
    state,target,_=world.sample(batch,generator=g)
    actions=torch.eye(world.cfg.action_dim).view(1,world.cfg.action_dim,1,world.cfg.action_dim).expand(batch,-1,1,-1)
    with torch.no_grad():
        out=model(state,candidate_actions=actions)
    pred=out.action_value.argmax(1)
    return {
        "action_accuracy":float((pred==target).float().mean()),
        "forecast_finite":bool(torch.isfinite(out.forecast).all()),
        "score_finite":bool(torch.isfinite(out.action_value).all()),
        "steps":steps,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=0); p.add_argument("--batch",type=int,default=128)
    args=p.parse_args(); seed_all(args.seed)
    w=SyntheticOperationalWorld(WorldConfig(state_dim=8,action_dim=4))
    cfg=TACOSMConfig(input_dim=8,hidden_dim=32,structure_dim=16,state_slots=8,top_k=3,action_dim=4,forecast_horizon=1)
    m=TACOSM(cfg)
    print(json.dumps({"parameter_count":m.parameter_count(),**evaluate(m,w,batch=args.batch,seed=args.seed)},indent=2))

if __name__=="__main__": main()
