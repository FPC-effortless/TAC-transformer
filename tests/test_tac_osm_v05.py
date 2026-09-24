import torch
from tac_osm.environment import SyntheticOperationalWorld, WorldConfig
from tac_osm.experiment_v05 import ExperimentConfig, _context_pair, _forecast_with_context, action_candidates, train_one

def test_v05_context_intervention_changes_true_world_outcome():
    cfg=ExperimentConfig(eval_batch=32); device=torch.device("cpu")
    world=SyntheticOperationalWorld(WorldConfig(cfg.input_dim,cfg.action_dim,0.0,cfg.context_dim,1))
    state=torch.randn(cfg.eval_batch,cfg.input_dim); ca,cb=_context_pair(cfg,cfg.eval_batch,device)
    action=torch.zeros(cfg.eval_batch,dtype=torch.long)
    ya=world.transition(state,action,context=ca); yb=world.transition(state,action,context=cb)
    assert torch.mean(torch.abs(yb-ya)).item()>0.0

def test_v05_model_intervention_is_state_mediated():
    cfg=ExperimentConfig(train_steps=2,eval_batch=8); device=torch.device("cpu")
    model=train_one(0,cfg,device); state=torch.randn(cfg.eval_batch,cfg.input_dim)
    ca,cb=_context_pair(cfg,cfg.eval_batch,device); t1=torch.cat([state,torch.zeros_like(ca)],-1)
    candidates=action_candidates(cfg.eval_batch,cfg.action_dim,device)
    pa=_forecast_with_context(model,t1,ca,candidates); pb=_forecast_with_context(model,t1,cb,candidates)
    assert torch.isfinite(pa).all() and torch.isfinite(pb).all()
    assert torch.mean(torch.abs(pb-pa)).item()>0.0

def test_v05_five_seed_smoke_is_finite():
    cfg=ExperimentConfig(train_steps=3,batch_size=16,eval_batch=32); device=torch.device("cpu")
    for seed in range(5):
        model=train_one(seed,cfg,device)
        assert model.parameter_count()>0
