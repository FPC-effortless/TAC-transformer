import torch
from tac_osm import TACOSM,TACOSMConfig
from tac_osm.environment import SyntheticOperationalWorld,WorldConfig

def test_world_oracle_is_action_sensitive():
    w=SyntheticOperationalWorld(WorldConfig(state_dim=8,action_dim=4,noise_std=0))
    s=torch.randn(16,8)
    a0=torch.zeros(16,dtype=torch.long); a1=torch.ones(16,dtype=torch.long)
    assert not torch.allclose(w.transition(s,a0),w.transition(s,a1))

def test_multi_step_forecast_and_candidate_scores():
    cfg=TACOSMConfig(input_dim=8,hidden_dim=24,structure_dim=12,state_slots=6,top_k=2,action_dim=4,forecast_horizon=3)
    m=TACOSM(cfg); x=torch.randn(3,8); actions=torch.randn(3,4,3,4)
    out=m(x,candidate_actions=actions)
    assert out.forecast.shape==(3,4,3,8)
    assert out.action_value.shape==(3,4)
    assert torch.isfinite(out.forecast).all()
