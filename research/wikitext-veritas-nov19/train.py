"""Train character models on the published WikiText-2 raw training split."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from program_lm import WindowProgramLM
from nanogpt_model import GPT, GPTConfig

def read(split):
    path=Path(__file__).resolve().parent/(split+'.parquet')
    text='\n'.join(pq.read_table(path).column('text').to_pylist())
    return torch.tensor(list(text.encode()),dtype=torch.long)

def make(name):
    if name=='program':
        return WindowProgramLM(256,128,1,32,5)
    return GPT(GPTConfig(vocab_size=256,block_size=64,n_embd=72,n_head=4,n_layer=2,dropout=0.0,bias=True))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['program','gpt'],required=True)
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--steps',type=int,default=2500)
    a=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    train,valid=read('train'),read('validation')
    model=make(a.model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002)
    generator=torch.Generator().manual_seed(a.seed+567)
    starts=torch.arange(0,len(valid)-65,64*4)
    def get(data,idx):
        pos=idx[:,None]+torch.arange(64)
        return data[pos],data[pos+1]
    def objective(x,y):
        if a.model=='gpt':
            return model(x,y)[1]
        return F.cross_entropy(model(x).reshape(-1,256),y.reshape(-1))
    x,y=get(train,torch.tensor([0,100]))
    with torch.no_grad():
        def logits(tokens):
            return model(tokens,tokens.contiguous())[0] if a.model=='gpt' else model(tokens)
        torch.testing.assert_close(logits(x)[:,:32],logits(x[:,:32]),atol=1e-5,rtol=1e-4)
    t=time.perf_counter()
    for step in range(a.steps):
        x,y=get(train,torch.randint(0,len(train)-65,(16,),generator=generator))
        optimizer.zero_grad(set_to_none=True)
        loss=objective(x,y)
        loss.backward()
        optimizer.step()
        if (step+1)%500==0:
            print(f'{a.model} seed={a.seed} step={step+1} loss={loss.item():.4f}',flush=True)
    elapsed=time.perf_counter()-t
    model.eval()
    with torch.no_grad():
        val=sum(float(objective(*get(valid,b)))*len(b) for b in starts.split(32))/len(starts)
    out=Path(os.environ.get('TRAINLAB_OUTPUT_DIR','.'))
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'checkpoint.pt'
    torch.save({'model':a.model,'seed':a.seed,'steps':a.steps,'state':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':generator.get_state()},checkpoint)
    metrics={'validation_nats_per_byte':val,'training_seconds':elapsed,'parameters':sum(p.numel() for p in model.parameters()),'train_tokens':a.steps*16*64}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics),flush=True)
if __name__=='__main__':
    main()
