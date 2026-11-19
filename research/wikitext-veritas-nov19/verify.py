"""Recompute a saved checkpoint's held-out validation loss."""
import argparse
import hashlib
import json
import torch
from train import read, make
from torch.nn import functional as F

def main():
    p=argparse.ArgumentParser()
    p.add_argument('checkpoint')
    a=p.parse_args()
    torch.set_num_threads(2)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=make(checkpoint['model'])
    model.load_state_dict(checkpoint['state'],strict=True)
    model.eval()
    data=read('validation')
    starts=torch.arange(0,len(data)-65,64*4)
    loss_sum=0.0
    with torch.no_grad():
        for indices in starts.split(32):
            position=indices[:,None]+torch.arange(64)
            x,y=data[position],data[position+1]
            logits=model(x,x.contiguous())[0] if checkpoint['model']=='gpt' else model(x)
            loss_sum+=float(F.cross_entropy(logits.reshape(-1,256),y.reshape(-1)))*len(indices)
    print(json.dumps({'model':checkpoint['model'],'seed':checkpoint['seed'],
      'steps':checkpoint['steps'],'checkpoint_sha256':hashlib.sha256(open(a.checkpoint,'rb').read()).hexdigest(),
      'validation_nats_per_byte':loss_sum/len(starts),'validation_windows':len(starts)},indent=2))
if __name__=='__main__':
    main()
