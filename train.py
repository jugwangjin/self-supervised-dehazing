import os
# config
import argparse
from trainer.trainer_sm import Trainer
import json 

def main(args):
    trainer = Trainer(args)
    
    trainer.train()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='realhaze')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--valbatchsize', type=int, default=1)
    parser.add_argument('--numworkers', type=int, default=4)
    parser.add_argument('--pinmemory', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=20202464)
    parser.add_argument('--outdir', type=str, default='/data1/gwangjin/gwangjin/CVF-dehazing/baseline')
    parser.add_argument('--dataroot', type=str, default='/data1/gwangjin/dehazing_bench/RESIDE_beta')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--usedataparallel', type=bool, default=True)
    parser.add_argument('--validateevery', type=int, default=5)
    parser.add_argument('--scheduler_step', type=int, default=5)
    parser.add_argument('--model', type=str, default='CVFModel')
    parser.add_argument('--trainer', type=str, default='TrainStep')
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    args = vars(args)

    if args['config'] is not None:
        with open('config.json', 'r') as f:
            opt = json.load(f)[args['config']]
            for key in opt:
                args[key] = opt[key]

    print(args)

    main(args)
