import os
# config
import argparse
import trainer
import json 
import importlib

def main(args):
    trainer_module = importlib.import_module("trainer."+args["trainer"])
    trainer_ = trainer_module.Trainer(args)
    trainer_.train()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    args = vars(args)
    
    with open('config.json', 'r') as f:
        opt = json.load(f)[args['config']]
        for key in opt:
            args[key] = opt[key]

    print(args)

    main(args)





