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
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--config_lambda', type=str, required=True)
    parser.add_argument('--use_bean', action='store_true', default=False)
    args = parser.parse_args()
    args = vars(args)
    
    with open('config.json', 'r') as f:
        opt = json.load(f)["options"][args['config']]
        for key in opt:
            args[key] = opt[key]
        opt = json.load(f)["lambdas"][args['config_lambda']]
        args["lambdas"] = opt

    if args["use_bean"] == True:
        args["outdir"] = os.path.join("/Bean/log/gwangjin/CVF-dehazing/", args["config"], args["config_lambda"])
        args["dataroot"] = os.path.join("/Bean/data/gwangjin/", args["dataroot"])
    else:
        args["outdir"] = os.path.join("/Jarvis/workspace/gwangjin/dehazing/cvf-results/", args["config"], args["config_lambda"])
        args["dataroot"] = os.path.join("/data1/gwangjin/dehazing_bench/", args["dataroot"])

    print(args)

    main(args)