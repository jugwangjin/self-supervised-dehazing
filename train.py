import os
# config
import argparse
from trainer.trainer import Trainer


def main(args):
    trainer = Trainer()
    trainer.train()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()

    main(args)
