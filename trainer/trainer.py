import torch
from .train_step import TrainStep
from .decompose import Decompose
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision
import sys
sys.path.append('..')

import model.model as model
import os
from tqdm import tqdm
import numpy
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        f = model.CVFModel()
        self.trainer = torch.nn.DataParallel(TrainStep(f).to(args["device"])) if args["usedataparallel"] else TrainSter(f).to(args["device"])
        self.trainer.requires_grad=False
        
        self.optimizer = torch.optim.Adam(self.trainer.module.f.parameters() if args["usedataparallel"]
                                        else self.trainer.f.parameters(), lr=args["lr"], weight_decay=1e-9, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args["scheduler_step"], gamma=0.99)
        self.epochs = args["epochs"]
        
        if args["dataset"] == 'realhaze':
            from dataset import RealHazyDataset
            dataset = RealHazyDataset
        elif args["dataset"] == 'reside':
            from dataset import RESIDEHazyDataset
            dataset = RESIDEHazyDataset
        else:
            raise Exception("Not implemented dataset")

        trainDataset = dataset(root=args["dataroot"], mode='train', )
        g = torch.Generator()
        g.manual_seed(args["seed"])
        self.trainLoader = torch.utils.data.DataLoader(trainDataset, args["batchsize"], shuffle=True, num_workers=args["numworkers"],
                                            pin_memory=args["pinmemory"], 
                                            worker_init_fn = seed_worker, generator=g, drop_last=True)
        valDataset = dataset(root=args["dataroot"], mode='validation', )
        self.valLoader = torch.utils.data.DataLoader(valDataset, args["valbatchsize"], shuffle=False, num_workers=args["numworkers"],
                                            pin_memory=args["pinmemory"], 
                                            worker_init_fn = seed_worker, generator=g)

        self.outDir = args["outdir"]
        num = 0
        while os.path.isdir(self.outDir+str(num)):
            num+=1
        self.outDir = self.outDir + str(num)
        os.makedirs(self.outDir, exist_ok=True)
        os.makedirs(os.path.join(self.outDir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.outDir, 'checkpoints'), exist_ok=True)
        
        self.decompose = torch.nn.DataParallel(Decompose().to(args["device"])) if args["usedataparallel"] else Decompose().to(args["device"])
        self.decompose.requires_grad=False

    def train(self):
        trainLosses = []
        validationLosses = []

        progBar = tqdm(range(self.epochs))
        for epoch in progBar:
            progBar.set_description(f'[Progress] epoch {epoch}/{self.epochs} - saving at {self.outDir}')
            trainLoss = self.train_epoch()
            self.scheduler.step()

            if epoch % self.args["validateevery"] == self.args["validateevery"] - 1:
                validationLoss = self.validation_epoch()

            trainLosses.append(trainLoss)
            while len(validationLosses) < len(trainLosses):
                validationLosses.append(validationLoss)

            plt.clf()
            plt.plot(trainLosses, label='train')
            plt.plot(validationLosses, label='validation')
            plt.legend()
            plt.savefig(os.path.join(self.outDir, 'train_plot.png'))

            if epoch % 10 == 0:
                torch.save({'f': self.trainer.module.f.state_dict() if self.args["usedataparallel"]
                                    else self.trainer.f.state_dict(),
                            'optim': self.optimizer.state_dict()},
                            os.path.join(self.outDir, 'checkpoints', 'checkpoint.tar'))


        validationLoss = self.validation_epoch()
        torch.save({'f': self.trainer.module.f.state_dict() if self.args["usedataparallel"] 
                            else self.trainer.f.state_dict()},
                    os.path.join(self.outDir, 'checkpoints', 'final.tar'))

    def train_epoch(self):
        numSamples = 0
        accumLosses = 0
        progBar = tqdm(self.trainLoader)
        for batchIdx, img in enumerate(progBar):
            img = img.to(self.args["device"])
            loss = self.trainer(img).mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainer.module.f.parameters() if self.args["usedataparallel"]
                                            else self.trainer.f.parameters(), 1)
            self.optimizer.step()
            numSamples += img.size(0)
            accumLosses += loss.item() * img.size(0)

            progBar.set_description(f'[Train] batch {batchIdx}/{len(progBar)} loss {loss:.2f} acc {accumLosses/numSamples:.2f}')

        return accumLosses / numSamples

    @torch.no_grad()
    def validation_epoch(self):
        numSamples = 0
        accumLosses = 0
        progBar = tqdm(self.valLoader)
        f = self.trainer.module.f if self.args["usedataparallel"] else self.trainer.f
        if self.args["usedataparallel"]:
            f = torch.nn.DataParallel(f)
        for batchIdx, img in enumerate(progBar):
            img = img.to(self.args["device"])
            loss = self.trainer(img).mean()
            numSamples += img.size(0)
            accumLosses += loss.item() * img.size(0)
                
            noiseD, noiseI, clean = f(img)
            T, A, captureNoise = self.decompose(noiseD, noiseI, clean)
            rec = clean + clean * noiseD + noiseI

            for idx in range(img.size(0)):
                torchvision.utils.save_image(noiseD[idx], os.path.join(self.outDir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_noiseD.png'))
                torchvision.utils.save_image(noiseI[idx], os.path.join(self.outDir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_noiseI.png'))
                torchvision.utils.save_image(clean[idx], os.path.join(self.outDir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_clean.png'))
                torchvision.utils.save_image(img[idx], os.path.join(self.outDir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_img.png'))
                torchvision.utils.save_image(T[idx], os.path.join(self.outDir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_T.png'))
                torchvision.utils.save_image(A[idx], os.path.join(self.outDir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_A.png'))
                torchvision.utils.save_image(captureNoise[idx], os.path.join(self.outDir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_captureNoise.png'))
                torchvision.utils.save_image(rec[idx], os.path.join(self.outDir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_reconstuct.png'))
            
            progBar.set_description(f'[Val] batch {batchIdx}/{len(progBar)} loss {loss:.2f} acc {accumLosses/numSamples:.2f}')

        return accumLosses / numSamples


if __name__=='__main__':
    print('Module file')