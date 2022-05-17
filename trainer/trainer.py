import torch
from .train_step import TrainStep
from decompose import Decompose
from dataset import RealHazyDataset
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import model.model.CVFModel as CVFModel
import os
from tqdm import tqdm

EPOCHS = 1000
LR = 1e-4
BATCHSIZE = 64
VALBATCHSIZE = 1
NUMWORKERS = 4
PINMEMORY=True
SEED = 20202464
OUTDIR = '/Bean/log/gwangjin/CVF-dehazing/baseline'
DATAROOT = '/Bean/data/gwangjin/RESIDE_beta'
DEVICE='cuda'
USEDATAPARALLEL = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        f = CVFModel()
        self.f = torch.nn.DataParallel(f.to(DEVICE)) if USEDATAPARALLEL else f.to(DEVICE)
        self.trainStep = torch.nn.DataParallel(TrainStep().to(DEVICE)) if USEDATAPARALLEL else TrainSter().to(DEVICE)
        self.trainStep.requires_grad=False
        
        self.optimizer = torch.optim.Adam(f.parameters(), lr=lr, weight_decay=1e-9, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.99)
        self.epochs = EPOCHS

        trainDataset = RealHazyDataset(root=DATAROOT, mode='train', )
        g = torch.Generator()
        g.manual_seed(SEED)
        self.trainLoader = torch.utils.data.DataLoader(train_dataset, BATCHSIZE, shuffle=True, num_workers=NUMWORKERS,
                                            pin_memory=PINMEMORY, 
                                            worker_init_fn = seed_worker, generator=g, drop_last=True)
        valDataset = RealHazyDataset(root=DATAROOT, mode='validation', )
        self.valLoader = torch.utils.data.DataLoader(valDataset, VALBATCHSIZE, shuffle=False, num_workers=NUMWORKERS,
                                            pin_memory=PINMEMORY, 
                                            worker_init_fn = seed_worker, generator=g)

        self.outDir = OUTDIR
        num = 0
        while os.path.isdir(self.outDir+str(num)):
            num+=1
        self.outDir = self.outDir + str(num)
        os.makedirs(self.outDir, exist_ok=True)
        os.makedirs(os.path.join(self.outDir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.outDir, 'checkpoints'), exist_ok=True)
        
        self.decompose = torch.nn.DataParallel(Decompose().to(DEVICE)) if USEDATAPARALLEL else Decompose().to(DEVICE)
        self.decompose.requires_grad=False

    def train(self):
        trainLosses = []
        validationLosses = []

        progBar = tqdm(range(self.epochs))
        for epoch in progBar:
            trainLoss = self.train_epoch()
            validationLoss = self.validation_epoch()
            self.scheduler.step()

            trainLosses.append(trainLoss)
            validationLosses.append(validationLoss)

            plt.clf()
            plt.plot(trainLosses, label='train')
            plt.plot(validationLosses, label='validation')
            plt.legend()
            plt.savefig(os.path.join(self.outDir, 'train_plot.png'))

            if epoch % 10 == 0:
                fStateDict = self.f.module.state_dict() if USEDATAPARALLEL else self.f.state_dict()
                torch.save({'f': fStateDict,
                            'optim': self.optimizer.state_dict()},
                            os.path.join(self.outDir, 'checkpoints', 'checkpoint.tar'))

            progBar.set_description(f'[Progress] epoch {epoch}/{self.epohcs}')

        fStateDict = self.f.module.state_dict() if USEDATAPARALLEL else self.f.state_dict()
        torch.save({'f': fStateDict},
                    os.path.join(self.outDir, 'checkpoints', 'final.tar'))

    def train_epoch(self):
        numSamples = 0
        accumLosses = 0
        progBar = tqdm(self.trainLoader)
        for batchIdx, img in enumerate(progbar):
            loss = self.trainStep(self.f, img)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            numSamples += img.size(0)
            accumLosses += loss.item() * img.size(0)

            progbar.set_description(f'[Train] batch {batchIdx} loss {loss:.2f} acc {accumLosses/numSamples:.2f}')

        return accumLosses / numSamples

    @torch.no_grad()
    def validation_epoch(self):
        numSamples = 0
        accumLosses = 0
        progBar = tqdm(self.valLoader)
        for batchIdx, img in enumerate(progBar):
            loss = self.trainStep(self.f, img)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            numSamples += img.size(0)
            accumLosses += loss.item() * img.size(0)
                
            noiseD, noiseI, clean = self.f(img)
            T, A, captureNoise = self.decompose(noiseD, noiseI, clean)

            for idx in range(img.size(0)):
                torchvision.utils.save_image(noiseD[idx], os.path.join(self.outDir, 'results', f'{batchIdx * VALBATCHSIZE + idx}_noiseD.png'))
                torchvision.utils.save_image(noiseI[idx], os.path.join(self.outDir, 'results', f'{batchIdx * VALBATCHSIZE + idx}_noiseI.png'))
                torchvision.utils.save_image(clean[idx], os.path.join(self.outDir, 'results', f'{batchIdx * VALBATCHSIZE + idx}_clean.png'))
                torchvision.utils.save_image(img[idx], os.path.join(self.outDir, 'results', f'{batchIdx * VALBATCHSIZE + idx}_img.png'))
                torchvision.utils.save_image(T[idx], os.path.join(self.outDir, 'results', f'{batchIdx * VALBATCHSIZE + idx}_T.png'))
                torchvision.utils.save_image(A[idx], os.path.join(self.outDir, 'results', f'{batchIdx * VALBATCHSIZE + idx}_A.png'))
                torchvision.utils.save_image(captureNoise[idx], os.path.join(self.outDir, 'results', f'{batchIdx * VALBATCHSIZE + idx}_captureNoise.png'))
            
            progbar.set_description(f'[Val] batch {batchIdx} loss {loss:.2f} acc {accumLosses/numSamples:.2f}')

        return accumLosses / numSamples


if __name__=='__main__':
    print('Module file')