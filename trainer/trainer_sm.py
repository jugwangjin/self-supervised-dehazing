import torch
from .decompose import Decompose
from .train_step import TrainStepSM
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
        f = model.SMCVFModel()

        trainer = TrainStepSM

        self.trainer = torch.nn.DataParallel(trainer(f).to(args["device"])) if args["usedataparallel"] else trainer(f).to(args["device"])
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

        train_dataset = dataset(root=args["dataroot"], mode='train', )
        g = torch.Generator()
        g.manual_seed(args["seed"])
        self.train_loader = torch.utils.data.DataLoader(train_dataset, args["batchsize"], shuffle=True, num_workers=args["numworkers"],
                                            pin_memory=args["pinmemory"], 
                                            worker_init_fn = seed_worker, generator=g, drop_last=True)
        val_dataset = dataset(root=args["dataroot"], mode='validation', )
        self.val_loader = torch.utils.data.DataLoader(val_dataset, args["valbatchsize"], shuffle=False, num_workers=args["numworkers"],
                                            pin_memory=args["pinmemory"], 
                                            worker_init_fn = seed_worker, generator=g)

        self.out_dir = args["outdir"]
        num = 0
        while os.path.isdir(self.out_dir+str(num)):
            num+=1
        self.out_dir = self.out_dir + str(num)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'checkpoints'), exist_ok=True)
        
        self.decompose = torch.nn.DataParallel(Decompose().to(args["device"])) if args["usedataparallel"] else Decompose().to(args["device"])
        self.decompose.requires_grad=False

    def train(self):
        train_losses = []
        validation_losses = []

        prog_bar = tqdm(range(self.epochs))
        for epoch in prog_bar:
            prog_bar.set_description(f'[Progress] epoch {epoch}/{self.epochs} - saving at {self.out_dir}')
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            self.scheduler.step()

            if epoch % self.args["validateevery"] == self.args["validateevery"] - 1:
                validation_loss = self.validation_epoch()

                while len(validation_losses) < len(train_losses):
                    validation_losses.append(validation_loss)

            plt.clf()
            plt.plot(train_losses, label='train')
            plt.plot(validation_losses, label='validation')
            plt.legend()
            plt.savefig(os.path.join(self.out_dir, 'train_plot.png'))

            if epoch % 10 == 0:
                torch.save({'f': self.trainer.module.f.state_dict() if self.args["usedataparallel"]
                                    else self.trainer.f.state_dict(),
                            'optim': self.optimizer.state_dict()},
                            os.path.join(self.out_dir, 'checkpoints', 'checkpoint.tar'))

        validation_loss = self.validation_epoch()
        torch.save({'f': self.trainer.module.f.state_dict() if self.args["usedataparallel"] 
                            else self.trainer.f.state_dict()},
                    os.path.join(self.out_dir, 'checkpoints', 'final.tar'))

    def train_epoch(self):
        num_samples = 0
        accum_losses = 0
        prog_bar = tqdm(self.train_loader)
        self.trainer.train()
        for batchIdx, img in enumerate(prog_bar):
            img = img.to(self.args["device"])
            self.optimizer.zero_grad()
            loss, L_package = self.trainer(img, return_package=True)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainer.module.f.parameters() if self.args["usedataparallel"]
                                            else self.trainer.f.parameters(), 1)
            self.optimizer.step()
            num_samples += img.size(0)
            accum_losses += loss.item() * img.size(0)

            desc = f'[Train] loss {loss:.2f} acc {accum_losses/num_samples:.2f}'
            for k in L_package:
                desc = desc + f' {k}:{L_package[k].mean():.2f}'
            prog_bar.set_description(desc)

        return accum_losses / num_samples

    @torch.no_grad()
    def validation_epoch(self):
        num_samples = 0
        accum_losses = 0
        prog_bar = tqdm(self.val_loader)
        self.trainer.eval()
        f = self.trainer.module.f if self.args["usedataparallel"] else self.trainer.f
        if self.args["usedataparallel"]:
            f = torch.nn.DataParallel(f)
        f.eval()
        for batchIdx, img in enumerate(prog_bar):
            img = img.to(self.args["device"])
            loss = self.trainer(img).mean()
            num_samples += img.size(0)
            accum_losses += loss.item() * img.size(0)
                
            noise_D, noise_I, noise_C, clean = f(img)
            T = (noise_D + 1).clamp(0, 1)
            A = (noise_I / (1 - T)).clamp(0, 1)
            rec = clean + clean * noise_D + noise_I + noise_C

            for idx in range(img.size(0)):
                torchvision.utils.save_image(noise_D[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_noise_D.png'))
                torchvision.utils.save_image(noise_I[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_noise_I.png'))
                torchvision.utils.save_image(clean[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_clean.png'))
                torchvision.utils.save_image(img[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_img.png'))
                torchvision.utils.save_image(T[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_T.png'))
                torchvision.utils.save_image(A[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_A.png'))
                torchvision.utils.save_image(noise_C[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_noise_C.png'))
                torchvision.utils.save_image(rec[idx], os.path.join(self.out_dir, 'results', f'{batchIdx * self.args["valbatchsize"] + idx}_reconstuct.png'))
            
            prog_bar.set_description(f'[Val] batch {batchIdx}/{len(prog_bar)} loss {loss:.2f} acc {accum_losses/num_samples:.2f}')

        return accum_losses / num_samples


if __name__=='__main__':
    print('Module file')