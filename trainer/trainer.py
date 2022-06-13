import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision
import sys
sys.path.append('..')

import model.model as model
import trainer.train_step as train_step
import trainer.saver as saver
import dataset

import os
import shutil
from tqdm import tqdm
import numpy
import random

import importlib

torch.manual_seed(20202464)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

class Trainer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        f = getattr(model, self.args["model"])()

        trainer = getattr(train_step, self.args["trainstep"])

        self.trainer = torch.nn.DataParallel(trainer(f, args).to(args["device"])) if args["usedataparallel"] else trainer(f, args).to(args["device"])
        
        self.optimizer = torch.optim.Adam(self.trainer.module.f.parameters() if args["usedataparallel"]
                                        else self.trainer.f.parameters(), lr=args["lr"], weight_decay=1e-9, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args["scheduler_step"], gamma=0.99)
        self.epochs = args["epochs"]
        
        self.saver = getattr(saver, self.args["saver"])()

        if ["patchsize"] not in self.args["patchsize"]:
            self.args["patchsize"] = 128

        train_dataset_module = getattr(dataset, self.args["traindataset"])
        val_dataset_module = getattr(dataset, self.args["valdataset"])

        train_dataset = train_dataset_module(root=args["dataroot"], mode='train', patch_size=self.args["patchsize"])
        g = torch.Generator()
        g.manual_seed(args["seed"])
        self.train_loader = torch.utils.data.DataLoader(train_dataset, args["batchsize"], shuffle=True, num_workers=args["numworkers"],
                                            pin_memory=args["pinmemory"], 
                                            worker_init_fn = seed_worker, generator=g, drop_last=True)
        val_dataset = val_dataset_module(root=args["dataroot"], mode='val', )
        self.val_loader = torch.utils.data.DataLoader(val_dataset, args["valbatchsize"], shuffle=False, num_workers=args["numworkers"],
                                            pin_memory=args["pinmemory"])

        self.out_dir = args["outdir"]
        num = 0
        while os.path.isdir(os.path.join(self.out_dir, str(num))):
            num+=1
        self.out_dir = os.path.join(self.out_dir, str(num))
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'training_samples'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'codes'), exist_ok=True)
        shutil.copy(os.path.join('model', 'model.py'), os.path.join(self.out_dir, 'codes', 'model.py'))
        shutil.copy(os.path.join('trainer', 'train_step.py'), os.path.join(self.out_dir, 'codes', 'train_step.py'))
        shutil.copy(os.path.join('trainer', 'trainer.py'), os.path.join(self.out_dir, 'codes', 'trainer.py'))
        shutil.copy(os.path.join('config.json'), os.path.join(self.out_dir, 'codes', 'config.json'))

        
        with open(os.path.join(self.out_dir, 'args.txt'), 'w') as f:
            f.write(str(self.args))


        self.min_val_loss = 10000

        
    def train(self):
        train_losses = []
        validation_losses = []

        # validation_loss = self.validation_epoch()
        prog_bar = tqdm(range(self.epochs))
        for epoch in prog_bar:
            prog_bar.set_description(f'[Progress] epoch {epoch}/{self.epochs} - saving at {self.out_dir}')
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            self.scheduler.step()

            if epoch % self.args["validateevery"] == 0:
                torch.cuda.empty_cache()
                validation_loss = self.validation_epoch()
                torch.cuda.empty_cache()
                while len(validation_losses) < len(train_losses):
                    validation_losses.append(validation_loss)

                if self.min_val_loss > validation_loss:
                    self.min_val_loss = validation_loss
                    os.makedirs(os.path.join(self.out_dir, "best_val_results"), exist_ok=True)
                    os.system(f'cp -r {os.path.join(self.out_dir, "results")}/* {os.path.join(self.out_dir, "best_val_results")}')
                    torch.save({'f': self.trainer.module.f.state_dict() if self.args["usedataparallel"]
                                    else self.trainer.f.state_dict(),
                            'optim': self.optimizer.state_dict()},
                            os.path.join(self.out_dir, 'checkpoints', 'best_val.tar'))

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
        print(len(self.train_loader), 'num_batch')
        for batchIdx, img in enumerate(prog_bar):
            img = img.to(self.args["device"])

            self.optimizer.zero_grad()
            loss, L_package = self.trainer(img)
            # loss, L_package = self.trainer(img, return_package=True)
            loss = loss.mean()
            loss.backward()

            # nan gradient handling
            num_grad = 0
            acc_grad = 0
            for p in self.trainer.module.f.parameters():
                acc_grad += p.grad.norm()
                num_grad += 1
                p = torch.nan_to_num(p, nan=0, posinf=0, neginf=0)

            # prevent unstable training
            torch.nn.utils.clip_grad_norm_(self.trainer.module.f.parameters() if self.args["usedataparallel"]
                                            else self.trainer.f.parameters(), 1)

            self.optimizer.step()

            # information 
            num_samples += img.size(0)
            accum_losses += loss.item() * img.size(0)

            desc = f'[Train] L {loss:.4f} AC {accum_losses/num_samples:.4f}'
            for k in L_package:
                desc = desc + f' {k}:{L_package[k].mean():.4f}'
            desc = desc + f' g {acc_grad / num_grad:.4f}'
            prog_bar.set_description(desc)

            # error handling
            if acc_grad/num_grad > 5:
                print(desc)

            if batchIdx < 5:
                trainer = self.trainer.module if self.args["usedataparallel"] else self.trainer
                f = trainer.f
                T, A, clean = f(img)
                rec = clean * T + A * (1 - T)
                torchvision.utils.save_image(clean[0], os.path.join(self.out_dir, 'training_samples', f'{batchIdx}_clean.png'))
                torchvision.utils.save_image(img[0], os.path.join(self.out_dir, 'training_samples', f'{batchIdx}_img.png'))
                torchvision.utils.save_image(T[0], os.path.join(self.out_dir, 'training_samples', f'{batchIdx}_T.png'))
                torchvision.utils.save_image(A[0], os.path.join(self.out_dir, 'training_samples', f'{batchIdx}_A.png'))
                torchvision.utils.save_image(rec[0], os.path.join(self.out_dir, 'training_samples', f'{batchIdx}_reconstuct.png'))
    
    
        return accum_losses / num_samples

    @torch.no_grad()
    def validation_epoch(self):
        torch.cuda.empty_cache()
        num_samples = 0
        accum_losses = 0
        prog_bar = tqdm(self.val_loader)
        self.trainer.eval()
        f = self.trainer.module.f if self.args["usedataparallel"] else self.trainer.f
        if self.args["usedataparallel"]:
            f = torch.nn.DataParallel(f)
        f.eval()

        for batchIdx, img in enumerate(prog_bar):
            try:
                img = img.to(self.args["device"])
                loss, L_package = self.trainer(img)
                # loss, L_package = self.trainer(img, return_package=True)
                loss = loss.mean()
                num_samples += img.size(0)
                accum_losses += loss.item() * img.size(0)
                    
                T, A, clean = f(img)

                for idx in range(img.size(0)):
                    self.saver.save_image(img, T, A, clean, idx, batchIdx, self.out_dir, self.args["valbatchsize"])
       
                desc = f'[Val] L {loss:.4f} AC {accum_losses/num_samples:.4f}'
                for k in L_package:
                    desc = desc + f' {k}:{L_package[k].mean():.4f}'
                prog_bar.set_description(desc)

            except Exception as e:
                print(f'passing {batchIdx}, img {img.shape}')
                print(e)
                pass
        torch.cuda.empty_cache()

        return accum_losses / num_samples


if __name__=='__main__':
    print('Module file')