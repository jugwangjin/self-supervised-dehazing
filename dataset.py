from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
import os
import random
import torch

IMG_EXT = ['jpg', 'jpeg', 'JPEG', 'JPG', 'png', 'PNG']

MODES = ['train', 'validation']  

TRAIN_VAL_RATIO = 0.99

class RESIDEHazyDataset(torch.utils.data.Dataset):
    '''
    Dataset with only RESIDE_beta test set - internet collected unpaired hazy images
    '''
    def __init__(self, root, mode, patchSize=128):
        self.TRAIN_VAL_RATIO = 0.999
        assert os.path.isdir(root)
        assert mode in MODES

        self.patchSize = patchSize
        self.mode = mode

        dir2 = os.path.join(root, 'train', 'OTS', 'haze')

        imgset2 = [os.path.join(dir2, fn) for fn in os.listdir(dir2) if any(fn.endswith(EXT) for EXT in IMG_EXT)]
        
        imgset = imgset2
        sorted(imgset)
        numImg = len(imgset)
        splitIdx = int(numImg * self.TRAIN_VAL_RATIO)
        random.seed(20202464)
        random.shuffle(imgset)
        if mode == 'train':
            imgset = imgset[:splitIdx]
        elif mode == 'validation':
            imgset = imgset[splitIdx:]

        self.imgset = imgset

        print(f'Dataset built - mode {mode}, length {len(self.imgset)}')

        self.resize = torchvision.transforms.Resize(self.patchSize)
        if mode == 'train':
            self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.RandomCrop(self.patchSize),
                            torchvision.transforms.RandomHorizontalFlip(p=0.1),
                            torchvision.transforms.RandomVerticalFlip(p=0.1),
                            torchvision.transforms.ToTensor(),
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
            ])

    def randomRotation(self, sample):
        return torchvision.transforms.functional.rotate(sample, random.choice([90, 180, 270]))

    def __getitem__(self, index):
        img = Image.open(self.imgset[index]).convert("RGB")
        if not self.mode == 'train':
            return self.transform(img)
        
        if min(img.size) < self.patchSize:
            img = self.resize(img)
        if torch.rand(1) < 0.25:
            img = self.randomRotation(img)
        
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgset)


class RealHazyDataset(torch.utils.data.Dataset):
    '''
    Dataset with only RESIDE_beta test set - internet collected unpaired hazy images
    '''
    def __init__(self, root, mode, patchSize=64):
        assert os.path.isdir(root)
        assert mode in MODES

        self.patchSize = patchSize
        self.mode = mode

        dir1 = os.path.join(root, 'test', 'RTTS', 'JPEGImages')
        dir2 = os.path.join(root, 'test', 'UnannotatedHazyImages')

        imgset1 = [os.path.join(dir1, fn) for fn in os.listdir(dir1) if any(fn.endswith(EXT) for EXT in IMG_EXT)]
        imgset2 = [os.path.join(dir2, fn) for fn in os.listdir(dir2) if any(fn.endswith(EXT) for EXT in IMG_EXT)]
        
        imgset = imgset1 + imgset2
        sorted(imgset)
        numImg = len(imgset)
        splitIdx = int(numImg * TRAIN_VAL_RATIO)
        if mode == 'train':
            imgset = imgset[:splitIdx]
        elif mode == 'validation':
            imgset = imgset[splitIdx:]

        self.imgset = imgset

        print(f'Dataset built - mode {mode}, length {len(self.imgset)}')

        self.resize = torchvision.transforms.Resize(self.patchSize)
        if mode == 'train':
            self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.RandomCrop(self.patchSize),
                            torchvision.transforms.RandomHorizontalFlip(p=0.1),
                            torchvision.transforms.RandomVerticalFlip(p=0.1),
                            torchvision.transforms.ToTensor(),
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
            ])

    def randomRotation(self, sample):
        return torchvision.transforms.functional.rotate(sample, random.choice([90, 180, 270]))

    def __getitem__(self, index):
        img = Image.open(self.imgset[index]).convert("RGB")
        if not self.mode == 'train':
            return self.transform(img)
        
        if min(img.size) < self.patchSize:
            img = self.resize(img)
        if torch.rand(1) < 0.25:
            img = self.randomRotation(img)
        
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgset)