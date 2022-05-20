from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
import os
import random
import torch

IMG_EXT = ['jpg', 'jpeg', 'JPEG', 'JPG', 'png', 'PNG']

MODES = ['train', 'validation']  

class RESIDEHazyDataset(torch.utils.data.Dataset):
    '''
    Dataset with only RESIDE_beta test set - internet collected unpaired hazy images
    '''
    def __init__(self, root, mode, patch_size=128):
        self.TRAIN_VAL_RATIO = 0.999
        assert os.path.isdir(root)
        assert mode in MODES

        self.patch_size = patch_size
        self.mode = mode

        dir2 = os.path.join(root, 'train', 'OTS', 'haze')

        img_set2 = [os.path.join(dir2, fn) for fn in os.listdir(dir2) if any(fn.endswith(EXT) for EXT in IMG_EXT)]
        
        img_set = img_set2
        sorted(img_set)
        num_img = len(img_set)
        split_idx = int(num_img * self.TRAIN_VAL_RATIO)
        random.seed(20202464)
        random.shuffle(img_set)
        if mode == 'train':
            img_set = img_set[:split_idx]
        elif mode == 'validation':
            img_set = img_set[split_idx:]

        self.img_set = img_set

        print(f'Dataset built - mode {mode}, length {len(self.img_set)}')

        self.resize = torchvision.transforms.Resize(self.patch_size)
        if mode == 'train':
            self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.RandomCrop(self.patch_size),
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
        img = Image.open(self.img_set[index]).convert("RGB")
        if not self.mode == 'train':
            return self.transform(img)
        
        if min(img.size) < self.patch_size:
            img = self.resize(img)
        if torch.rand(1) < 0.25:
            img = self.randomRotation(img)
        
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_set)


class RealHazyDataset(torch.utils.data.Dataset):
    '''
    Dataset with only RESIDE_beta test set - internet collected unpaired hazy images
    '''
    def __init__(self, root, mode, patch_size=128):
        assert os.path.isdir(root)
        assert mode in MODES
        self.TRAIN_VAL_RATIO = 0.99

        self.patch_size = patch_size
        self.mode = mode

        dir1 = os.path.join(root, 'test', 'RTTS', 'JPEGImages')
        dir2 = os.path.join(root, 'test', 'UnannotatedHazyImages')

        img_set1 = [os.path.join(dir1, fn) for fn in os.listdir(dir1) if any(fn.endswith(EXT) for EXT in IMG_EXT)]
        img_set2 = [os.path.join(dir2, fn) for fn in os.listdir(dir2) if any(fn.endswith(EXT) for EXT in IMG_EXT)]
        
        img_set = img_set1 + img_set2
        sorted(img_set)
        num_img = len(img_set)
        split_idx = int(num_img * self.TRAIN_VAL_RATIO)
        if mode == 'train':
            img_set = img_set[:split_idx]
        elif mode == 'validation':
            img_set = img_set[split_idx:]

        self.img_set = img_set

        print(f'Dataset built - mode {mode}, length {len(self.img_set)}')

        self.resize = torchvision.transforms.Resize(self.patch_size)
        if mode == 'train':
            self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.RandomCrop(self.patch_size),
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
        img = Image.open(self.img_set[index]).convert("RGB")
        if not self.mode == 'train':
            return self.transform(img)
        
        if min(img.size) < self.patch_size:
            img = self.resize(img)
        if torch.rand(1) < 0.25:
            img = self.randomRotation(img)
        
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_set)