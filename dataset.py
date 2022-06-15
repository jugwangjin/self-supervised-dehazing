from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
import os
import random
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from math import ceil

IMG_EXT = ['jpg', 'jpeg', 'JPEG', 'JPG', 'png', 'PNG']

MODES = ['train', 'val']  

class RESIDEHazyDataset(torch.utils.data.Dataset):
    '''
    Dataset with only RESIDE_beta test set - internet collected unpaired hazy images
    '''
    def __init__(self, root, mode, patch_size=128, ratio=1):
        self.TRAIN_VAL_RATIO = 0.999
        assert os.path.isdir(root)
        assert mode in MODES

        self.patch_size = patch_size
        self.mode = mode

        dir2 = os.path.join(root, 'train', 'OTS', 'haze')

        img_set2 = [os.path.join(dir2, fn) for fn in os.listdir(dir2) if any(fn.endswith(EXT) for EXT in IMG_EXT)]
        
        img_set = img_set2
        img_set = sorted(img_set)
        num_img = len(img_set)
        split_idx = int(num_img * self.TRAIN_VAL_RATIO)
        random.seed(20202464)
        random.shuffle(img_set)
        if mode == 'train':
            img_set = img_set[:split_idx]
            if ratio < 1:
                img_set = img_set[:ceil(len(img_set) * ratio)]
        elif mode == 'val':
            img_set = img_set[split_idx:]

        self.img_set = img_set

        print(f'Dataset built - mode {mode}, length {len(self.img_set)}')

        self.resize = torchvision.transforms.Resize(self.patch_size)
        if mode == 'train':
            self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.RandomCrop(self.patch_size),
                            torchvision.transforms.RandomHorizontalFlip(p=0.4),
                            # torchvision.transforms.RandomVerticalFlip(p=0.1),
                            torchvision.transforms.ToTensor(),
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        img = Image.open(self.img_set[index]).convert("RGB")
        if not self.mode == 'train':
            return self.transform(img)
        
        if min(img.size) < self.patch_size:
            img = self.resize(img)
        
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
        img_set = sorted(img_set)
        num_img = len(img_set)
        split_idx = int(num_img * self.TRAIN_VAL_RATIO)
        if mode == 'train':
            img_set = img_set[:split_idx]
        elif mode == 'val':
            img_set = img_set[split_idx:]

        self.img_set = img_set

        print(f'Dataset built - mode {mode}, length {len(self.img_set)}')

        self.resize = torchvision.transforms.Resize(self.patch_size)
        if mode == 'train':
            self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.RandomCrop(self.patch_size),
                            torchvision.transforms.RandomHorizontalFlip(p=0.4),
                            # torchvision.transforms.RandomVerticalFlip(p=0.1),
                            torchvision.transforms.ToTensor(),
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        img = Image.open(self.img_set[index]).convert("RGB")
        if not self.mode == 'train':
            return self.transform(img)
        
        if min(img.size) < self.patch_size:
            img = self.resize(img)
        
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_set)


class RESIDEStandardDataset(torch.utils.data.Dataset):
    '''
    Dataset with only RESIDE_beta test set - internet collected unpaired hazy images
    '''
    def __init__(self, root, mode='train', patch_size=128):
        assert os.path.isdir(root)

        self.patch_size = patch_size
        if mode == 'train':
            dir2 = os.path.join(root, 'train', 'hazy')
            img_set2 = [os.path.join(dir2, fn) for fn in os.listdir(dir2) if any(fn.endswith(EXT) for EXT in IMG_EXT)]
        elif mode == 'val':
            # use subset of RESIDE-standard: they have 5000 test image - 10 hazy images per clear image
            dir2 = os.path.join(root, 'test', 'SOTS', 'indoor', 'hazy')
            img_set2 = [os.path.join(dir2, fn) for fn in os.listdir(dir2) if any(fn.endswith(EXT) for EXT in IMG_EXT)]
            img_set2 = [fn for fn in img_set2 if fn.endswith('_6.png') or fn.endswith('_10.png')]
        
        img_set = img_set2
        img_set = sorted(img_set)

        self.img_set = img_set

        print(f'Dataset built -  length {len(self.img_set)}')

        self.resize = torchvision.transforms.Resize(self.patch_size)
        if mode == 'train':
            self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.RandomCrop(self.patch_size),
                                torchvision.transforms.RandomHorizontalFlip(p=0.4),
                                torchvision.transforms.ToTensor(),
                ])
        elif mode == 'val':
            self.transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                ])

    def __getitem__(self, index):
        img = Image.open(self.img_set[index]).convert("RGB")
        
        if min(img.size) < self.patch_size:
            img = self.resize(img)
            
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_set)


class RESIDEStandardPairedDataset(torch.utils.data.Dataset):
    '''
    Dataset with only RESIDE_beta test set - internet collected unpaired hazy images
    '''
    def __init__(self, root, mode='val', ratio = 0.1, patch_size=128):
        assert os.path.isdir(root)
        self.patch_size = patch_size

        dir2 = os.path.join(root, mode, 'hazy')

        img_set2 = [os.path.join(dir2, fn) for fn in os.listdir(dir2) if any(fn.endswith(EXT) for EXT in IMG_EXT)]
        
        img_set = img_set2
        img_set = sorted(img_set)
        num_img = len(img_set)
        split_idx = int(num_img * 0.1) # 200 out of 2000 samples
        random.seed(20202464)
        random.shuffle(img_set)
        img_set = img_set[:split_idx]

        self.img_set = img_set

        print(f'Dataset built - length {len(self.img_set)}')

        self.resize = torchvision.transforms.Resize(self.patch_size)
        self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        img = Image.open(self.img_set[index]).convert("RGB")
        clear_img_name = self.img_set[index].split('/')[:-1] + [self.img_set[index].split('/')[-1].split("_")[0]+".png"]
        clear_img_name = "/"+os.path.join(*clear_img_name)
        clear_img_name = clear_img_name.replace("hazy", "clear")
        clear_img = Image.open(clear_img_name).convert("RGB")
        
        if min(img.size) < self.patch_size:
            img = self.resize(img)
            clear_img = self.resize(clear_img)
    
        i,j,h,w = transforms.RandomCrop.get_params(img, output_size = (self.patch_size, self.patch_size))
        img = TF.crop(img, i, j, h, w)
        clear_img = TF.crop(clear_img, i, j, h, w)

        if torch.rand(1) < 0.4:
            img = TF.hflip(img)
            clear_img = TF.hflip(clear_img)

        img = self.transform(img)
        clear_img = self.transform(clear_img)
        return img, clear_img

    def __len__(self):
        return len(self.img_set)



class RESIDEStandardTestDataset(torch.utils.data.Dataset):
    '''
    Dataset with only RESIDE_beta test set - internet collected unpaired hazy images
    '''
    def __init__(self, root, mode='val', ratio = 0.1, patch_size=128):
        assert os.path.isdir(root)
        self.patch_size = patch_size

        dir2 = os.path.join(root, 'test', 'SOTS', 'indoor', 'hazy')

        img_set2 = [os.path.join(dir2, fn) for fn in os.listdir(dir2) if any(fn.endswith(EXT) for EXT in IMG_EXT)]
        
        img_set = img_set2
        img_set = sorted(img_set)
        num_img = len(img_set)

        self.img_set = img_set

        print(f'Dataset built - length {len(self.img_set)}')

        self.transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        img = Image.open(self.img_set[index]).convert("RGB")
        clear_img_name = self.img_set[index].split('/')[:-2] + ['gt'] + [self.img_set[index].split('/')[-1].split("_")[0]+".png"]
        clear_img_name = "/"+os.path.join(*clear_img_name)
        clear_img = Image.open(clear_img_name).convert("RGB")

        img = self.transform(img)
        clear_img = self.transform(clear_img)
        return img, clear_img, self.img_set[index]

    def __len__(self):
        return len(self.img_set)



class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode='val', ratio = 0.1, patch_size=128):
        reside_standart_dataset = RESIDEStandardDataset(root = os.path.join(root, "RESIDE_standard"), mode=mode, patch_size=patch_size)
        reside_hazy_dataset = RESIDEHazyDataset(root = os.path.join(root, "RESIDE_beta"), mode=mode, patch_size=patch_size, ratio=0.25)
        real_hazy_dataset = RealHazyDataset(root = os.path.join(root, "RESIDE_beta"), mode=mode, patch_size=patch_size)

        self.dataset = torch.utils.data.ConcatDataset([reside_standart_dataset, reside_hazy_dataset, real_hazy_dataset])
        print(f'Dataset built - length {len(self.dataset)}, from res_std {len(reside_standart_dataset)}, res_outd {len(reside_hazy_dataset)}, real_hazy {len(real_hazy_dataset)}')

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)