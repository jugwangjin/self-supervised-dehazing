from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
import os
import random

IMG_EXT = ['jpg', 'jpeg', 'JPEG', 'JPG', 'png', 'PNG']

MODES = ['train', 'validation']  

TRAIN_VAL_RATIO = 0.99

class RealHazyDataset(Dataset):
    '''
    Dataset with only RESIDE_beta test set - internet collected unpaired hazy images
    '''
    def __init__(self, root, mode, patchSize=64):
        assert os.isdir(root)
        assert mode in MODES

        self.patchSize = patchSize
        self.mode = mode

        dir1 = os.path.join(root, 'test', 'RTTS', 'JPEGImages')
        dir2 = os.path.join(root, 'test', 'UnannotatedHazyImages')

        imgset1 = [os.path.join(dir1, fn) for fn in os.listdir(dir1) if any(fn.endswith(EXT) for EXT in IMG_EXT)]
        imgset2 = [os.path.join(dir2, fn) for fn in os.listdir(dir2) if any(fn.endswith(EXT) for EXT in IMG_EXT)]
        
        imgset = imgset1 + imgset2
        imgset.sorted()
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
            self.transform = torchvision.Compose([
                            torchvision.RandomCrop(self.patchSize),
                            torchvision.RandomHorizontalFlip(p=0.1),
                            torchvision.RandomVerticalFlip(p=0.1),
                            torchvision.ToTensor(),
            ])
        else:
            self.transform = torchvision.Compose([
                            torchvision.ToTensor(),
            ])

    def randomRotation(self, sample):
        return torchvision.transforms.functional.rotate(sample, random.choice([90, 180, 270]))

    def __getitem__(self, index):
        img = Image.Open(self.imgset[index]).conver("RGB")
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