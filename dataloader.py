from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
import os

IMG_EXT = ['jpg', 'jpeg', 'JPEG', 'JPG', 'png', 'PNG']

MODES = ['train', 'validation']  

TRAIN_VAL_RATIO = 0.99

class RealHazyDataset(Dataset):
    '''
    Dataset with only RESIDE_beta test set - internet collected unpaired hazy images
    '''
    def __init__(self, root, mode):
        assert os.isdir(root)
        assert mode in MODES

        dir1 = os.path.join(root, 'RTTS', 'JPEGImages')
        dir2 = os.path.join(root, 'UnannotatedHazyImages')

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

        self.resize = torchvision.transforms.Resize(256)
        self.transform = torchvision.Compose([
                        torchvision.RandomCrop(256),
                        torchvision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                        torchvision.ToTensor(),
        ])


    def __getitem__(self, index):
        img = Image.Open(self.imgset[index]).conver("RGB")
        if min(img.size) < 256:
            img = self.resize(img)
        img_tensor = self.transform(img)
        return img_tensor

    def __len__(self):
        return len(self.imgset)