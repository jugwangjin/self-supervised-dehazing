from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision
import os

img_tags = ['jpg', 'jpeg', 'JPEG', 'JPG', 'png', 'PNG']

class RealHazyDataset(Dataset):
    '''
    Dataset with only RESIDE_beta test set - internet collected unpaired hazy images
    '''
    def __init__(self, root, mode):
        dir_1 = os.path.join(root, 'RTTS', 'JPEGImages')
        dir_2 = os.path.join(root, 'UnannotatedHazyImages')

        imgset_1 = [os.path.join(dir_1, fn) for fn in os.listdir(dir_1) if any(fn.endswith(img_tags))]
        imgset_2 = [os.path.join(dir_2, fn) for fn in os.listdir(dir_2) if any(fn.endswith(img_tags))]
        
        imgset = imgset_1 + imgset_2
        imgset.sorted()

        




    def __getitem__(self, index):
        fn = self.data[index]
        import numpy as np
        import cv2
        
        hazy = Image.open(fn['hazy']).convert("RGB")
        try:
            clear = Image.open(fn['clear']).convert("RGB")
        except:
            clear = Image.open(fn['clear'].replace('.jpg', '.png')).convert("RGB")
            
        # depth = Image.open(fn['depth']).convert("L") 
        if self.mode == 'train':
            i,j,h,w = transforms.RandomCrop.get_params(hazy, output_size = (512,512))
            hazy = TF.crop(hazy, i, j, h, w)
            clear = TF.crop(clear, i, j, h, w)
            # depth = TF.crop(depth, i, j, h, w)            
            #data augumentation
            if self.augment:
                hazy, clear = augment(hazy, clear)
                # hazy, clear, depth = augment(hazy, clear, depth)

        # depth = self.depth(self.transform(cv2.imread(fn['hazy'])).unsqueeze(0))
        # print(hazy.shape, depth.shape)

        # exit()
        hazy = self.transform(hazy)
        clear = self.transform(clear)

        # depth = self.transform(depth)
        return hazy, clear
        # return hazy, clear, depth 

    def __len__(self):
        return len(self.data)