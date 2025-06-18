from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pytorch_lightning as pl
from PIL import Image
import os
import numpy as np
import torch

from pyosv.io.reader import load

class EuroSATDataset(Dataset):
    def __init__(self, root_dir, whitelist_classes, bands = [3,2,1], mode='aec'): # or class
        self.root_dir  = root_dir
        self.classes   = sorted(os.listdir(root_dir))
        self.mode      = mode
        self.bands     = bands

        self.mmin, self.mmax = [200]*len(bands), [1800]*len(bands) # FIXME: read from file 

        if self.mode not in ['aec', 'class']: raise Exception(f"Mode can be <aec> or <class>, found {self.mode}")

        self.data = []
        for class_label in self.classes:
            class_path = os.path.join(root_dir, class_label)

            if class_path.split(os.sep)[-1] in whitelist_classes:

                for img_file in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_file)
                    self.data.append((img_path, self.classes.index(class_label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.cuda.is_available(): device = torch.device('cuda')
        img_path, label = self.data[idx]
        b_in, _, _ = load(img_path)
        b_in = b_in.astype(np.float32) 
        b_in = b_in[..., self.bands]

        # Normalization
        b_in = minmaxscaler(b_in, mmin=self.mmin, mmax=self.mmax)
        b_in = np.moveaxis(b_in, -1, 0)

        b_in = torch.tensor(b_in, dtype=torch.float32, device=device)

        if self.mode == 'aec': b_ou = b_in.clone()
        if self.mode == 'class': b_ou = torch.tensor(label, dtype=torch.long, device=device) 
       
        return b_in, b_ou, img_path

class EuroSATDataModule(pl.LightningDataModule):
    def __init__(self, whitelist_classes, batch_size=64, bands=[3,2,1], mode='aec', num_workers=9):
        super().__init__()
        self.whitelist_classes = whitelist_classes
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.bands       = bands
        self.mode        = mode

    def setup(self, stage=None):
        transform          = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = EuroSATDataset(
            root_dir = os.path.join('EuroSAT-split', 'train'), 
            whitelist_classes = self.whitelist_classes, 
            bands=self.bands, 
            mode=self.mode
        )

        self.valid_dataset = EuroSATDataset(
            root_dir = os.path.join('EuroSAT-split', 'val'),  
             whitelist_classes = self.whitelist_classes, 
             bands=self.bands, 
             mode=self.mode
        )

        self.test_dataset = EuroSATDataset(
            root_dir = os.path.join('EuroSAT-split', 'test'),  
             whitelist_classes = self.whitelist_classes, 
             bands=self.bands, 
             mode=self.mode
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,  num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,  num_workers=self.num_workers, persistent_workers=True)
    

def minmaxscaler(img : np.ndarray, mmin : list = None, mmax : list = None) -> np.ndarray: 
    '''
        Apply the min max scaler to the input img:  '''
    

    if len(img.shape) != 3:
        raise Exception('Error: lenght of image shape must be 3 - (space, space, channels)')
    
    if mmin is not None:
        if len(mmin) != img.shape[-1]:
            raise Exception('Error: lenght of mmin must be equals to number of bands')
    
    if mmax is not None:
        if len(mmax) != img.shape[-1]:
            raise Exception('Error: lenght of mmax must be equals to number of bands')
    
    if mmin == None: raise Exception('Error: not normalizing over the dataset')
    if mmax == None: raise Exception('Error: not normalizing over the dataset')

    for i in range(img.shape[-1]): 

        img[:,:,i] = np.clip(img[:,:,i], mmin[i], mmax[i])

        num = img[:,:,i] - mmin[i]
        den = mmax[i] - mmin[i]
        
        img[:, :, i] = np.divide(num, den, out=np.zeros_like(num)+0.5, where=den != 0)

    return img