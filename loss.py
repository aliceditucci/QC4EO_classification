import torch.nn as nn
import numpy as np
import torch

from torchmetrics.image import StructuralSimilarityIndexMeasure

class CustomLoss(nn.Module):
    '''
        HUBER + SSIM + latent spread loss
    '''
    def __init__(self, loss_weigths:list=None):
        super(CustomLoss, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if loss_weigths == None: self.loss_weigths = [0.33, 0.33, 0.33]
        else: self.loss_weigths = loss_weigths

        self.huber_loss = nn.HuberLoss()
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def latent_spread_loss(self, z):
        z = (z - z.mean(0)) / (z.std(0) + 1e-6)
        similarity = torch.matmul(z, z.mT)
        identity = torch.eye(z.size(0), device=z.device)
        loss = ((similarity-identity)**2).mean()
        return loss

    def forward(self, reconstruction, y):
        huber = self.huber_loss(reconstruction, y)
        ssim  = (1 - self.ssim_loss(reconstruction, y)) 
        
        return self.loss_weigths[0]*huber + self.loss_weigths[1]*ssim #+ self.ssim_loss[2]#*self.latent_spread_loss(reconstruction)