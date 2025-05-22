import torch.nn as nn
import numpy as np
import torch

from torchmetrics.image import StructuralSimilarityIndexMeasure

class HuberSSIM(nn.Module):
    '''
        HUBER + SSIM
    '''
    def __init__(self, alpha, beta):
        super(HuberSSIM, self).__init__()
        self.alpha  = alpha
        self.beta   = beta
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        self.huber_loss = nn.HuberLoss()
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def forward(self, reconstruction, y):
        huber = self.huber_loss(reconstruction, y)
        ssim  = (1 - self.ssim_loss(reconstruction, y)) 
        
        return self.alpha*huber + self.beta*ssim