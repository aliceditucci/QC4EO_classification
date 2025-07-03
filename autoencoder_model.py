import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from loss import CustomLoss

import matplotlib.pyplot as plt

from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError


class Autoencoder_small(pl.LightningModule):
    def __init__(self, channels, num_qubits, initial_lr=0.002, loss_weigths=[0.2, 0.8, 0.5], gamma=0.1, lr_scheduler_f=3):
        super().__init__()

        self.num_qubits = num_qubits

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3 , padding =1),  #[B, C, 64, 64] -> [B, 16, 64, 64] (padding = same)
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #[B, 16, 64, 64] -> [B, 16, 32, 32]

            nn.Conv2d(16, 32, kernel_size=3, padding = 1), #[B, 16, 32, 32] ->  [B, 32, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #[B, 32, 32, 32] -> [B, 32, 16, 16]

            nn.Conv2d(32, 32, kernel_size=3, padding = 1), # [B, 32, 16, 16] -> [B, 32, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #  [B, 32, 16, 16] -> [B, 32, 8, 8]
        
            nn.Flatten(), #  [B, 2048] invece di [B, 2304]
            nn.Linear(2048, num_qubits)  # Bottleneck
        )

        self.decoder = nn.Sequential(
            nn.Linear(num_qubits, 2048),  # Fully connected layer to bring back 2048 features
            nn.ReLU(),
            nn.Unflatten(1, (32, 8, 8)),  # Reshape back to [B, 32, 8, 8]

            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2), # Upsample  [B, 32, 8, 8] -> [B, 32, 16, 16]
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), # Upsample  [B, 32, 16, 16] -> [B, 16, 32, 32]
            nn.ReLU(),

            nn.ConvTranspose2d(16, channels, kernel_size=2, stride=2),  # Upsample  [B, 16, 32, 32] -> [B, C, 64, 64]
            nn.Sigmoid()  # To output values in the range [0, 1] if image data is normalized
        )

        self.loss           = CustomLoss(loss_weigths=loss_weigths)
        self.initial_lr     = initial_lr
        self.lr_scheduler_f = lr_scheduler_f
        self.gamma          = gamma

        self.rmse_metric    = MeanSquaredError(squared=False)
        self.ssim_metric    = StructuralSimilarityIndexMeasure()
        self.psnr_metric    = PeakSignalNoiseRatio() 

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def configure_optimizers(self) -> torch.optim:
        '''
        Prepares the torch optimizer.
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

        return {"optimizer": optimizer, 'lr_scheduler': {"scheduler": scheduler, "interval": "epoch", "frequency": self.lr_scheduler_f}}
    
    def training_step(self, batch:tuple, batch_idx:int) -> torch.tensor:
        '''
        Overriding torch lighrning training step.
        It runs the training process on a batch of data.

        Parameters:
        -----------
        - batch (tuple of torch.tensor): tuple containing the input and ground truth torch tensors
        - batch_idx (int): integer representing an increasing index for the batches in training dataset
        
        Returns:
        --------
        - loss (torch.tensor): loss value for the batch
        '''

        x, y, path = batch
        reconstructions = self(x)
        loss = self.loss(reconstructions, y)
        rmse, ssim, psnr = self.compute_metrics(reconstructions, y)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_rmse', rmse, on_epoch=True, prog_bar=True, metric_attribute='rmse_metric')
        self.log('train_ssim', ssim, on_epoch=True, prog_bar=True, metric_attribute='ssim_metric')
        self.log('train_psnr', psnr, on_epoch=True, prog_bar=True, metric_attribute='psrn_metric')
        return loss

    def validation_step(self, batch:tuple, batch_idx:int) -> torch.tensor:
        '''
        Overriding torch lighrning validation step.
        It runs the validation process on a batch of data.

        Parameters:
        -----------
        - batch (tuple of torch.tensor): tuple containing the input and ground truth torch tensors
        - batch_idx (int): integer representing an increasing index for the batches in validation dataset
        
        Returns:
        --------
        - loss (torch.tensor): loss value for the batch
        '''
        x, y, path = batch
        reconstructions = self(x)
        loss = self.loss(reconstructions, y)
        
        rmse, ssim, psnr = self.compute_metrics(reconstructions, y)

        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
        self.log('valid_rmse', rmse, on_epoch=True, prog_bar=True, metric_attribute='rmse_metric')
        self.log('valid_ssim', ssim, on_epoch=True, prog_bar=True, metric_attribute='ssim_metric')
        self.log('valid_psnr', psnr, on_epoch=True, prog_bar=True, metric_attribute='psrn_metric')
        
        if ((self.current_epoch)  % 3 == 0) or self.current_epoch==0:
            log_visual_results(self.logger, self.current_epoch, batch, reconstructions, batch_idx)

        return loss

    def test_step(self, batch:tuple, batch_idx:int) -> torch.tensor:
        '''
        Overriding torch lighrning test step.
        It runs the validation process on a batch of data.

        Parameters:
        -----------
        - batch (tuple of torch.tensor): tuple containing the input and ground truth torch tensors
        - batch_idx (int): integer representing an increasing index for the batches in validation dataset
        
        Returns:
        --------
        - loss (torch.tensor): loss value for the batch
        '''
        x, y, path = batch
        reconstructions = self(x)
        loss = self.loss(reconstructions, y)
        
        rmse, ssim, psnr = self.compute_metrics(reconstructions, y)

        self.log('test_loss',  loss, on_epoch=True, prog_bar=True)
        self.log('test_rmse', rmse, on_epoch=True, prog_bar=True, metric_attribute='rmse_metric')
        self.log('test_ssim', ssim, on_epoch=True, prog_bar=True, metric_attribute='ssim_metric')
        self.log('test_psnr', psnr, on_epoch=True, prog_bar=True, metric_attribute='psrn_metric')
        
        if ((self.current_epoch)  % 3 == 0) or self.current_epoch==0:
            log_visual_results(self.logger, self.current_epoch, batch, reconstructions, batch_idx)

        return loss

    def compute_metrics(self, predictions, targets):
        rmse, ssim, psnr = [], [], []

        for b in range(predictions.shape[1]):
            rmse.append(self.rmse_metric(predictions, targets))
            ssim.append(self.ssim_metric(predictions, targets))
            psnr.append(self.psnr_metric(predictions, targets))
        
        rmse = torch.mean(torch.tensor(rmse))
        ssim = torch.mean(torch.tensor(ssim))
        psnr = torch.mean(torch.tensor(psnr))

        return rmse, ssim, psnr

class Autoencoder_sscnet(pl.LightningModule):
    def __init__(self, channels, num_qubits, initial_lr=0.002, loss_weigths=[0.2, 0.8, 0.5], gamma=0.1, lr_scheduler_f=3):
        super().__init__()

        self.num_qubits = num_qubits

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3 , padding =1),  #[B, C, 64, 64] -> [B, 64, 64, 64] (padding = same)
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding = 1), #[B, 64, 64, 64] ->  [B, 128, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #[B, 128, 64, 64] ->  [B, 128, 32, 32]

            nn.Conv2d(128, 256, kernel_size=3, padding = 1), #  [B, 128, 32, 32] ->  [B, 256, 32, 32]
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # [B, 256, 32, 32] -> [B, 256, 16, 16]

            nn.Conv2d(256, 256, kernel_size=3, padding = 1), #  [B, 256, 16, 16] ->  [B, 256, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #[B, 256, 16, 16] -> [B, 256, 8, 8]
        
            nn.Flatten(), #  [B, 256*8*8] invece di [B, 2304]
            nn.Linear(16384, num_qubits)  # Bottleneck
        )

        self.decoder = nn.Sequential(
            nn.Linear(num_qubits, 16384),  # Fully connected layer to bring back 2048 features
            nn.ReLU(),
            nn.Unflatten(1, (256, 8, 8)),  # Reshape back to [B, 32, 8, 8]

            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2), # Upsample  [B, 32, 8, 8] -> [B, 32, 16, 16]
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), # Upsample  [B, 32, 16, 16] -> [B, 16, 32, 32]
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), # Upsample  [B, 32, 16, 16] -> [B, 16, 32, 32]
            nn.ReLU(),

            nn.ConvTranspose2d(64, channels, kernel_size=3, stride=1, padding=1),  # Upsample  [B, 16, 32, 32] -> [B, C, 64, 64]
            nn.Sigmoid()  # To output values in the range [0, 1] if image data is normalized
        )

        self.loss           = CustomLoss(loss_weigths=loss_weigths)
        self.initial_lr     = initial_lr
        self.lr_scheduler_f = lr_scheduler_f
        self.gamma          = gamma
        self.rmse_metric    = MeanSquaredError(squared=False)
        self.ssim_metric    = StructuralSimilarityIndexMeasure()
        self.psnr_metric    = PeakSignalNoiseRatio() 

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def configure_optimizers(self) -> torch.optim:
        '''
        Prepares the torch optimizer.
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

        return {"optimizer": optimizer, 'lr_scheduler': {"scheduler": scheduler, "interval": "epoch", "frequency": self.lr_scheduler_f}}
    
    def training_step(self, batch:tuple, batch_idx:int) -> torch.tensor:
        '''
        Overriding torch lighrning training step.
        It runs the training process on a batch of data.

        Parameters:
        -----------
        - batch (tuple of torch.tensor): tuple containing the input and ground truth torch tensors
        - batch_idx (int): integer representing an increasing index for the batches in training dataset
        
        Returns:
        --------
        - loss (torch.tensor): loss value for the batch
        '''

        x, y, path = batch
        reconstructions = self(x)
        loss = self.loss(reconstructions, y)
        rmse, ssim, psnr = self.compute_metrics(reconstructions, y)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_rmse', rmse, on_epoch=True, prog_bar=True, metric_attribute='rmse_metric')
        self.log('train_ssim', ssim, on_epoch=True, prog_bar=True, metric_attribute='ssim_metric')
        self.log('train_psnr', psnr, on_epoch=True, prog_bar=True, metric_attribute='psrn_metric')
        return loss

    def validation_step(self, batch:tuple, batch_idx:int) -> torch.tensor:
        '''
        Overriding torch lighrning validation step.
        It runs the validation process on a batch of data.

        Parameters:
        -----------
        - batch (tuple of torch.tensor): tuple containing the input and ground truth torch tensors
        - batch_idx (int): integer representing an increasing index for the batches in validation dataset
        
        Returns:
        --------
        - loss (torch.tensor): loss value for the batch
        '''
        x, y, path = batch
        reconstructions = self(x)
        loss = self.loss(reconstructions, y)
        
        rmse, ssim, psnr = self.compute_metrics(reconstructions, y)

        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
        self.log('valid_rmse', rmse, on_epoch=True, prog_bar=True, metric_attribute='rmse_metric')
        self.log('valid_ssim', ssim, on_epoch=True, prog_bar=True, metric_attribute='ssim_metric')
        self.log('valid_psnr', psnr, on_epoch=True, prog_bar=True, metric_attribute='psrn_metric')
        
        if ((self.current_epoch)  % 3 == 0) or self.current_epoch==0:
            log_visual_results(self.logger, self.current_epoch, batch, reconstructions, batch_idx)

        return loss

    def test_step(self, batch:tuple, batch_idx:int) -> torch.tensor:
        '''
        Overriding torch lighrning test step.
        It runs the validation process on a batch of data.

        Parameters:
        -----------
        - batch (tuple of torch.tensor): tuple containing the input and ground truth torch tensors
        - batch_idx (int): integer representing an increasing index for the batches in validation dataset
        
        Returns:
        --------
        - loss (torch.tensor): loss value for the batch
        '''
        x, y, path = batch
        reconstructions = self(x)
        loss = self.loss(reconstructions, y)
        
        rmse, ssim, psnr = self.compute_metrics(reconstructions, y)

        self.log('test_loss',  loss, on_epoch=True, prog_bar=True)
        self.log('test_rmse', rmse, on_epoch=True, prog_bar=True, metric_attribute='rmse_metric')
        self.log('test_ssim', ssim, on_epoch=True, prog_bar=True, metric_attribute='ssim_metric')
        self.log('test_psnr', psnr, on_epoch=True, prog_bar=True, metric_attribute='psrn_metric')
        
        if ((self.current_epoch)  % 3 == 0) or self.current_epoch==0:
            log_visual_results(self.logger, self.current_epoch, batch, reconstructions, batch_idx)

        return loss

    def compute_metrics(self, predictions, targets):
        rmse, ssim, psnr = [], [], []

        for b in range(predictions.shape[1]):
            rmse.append(self.rmse_metric(predictions, targets))
            ssim.append(self.ssim_metric(predictions, targets))
            psnr.append(self.psnr_metric(predictions, targets))
        
        rmse = torch.mean(torch.tensor(rmse))
        ssim = torch.mean(torch.tensor(ssim))
        psnr = torch.mean(torch.tensor(psnr))

        return rmse, ssim, psnr

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.block(x)
 
class UNet(pl.LightningModule):
    def __init__(self, channels, num_qubits, initial_lr=0.002, loss_weigths=[0.2, 0.8, 0.5], gamma=0.1, lr_scheduler_f=3):
        super().__init__()

        self.num_qubits = num_qubits
 
        # Encoder
        self.encoder = nn.Sequential(
            DoubleConv(channels, 64),       # x1
            nn.MaxPool2d(2),                # down1
            DoubleConv(64, 128),            # x2
            nn.MaxPool2d(2),                # down2
            DoubleConv(128, 256),           # x3
            nn.MaxPool2d(2),                # down3
            DoubleConv(256, 512),           # x4
            nn.MaxPool2d(2),                # down4
            DoubleConv(512, 1024),          # bottleneck
            nn.Flatten(),
            nn.Linear(4*4*1024,num_qubits)
        )
        
        self.post_quantum = nn.Sequential(
            nn.Linear(num_qubits, 4*4*1024),
            nn.ReLU(),
            nn.Unflatten(1, (1024, 4, 4))
        )

        # Decoder (con upsampling + concatenazione skip)
        self.up1       = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1      = DoubleConv(1024, 512)
        self.up2       = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2      = DoubleConv(512, 256)
        self.up3       = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3      = DoubleConv(256, 128)
        self.up4       = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4      = DoubleConv(128, 64)
        self.out_conv  = nn.Conv2d(64, channels, kernel_size=1)
        self.final_act = nn.Sigmoid()  # o Softmax per multi-classe

        self.loss           = CustomLoss(loss_weigths=loss_weigths) 
        self.initial_lr     = initial_lr
        self.lr_scheduler_f = lr_scheduler_f
        self.gamma          = gamma
        self.rmse_metric    = MeanSquaredError(squared=False)
        self.ssim_metric    = StructuralSimilarityIndexMeasure()
        self.psnr_metric    = PeakSignalNoiseRatio() 
 
    def forward(self, x):
        # Encoder (manuale per salvare i punti skip)
        x1 = self.encoder[0](x)          # 64
        x2 = self.encoder[2](self.encoder[1](x1))  # 128
        x3 = self.encoder[4](self.encoder[3](x2))  # 256
        x4 = self.encoder[6](self.encoder[5](x3))  # 512
        x5 = self.encoder[8](self.encoder[7](x4))  # bottleneck
        x5 = self.encoder[9](x5)
        x5 = self.encoder[10](x5)
        x5 = self.post_quantum(x5)
        # Decoder con skip connections
        x = self.up1(x5)
        x = self.dec1(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.dec4(torch.cat([x, x1], dim=1))
        return self.final_act(self.out_conv(x))

    def configure_optimizers(self) -> torch.optim:
        '''
        Prepares the torch optimizer.
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)

        return {"optimizer": optimizer, 'lr_scheduler': {"scheduler": scheduler, "interval": "epoch", "frequency": self.lr_scheduler_f}}
    
    def training_step(self, batch:tuple, batch_idx:int) -> torch.tensor:
        '''
        Overriding torch lighrning training step.
        It runs the training process on a batch of data.

        Parameters:
        -----------
        - batch (tuple of torch.tensor): tuple containing the input and ground truth torch tensors
        - batch_idx (int): integer representing an increasing index for the batches in training dataset
        
        Returns:
        --------
        - loss (torch.tensor): loss value for the batch
        '''

        x, y, path = batch
        reconstructions = self(x)
        loss = self.loss(reconstructions, y)
        rmse, ssim, psnr = self.compute_metrics(reconstructions, y)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        self.log('train_rmse', rmse, on_epoch=True, prog_bar=True, metric_attribute='rmse_metric')
        self.log('train_ssim', ssim, on_epoch=True, prog_bar=True, metric_attribute='ssim_metric')
        self.log('train_psnr', psnr, on_epoch=True, prog_bar=True, metric_attribute='psrn_metric')
        return loss

    def validation_step(self, batch:tuple, batch_idx:int) -> torch.tensor:
        '''
        Overriding torch lighrning validation step.
        It runs the validation process on a batch of data.

        Parameters:
        -----------
        - batch (tuple of torch.tensor): tuple containing the input and ground truth torch tensors
        - batch_idx (int): integer representing an increasing index for the batches in validation dataset
        
        Returns:
        --------
        - loss (torch.tensor): loss value for the batch
        '''
        x, y, path = batch
        reconstructions = self(x)
        loss = self.loss(reconstructions, y)
        
        rmse, ssim, psnr = self.compute_metrics(reconstructions, y)

        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
        self.log('valid_rmse', rmse, on_epoch=True, prog_bar=True, metric_attribute='rmse_metric')
        self.log('valid_ssim', ssim, on_epoch=True, prog_bar=True, metric_attribute='ssim_metric')
        self.log('valid_psnr', psnr, on_epoch=True, prog_bar=True, metric_attribute='psrn_metric')
        
        if ((self.current_epoch)  % 3 == 0) or self.current_epoch==0:
            log_visual_results(self.logger, self.current_epoch, batch, reconstructions, batch_idx)

        return loss

    def test_step(self, batch:tuple, batch_idx:int) -> torch.tensor:
        '''
        Overriding torch lighrning test step.
        It runs the validation process on a batch of data.

        Parameters:
        -----------
        - batch (tuple of torch.tensor): tuple containing the input and ground truth torch tensors
        - batch_idx (int): integer representing an increasing index for the batches in validation dataset
        
        Returns:
        --------
        - loss (torch.tensor): loss value for the batch
        '''
        x, y, path = batch
        reconstructions = self(x)
        loss = self.loss(reconstructions, y)
        
        rmse, ssim, psnr = self.compute_metrics(reconstructions, y)

        self.log('test_loss',  loss, on_epoch=True, prog_bar=True)
        self.log('test_rmse', rmse, on_epoch=True, prog_bar=True, metric_attribute='rmse_metric')
        self.log('test_ssim', ssim, on_epoch=True, prog_bar=True, metric_attribute='ssim_metric')
        self.log('test_psnr', psnr, on_epoch=True, prog_bar=True, metric_attribute='psrn_metric')
        
        if ((self.current_epoch)  % 3 == 0) or self.current_epoch==0:
            log_visual_results(self.logger, self.current_epoch, batch, reconstructions, batch_idx)

        return loss

    def compute_metrics(self, predictions, targets):
        rmse, ssim, psnr = [], [], []

        for b in range(predictions.shape[1]):
            rmse.append(self.rmse_metric(predictions, targets))
            ssim.append(self.ssim_metric(predictions, targets))
            psnr.append(self.psnr_metric(predictions, targets))
        
        rmse = torch.mean(torch.tensor(rmse))
        ssim = torch.mean(torch.tensor(ssim))
        psnr = torch.mean(torch.tensor(psnr))

        return rmse, ssim, psnr

# TODO: to be moved to a utils .py file

def log_visual_results(logger, current_epoch, batch:tuple, reconstructions:torch.tensor, batch_idx:int):
    x, _, _ = batch # bs, c, w, h
    x = x.cpu().detach().numpy()
    yp = reconstructions
    yp = yp.cpu().detach().numpy()

    if x.shape[1] > 3: plot_bands = [3,2,1]
    else: plot_bands = [0,1,2]

    bsize = x.shape[0]//4

    fig, axes = plt.subplots(nrows=bsize*2, ncols=2, figsize=(15, 5*bsize*2))

    for i in range(bsize):
        xi = np.moveaxis(x[i,plot_bands,...], 0, -1) # bs, c, w, h
        yi = np.moveaxis(yp[i,plot_bands, ...], 0, -1)
    
        #### MAPS
        axes[2*i,0].imshow(xi)
        axes[2*i,0].axis(False)
        axes[2*i,0].set_title('Input')

        axes[2*i,1].imshow(yi)
        axes[2*i,1].axis(False)
        axes[2*i,1].set_title(f'Recontruction')
        colors = [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#7f7f7f",  # gray
            "#bcbd22",  # yellow-green
            "#17becf",  # cyan
            "#393b79",  # dark blue
            "#637939",  # olive green
            "#8c6d31",  # ochre
            "#843c39",  # dark red
            "#7b4173",  # violet
            "#5254a3",  # steel blue
            "#9c9ede",  # light purple
            "#cedb9c",  # light green
            "#e7ba52",  # mustard
            "#bd9e39"   # golden brown
        ]

        #### HIST
        for j in range(xi.shape[-1]):

            color = colors[j]

            axes[2*i+1,0].hist(xi[..., j].flatten(), density=True, bins=50, label=f"band_{j}", color=color)
            axes[2*i+1,1].hist(yi[..., j].flatten(), density=True, bins=50, color=color)

        axes[2*i+1,0].legend()
        
    plt.tight_layout()
    logger.experiment.add_figure(f'{batch_idx}', plt.gcf(), global_step=current_epoch)
    plt.close()