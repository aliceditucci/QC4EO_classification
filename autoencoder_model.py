import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from loss import HuberSSIM

import matplotlib.pyplot as plt
    
class Autoencoder_small(pl.LightningModule):
    def __init__(self, channels, num_qubits):
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

        self.loss           = HuberSSIM(alpha=0.2, beta=0.8) #FIXME: parameters to be fixed
        self.initial_lr     = 0.001
        self.lr_scheduler_f = 3
        self.gamma          = 0.1

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
    
    def get_latent_space(self, x):
        """
        Method to get the latent space representation of the input.
        This will return the output of the encoder (the reduced-dimensionality representation).
        """
        self.eval()  # Set to evaluation mode (important if using dropout, batch norm, etc.)
        with torch.no_grad():  # Disable gradient computation (not needed for inference)
            latent = self.encoder(x)  # Return only the encoded part
        return latent

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

        x, y = batch
        reconstructions = self(x)
        loss = self.loss(reconstructions, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
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
        x, y = batch
        reconstructions = self(x)
        loss = self.loss(reconstructions, y)
        
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
        
        if ((self.current_epoch)  % 3 == 0) or self.current_epoch==0:
            log_visual_results(self.logger, self.current_epoch, batch, reconstructions, batch_idx)

        return loss

#def Network
class Autoencoder_sscnet(pl.LightningModule):
    def __init__(self, channels, num_qubits):
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

        self.loss           = HuberSSIM(alpha=0.2, beta=0.8) #FIXME: parameters to be fixed
        self.initial_lr     = 0.001
        self.lr_scheduler_f = 3
        self.gamma          = 0.1

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
    def get_latent_space(self, x):
        """
        Method to get the latent space representation of the input.
        This will return the output of the encoder (the reduced-dimensionality representation).
        """
        self.eval()  # Set to evaluation mode (important if using dropout, batch norm, etc.)
        with torch.no_grad():  # Disable gradient computation (not needed for inference)
            latent = self.encoder(x)  # Return only the encoded part
        return latent

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

        x, y = batch
        reconstructions = self(x)
        loss = self.loss(reconstructions, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
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
        x, y = batch
        reconstructions = self(x)
        loss = self.loss(reconstructions, y)
        
        self.log('valid_loss', loss, on_epoch=True, prog_bar=True)
        
        if ((self.current_epoch)  % 3 == 0) or self.current_epoch==0:
            log_visual_results(self.logger, self.current_epoch, batch, reconstructions, batch_idx)

        return loss
    

# TODO: to be moved to a utils .py file
def log_visual_results(logger, current_epoch, batch:tuple, reconstructions:torch.tensor, batch_idx:int):
    x, _ = batch
    x = x.cpu().detach().numpy()
    yp = reconstructions
    yp = yp.cpu().detach().numpy()

    x  = np.moveaxis(x,  1, -1)
    yp = np.moveaxis(yp, 1, -1)

    if x.shape[-1] > 3: plot_bands = [3,2,1]
    else: plot_bands = [0,1,2]

    bsize = x.shape[0]

    fig, axes = plt.subplots(nrows=bsize*2, ncols=3, figsize=(15, 5*bsize*2))

    for i in range(bsize):
        #### MAPS
        axes[2*i,0].imshow(x[i,plot_bands, ...])
        axes[2*i,0].axis(False)
        axes[2*i,0].set_title('Input')

        axes[2*i,1].imshow(yp[i,plot_bands, ...])
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
        for j in range(x.shape[-1]):

            color = colors[j]

            axes[2*i+1,0].hist(x[i,j,...].flatten(), density=True, bins=50, label=f"band_{j}", color=color)
            axes[2*i+1,1].hist(yp[i,j,...].flatten(), density=True, bins=50, color=color)

        axes[2*i+1,0].legend()
        
    plt.tight_layout()
    logger.experiment.add_figure(f'{batch_idx}', plt.gcf(), global_step=current_epoch)
    plt.close()