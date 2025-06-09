import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

    
#def Network
class Autoencoder_small(nn.Module):
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
    
#def Network
class Autoencoder_sscnet(nn.Module):
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
    
#def Network
class UNet(nn.Module):
    def __init__(self, channels, num_qubits):
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
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, channels, kernel_size=1)
        self.final_act = nn.Sigmoid()  # o Softmax per multi-classe


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
    
    
    def get_latent_space(self, x):
        """
        Method to get the latent space representation of the input.
        This will return the output of the encoder (the reduced-dimensionality representation).
        """
        self.eval()  # Set to evaluation mode (important if using dropout, batch norm, etc.)
        with torch.no_grad():  # Disable gradient computation (not needed for inference)
            latent = self.encoder(x)  # Return only the encoded part
        return latent