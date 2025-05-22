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
    

