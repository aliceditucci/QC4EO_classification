from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import numpy as np
import argparse
import torch
import sys
import cv2
import os


from autoencoder_model import *
from EuroSAT_torch_loader import EuroSATDataModule

if __name__ == "__main__":

    #region input arguments
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="Number of qubits", required=False, type=int, default=12)
    parser.add_argument("--numbands", help="Number of bands", required=False, type=int, default=3)
    parser.add_argument("--epochs", help="Number of epochs", required=False, type=int, default=5)
    parser.add_argument("--perc", help="Percentage of dataset", required=False, type=float, default=0.3)
    parser.add_argument("--method", help="Type of autoencoder: 'small' or 'sscnet' ", required=False, type=str, default='small', choices=['small', 'sscnet'])
    parser.add_argument("--numclasses", help="Num of classes: 2 or all", required=False, type=int, default=2)
    parser.add_argument("--batchsize", help="Batchsize", required=False, type=int, default=16)
    parser.add_argument("--learning_rate", help="Learning rate", required=False, type=float, default=0.001)
    parser.add_argument("--maxp", help="Percentile max", required=False, type=float, default=99.0)
    parser.add_argument("--minp", help="Percentile min", required=False, type=float, default=0.1)


    args = parser.parse_args()

    num_qubits = args.N
    
    num_bands = args.numbands

    if num_bands == 3: bands = [0,1,2] # [3,2,1]
    else: bands = list(range(num_bands))

    num_epochs = args.epochs
    percentage_images = args.perc
    method = args.method

    num_classes = args.numclasses   #NON SERVE PER ORA
    if num_classes == 2:
        classes = ['River', 'SeaLake']
    else:
        classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
        
    batch_size = args.batchsize
    learning_rate = args.learning_rate
    maxp = args.maxp
    minp = args.minp

    print("\n\n#############################################################################")

    print("\nn_qubits: ", num_qubits)
    print("num bands: ", num_bands, 'bands: ', bands)
    print("epochs: ", num_epochs)
    print("percentage: ", percentage_images)
    print("method: ", method)
    print('classes: ', classes)
    print('batchsize: ', batch_size)
    print('learning rate: ', learning_rate)
    print('maxp: ', maxp, 'minp: ', minp)
    #endregion

    # Train Model
    torch.multiprocessing.set_start_method('spawn')  
    torch.set_float32_matmul_precision('high')
    # Instantiate LightningModule and DataModule

    #Initialize model
    if method == 'small': network = Autoencoder_small(num_bands, num_qubits)
    if method == 'sscnet': network = Autoencoder_sscnet(num_bands, num_qubits)

    data_module = EuroSATDataModule(
        whitelist_classes = classes,
        batch_size        = batch_size, 
        bands             = bands,
        mode              = 'aec',
        num_workers       = 12,
    )

    tb_logger   = pl.loggers.TensorBoardLogger(os.path.join('lightning_logs','classifiers'), name=method)

    # Instantiate ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('saved_models','classifiers'),
        filename=method,
        monitor='valid_loss',
        save_top_k=1,
        mode='min',
    )

    # Instantiate Trainer
    trainer = pl.Trainer(max_epochs=20, callbacks=[checkpoint_callback], logger=tb_logger)

    # Train the model
    trainer.fit(network, data_module)