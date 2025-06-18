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

def make_folder(log_name, split):
    os.makedirs(log_name, exist_ok=True)
    split_folder = os.path.join(log_name, split)
    os.makedirs(split_folder, exist_ok=True)
    # Classes
    os.makedirs(os.path.join(split_folder, 'AnnualCrop'), exist_ok=True)
    os.makedirs(os.path.join(split_folder, 'Forest'), exist_ok=True)
    os.makedirs(os.path.join(split_folder, 'HerbaceousVegetation'), exist_ok=True)
    os.makedirs(os.path.join(split_folder, 'Highway'), exist_ok=True)
    os.makedirs(os.path.join(split_folder, 'Industrial'), exist_ok=True)
    os.makedirs(os.path.join(split_folder, 'Pasture'), exist_ok=True)
    os.makedirs(os.path.join(split_folder, 'PermanentCrop'), exist_ok=True)
    os.makedirs(os.path.join(split_folder, 'Residential'), exist_ok=True)
    os.makedirs(os.path.join(split_folder, 'River'), exist_ok=True)
    os.makedirs(os.path.join(split_folder, 'SeaLake'), exist_ok=True)

def save_latent_space(loader, network, log_name):
    network.eval()

    with torch.no_grad():
        for batch in loader:
            b_in, b_ou, path = batch
            b_in = b_in.to(network.device)
            z = network.encoder(b_in).cpu().numpy()[0,...]

            y = b_ou.cpu().numpy()[0,...]

            path = path[0].replace('.tif', '.npz')
            path = path.replace('EuroSAT-split', log_name)

            np.savez(path, latent_space=z, label=y)

if __name__ == "__main__":

    #region input arguments
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",             help="Number of qubits", required=False, type=int, default=12)
    parser.add_argument("--numbands",      help="Number of bands", required=False, type=int, default=3)
    parser.add_argument("--epochs",        help="Number of epochs", required=False, type=int, default=5)
    parser.add_argument("--perc",          help="Percentage of dataset", required=False, type=float, default=0.3)
    parser.add_argument("--method",        help="Type of autoencoder: 'small' or 'sscnet' ", required=False, type=str, default='small', choices=['small', 'sscnet', 'unet'])
    parser.add_argument("--numclasses",    help="Num of classes: 2 or all", required=False, type=int, default=2)
    parser.add_argument("--batchsize",     help="Batchsize", required=False, type=int, default=32)


    args = parser.parse_args()

    num_qubits = args.N    
    num_bands = args.numbands

    if num_bands == 3: bands = [3,2,1] # [3,2,1]
    else: bands = list(range(num_bands))

    num_epochs = args.epochs
    percentage_images = args.perc
    method = args.method

    num_classes = args.numclasses   #NON SERVE PER ORA
    if num_classes == 2: classes = ['River', 'SeaLake']
    else: classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
        
    batch_size    = args.batchsize

    print("\n\n#############################################################################")

    print("\nn_qubits: ",    num_qubits)
    print("num bands: ",     num_bands, 'bands: ', bands)
    print("epochs: ",        num_epochs)
    print("percentage: ",    percentage_images)
    print("method: ",        method)
    print('classes: ',       classes)
    print('batchsize: ',     batch_size)
    #endregion

    # Train Model
    torch.multiprocessing.set_start_method('spawn')  
    torch.set_float32_matmul_precision('high')
    # Instantiate LightningModule and DataModule

    #Initialize model
    if method == 'small':  network = Autoencoder_small (num_bands, num_qubits)
    if method == 'sscnet': network = Autoencoder_sscnet(num_bands, num_qubits)
    if method == 'unet':   network = UNet(num_bands, num_qubits)

    data_module = EuroSATDataModule(
        whitelist_classes = classes,
        batch_size        = batch_size, 
        bands             = bands,
        mode              = 'aec',
        num_workers       = 4,
    )

    log_name = f"{method}_q{num_qubits}_b{num_bands}"

    tb_logger   = pl.loggers.TensorBoardLogger(os.path.join('lightning_logs','classifiers'), name=log_name)

    # Instantiate ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('saved_models','classifiers'),
        filename=log_name,
        monitor='valid_loss',
        save_top_k=1,
        mode='min',
    )

    # Instantiate Trainer
    trainer = pl.Trainer(max_epochs=num_epochs, callbacks=[checkpoint_callback], logger=tb_logger)

    # Train the model
    trainer.fit(network, data_module)


    ### Save latent space
    data_module = EuroSATDataModule(
        whitelist_classes = classes,
        batch_size        = 1, 
        bands             = bands,
        mode              = 'aec',
        num_workers       = 4,
    )

    data_module.setup()
    train_loader = data_module.train_dataloader()
    valid_loader = data_module.val_dataloader()
    test_loader  = data_module.test_dataloader()

    make_folder(log_name, 'train')
    make_folder(log_name, 'val')
    make_folder(log_name, 'test')

    save_latent_space(train_loader, network, log_name)
    save_latent_space(valid_loader, network, log_name)
    save_latent_space(test_loader, network, log_name)



    
