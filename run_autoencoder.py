from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from functools import partial
import numpy as np
import argparse
import optuna
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


def optuna_objective(trial: optuna.trial.Trial, parser) -> float:
    
    args = parser.parse_args()

    num_qubits = args.N    
    num_bands = args.numbands
    mode = args.mode

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
   
    if mode=='train':
        # FIXME these values will be fixed after tuning the models
        initial_lr     = 0.002
        loss_weigths   = [0.2, 0.8, 0.5]
        gamma          = 0.1
        lr_scheduler_f = 3

    if mode=='tune':
        initial_lr     = trial.suggest_float('initial_lr', 0.00001, 0.001)
        l1             = trial.suggest_float('l1', 0.1, 0.9)
        l2             = trial.suggest_float('l2', 0.1, 0.9)
        l3             = trial.suggest_float('l3', 0.1, 0.9)
        loss_weigths   = [l1, l2, l3] 
        gamma          = trial.suggest_float('gamma', 0.1, 0.9) 
        lr_scheduler_f = trial.suggest_int('lr_scheduler_f', 1, 5)

    print('Initial Learning Rate', initial_lr)
    print('Loss Weights', loss_weigths)
    print('Gamma', gamma)
    print('Learning Rate Scheduler Frequency', lr_scheduler_f)

    #Initialize model
    if method == 'small':  network = Autoencoder_small (num_bands, num_qubits, initial_lr=initial_lr, loss_weigths=loss_weigths, gamma=gamma, lr_scheduler_f=lr_scheduler_f)
    if method == 'sscnet': network = Autoencoder_sscnet(num_bands, num_qubits, initial_lr=initial_lr, loss_weigths=loss_weigths, gamma=gamma, lr_scheduler_f=lr_scheduler_f)
    if method == 'unet':   network = UNet(num_bands, num_qubits, initial_lr=initial_lr, loss_weigths=loss_weigths, gamma=gamma, lr_scheduler_f=lr_scheduler_f)

    data_module = EuroSATDataModule(
        whitelist_classes = classes,
        batch_size        = batch_size, 
        bands             = bands,
        mode              = 'aec',
        num_workers       = 8,
    )

    log_name = f"{method}_q{num_qubits}_b{num_bands}_lr{initial_lr}_lw_{loss_weigths}_g{gamma}_lf{lr_scheduler_f}"

    tb_logger   = pl.loggers.TensorBoardLogger(os.path.join('lightning_logs','classifiers'), name=log_name)

    # Instantiate ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('saved_models','classifiers'),
        filename=log_name,
        monitor='valid_loss',
        save_top_k=1,
        mode='min',
    )

    early_stopping = EarlyStopping(
        monitor='valid_loss',
        patience=int(num_epochs*0.05),
        verbose=True,
        mode='min'
    )

    # Instantiate Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        callbacks=[checkpoint_callback, early_stopping], 
        logger=tb_logger,
        log_every_n_steps=1
    )

    # Train the model
    trainer.fit(network, data_module)


    if mode=='train':
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
        
  
    return trainer.callback_metrics["valid_loss"].item()


if __name__ == "__main__":

    torch.multiprocessing.set_start_method('spawn')  
    torch.set_float32_matmul_precision('high')

    #region input arguments
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",             help="Number of qubits", required=False, type=int, default=12)
    parser.add_argument("--numbands",      help="Number of bands", required=False, type=int, default=3)
    parser.add_argument("--epochs",        help="Number of epochs", required=False, type=int, default=100)
    parser.add_argument("--perc",          help="Percentage of dataset", required=False, type=float, default=0.3)
    parser.add_argument("--method",        help="Type of autoencoder: 'small' or 'sscnet' ", required=False, type=str, default='small', choices=['small', 'sscnet', 'unet'])
    parser.add_argument("--numclasses",    help="Num of classes: 2 or all", required=False, type=int, default=2)
    parser.add_argument("--batchsize",     help="Batchsize", required=False, type=int, default=32)
    parser.add_argument("--mode",          help="Train or tune mdoe", required=True, type=str, default='train')


    pruner = optuna.pruners.MedianPruner()
    study  = optuna.create_study(direction="minimize", pruner=pruner)


    args = parser.parse_args()

    if args.mode =='tune':
        # NOTE this shall be selected depending on your needs
        n_trials = 20
        timeout = 21600*4 # s -> 6 hours

    if args.mode =='train':
        n_trials = 1
        timeout = 21600 


    study.optimize(
        partial(
            optuna_objective,
            parser = parser
        ),
        n_trials = n_trials,
        timeout  = timeout
    )

    if args.mode =='tune':
        print(f"Numer of completed trials: {len(study.trials)}")
        print("Best trial")
        trial=study.best_trial
        print(f"Value: {trial.value}")
        print(f"Params")
        for key, value in trial.params.items(): print(f"{key}:{value}")
