#!/bin/bash

conda activate pytorch
rm -r lightning_logs
rm -r saved_models
rm log_small
rm log_sscnet
rm log_unet

for qubit in 4 8 12 16 20 40 80 100
do
    python run_autoencoder.py --N $qubit --numbands 3 --epochs 50 --perc 1 --method small  > log_small  2>&1 &
    python run_autoencoder.py --N $qubit --numbands 3 --epochs 50 --perc 1 --method sscnet > log_sscnet 2>&1 &
    python run_autoencoder.py --N $qubit --numbands 3 --epochs 50 --perc 1 --method unet   > log_unet   2>&1
done

