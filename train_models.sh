#!/bin/bash

rm -r lightning_logs
rm -r saved_models
for qubit in 4 8 12 16 20 40 80 100
do
    echo "Training models for $qubit qubits"
    python run_autoencoder.py --N $qubit --numbands 3 --epochs 100 --perc 1 --method small --mode train
    python run_autoencoder.py --N $qubit --numbands 3 --epochs 100 --perc 1 --method sscnet --mode train
    python run_autoencoder.py --N $qubit --numbands 3 --epochs 100 --perc 1 --method unet --mode train
done

