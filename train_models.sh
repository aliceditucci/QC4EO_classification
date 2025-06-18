#!/bin/bash

rm -r lightning_logs
rm -r saved_models
for qubit in 4 8 12 16 20 40 80 100
do
    echo "Training models for $qubit qubits"
    python run_autoencoder.py --N $qubit --numbands 3 --epochs 50 --perc 1 --method small
    python run_autoencoder.py --N $qubit --numbands 3 --epochs 50 --perc 1 --method sscnet 
    python run_autoencoder.py --N $qubit --numbands 3 --epochs 50 --perc 1 --method unet
done

