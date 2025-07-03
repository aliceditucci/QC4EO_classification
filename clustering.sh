#!/bin/bash

split='train'

for qubit in 4 8 12 16 20 40 80 100
do
    python clustering.py --N $qubit --numbands 3 --method small --split $split
    python clustering.py --N $qubit --numbands 3 --method sscnet --split $split
    python clustering.py --N $qubit --numbands 3 --method unet --split $split
done