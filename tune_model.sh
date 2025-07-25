#!/bin/bash


for q in 4 8 12 16 20 24 28 32
do
  for m in small sscnet
  do
    echo "TUNING - m $m - q $q"
    python run_autoencoder.py --N $q --numbands 3 --epochs 60 --perc 1 --method $m --mode tune --numclasses 10
  done
done
