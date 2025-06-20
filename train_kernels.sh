#!/bin/bash

# rm -r lightning_logs
# rm -r saved_models
# for qubit in 4 8 12 16 20 40 80 100
# for qubit in 4

# rm -r logs
# mkdir logs

for mps_sim in 0; do
for qubit in 4 8 12 16 20
do
    echo "Training models for $qubit qubits mps_sim $mps_sim all compress methods" 
    python run_qkernels_parallel.py --N $qubit --mps_sim $mps_sim --compress_method small --kernel_type projected --ZZ_reps 1 --ent_type linear --compute_entropy 0 >> logs/small_${qubit}_projected_linear_1_${mps_sim}_0.log 2>&1 &
    python run_qkernels_parallel.py --N $qubit --mps_sim $mps_sim --compress_method sscnet --kernel_type projected --ZZ_reps 1 --ent_type linear --compute_entropy 0 >> logs/sscnet_${qubit}_projected_linear_1_${mps_sim}_0.log 2>&1 & 
    python run_qkernels_parallel.py --N $qubit --mps_sim $mps_sim --compress_method unet --kernel_type projected --ZZ_reps 1 --ent_type linear --compute_entropy 0 >> logs/unet_${qubit}_projected_linear_1_${mps_sim}_0.log 2>&1 &
    #python run_qkernels_parallel.py --N $qubit --mps_sim 1 --compress_method small --kernel_type projected --ZZ_reps 1 --ent_type linear --compute_entropy 0 
    
    wait 
done
done


# #!/bin/bash

# rm -r lightning_logs
# rm -r saved_models

# # List of qubit sizes to iterate over
# for qubit in 4 8 12 16
# do
#     echo "Training models for $qubit qubits"

#     # Run each model training in background
#     python run_autoencoder.py --N $qubit --numbands 3 --epochs 50 --perc 1 --method small &
#     python run_autoencoder.py --N $qubit --numbands 3 --epochs 50 --perc 1 --method sscnet &
#     python run_autoencoder.py --N $qubit --numbands 3 --epochs 50 --perc 1 --method unet &

#     wait  # Wait for all 3 background jobs for this qubit to finish before moving to the next
# done


# #!/bin/bash

# rm -r lightning_logs
# rm -r saved_models

# # Generate all commands
# for qubit in 4 8 12 16
# do
#     for method in small sscnet unet
#     do
#         echo "python run_autoencoder.py --N $qubit --numbands 3 --epochs 50 --perc 1 --method $method"
#     done
# done > job_list.txt

# # Run them in parallel (e.g., 4 jobs at a time)
# cat job_list.txt | parallel -j 4