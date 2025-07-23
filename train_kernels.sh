#!/bin/bash

# rm -r lightning_logs
# rm -r saved_models
# for qubit in 4 8 12 16 20 40 80 100
# for qubit in 4

# rm -r logs
# mkdir logs

# for kernel_type in fidelity; do
# for qubit in 4; do
# for reps in 1; do
# for ent in linear; do
# for mps_sim in 1; do
#     echo "Training models for $qubit qubits mps_sim $mps_sim all compress methods $reps $ent $mps_sim" 
#     python run_qkernels_parallel.py --N $qubit --mps_sim $mps_sim --compress_method small --kernel_type $kernel_type --ZZ_reps $reps --ent_type $ent --compute_entropy 0 >> logs/small_${qubit}_${kernel_type}_${ent}_${reps}_${mps_sim}_0.log 2>&1 &
#     #python run_qkernels_parallel.py --N $qubit --mps_sim $mps_sim --compress_method sscnet --kernel_type $kernel_type --ZZ_reps $reps --ent_type $ent --compute_entropy 0 >> logs/sscnet_${qubit}_${kernel_type}_${ent}_${reps}_${mps_sim}_0.log 2>&1 & 
#     #python run_qkernels_parallel.py --N $qubit --mps_sim $mps_sim --compress_method unet --kernel_type $kernel_type --ZZ_reps $reps --ent_type $ent --compute_entropy 0 >> logs/unet_${qubit}_${kernel_type}_${ent}_${reps}_${mps_sim}_0.log 2>&1 &
#     # python run_qkernels_parallel.py --N $qubit --mps_sim $mps_sim --compress_method small --kernel_type $kernel_type --ZZ_reps $reps --ent_type $ent --compute_entropy 0 >> logs/small_${qubit}_${kernel_type}_${ent}_${reps}_${mps_sim}_0.log 2>&1 &
    
# done
# done
# done
# done
# done
# wait


for kernel_type in projected; do
    for qubit in 4 8 12 16; do
        for ent in none; do
            for reps in 1 2; do
                for mps_sim in 1; do
                    for method in small sscnet unet; do
                        echo "Training models for $qubit qubits mps_sim $mps_sim method $method $reps $ent"
                        python run_qkernels_parallel.py --N $qubit --mps_sim $mps_sim \
                            --compress_method $method --kernel_type $kernel_type --ZZ_reps $reps --ent_type $ent --compute_entropy 0 \
                            >> logs/${method}_${qubit}_${kernel_type}_${ent}_${reps}_${mps_sim}_0.log 2>&1 #&
                            # --compress_method $method --kernel_type $kernel_type --ZZ_reps $reps --ent_type $ent --compute_entropy 0 \
                            # >> logs/fine_${method}_${qubit}_${kernel_type}_${ent}_${reps}_${mps_sim}_0.log 2>&1 #&
                            # --compress_method $method --kernel_type $kernel_type --ZZ_reps $reps --ent_type $ent --compute_entropy 0 \
                            # >> logs/${method}_${qubit}_${kernel_type}_${ent}_${reps}_${mps_sim}_0.log 2>&1 #&
                        # python run_qkernels_parallel.py --N $qubit --mps_sim $mps_sim --compress_method $method --kernel_type $kernel_type \
                        #      --ZZ_reps $reps --ent_type $ent --compute_entropy 0 \
                        #      >> test.log 2>&1 &
                    #wait
                    done
                done
            done
        done
    done
done

# for kernel_type in fidelity; do
# for qubit in 4; do
# for reps in 1; do
# for ent in linear; do
# for mps_sim in 0 1; do
#     echo "Training models for $qubit qubits mps_sim $mps_sim all compress methods $reps $ent $mps_sim" 
#     python run_qkernels_parallel.py --N $qubit --mps_sim $mps_sim --compress_method small --kernel_type $kernel_type --ZZ_reps $reps --ent_type $ent --compute_entropy 0 >> logs/small_${qubit}_${kernel_type}_${ent}_${reps}_${mps_sim}_0.log 2>&1 &
#     python run_qkernels_parallel.py --N $qubit --mps_sim $mps_sim --compress_method sscnet --kernel_type $kernel_type --ZZ_reps $reps --ent_type $ent --compute_entropy 0 >> logs/sscnet_${qubit}_${kernel_type}_${ent}_${reps}_${mps_sim}_0.log 2>&1 & 
#     python run_qkernels_parallel.py --N $qubit --mps_sim $mps_sim --compress_method unet --kernel_type $kernel_type --ZZ_reps $reps --ent_type $ent --compute_entropy 0 >> logs/unet_${qubit}_${kernel_type}_${ent}_${reps}_${mps_sim}_0.log 2>&1 &
#     #python run_qkernels_parallel.py --N $qubit --mps_sim $mps_sim --compress_method small --kernel_type $kernel_type --ZZ_reps $reps --ent_type $ent --compute_entropy 0 >> test.log 2>&1 &
    
# done
# wait
# done
# done
# done
# done

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