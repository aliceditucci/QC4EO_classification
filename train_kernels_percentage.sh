for qubit in 4 8 12 16; do
    for kernel_type in fidelity; do
        for percentage_train in 0.1 0.3 0.5 0.7 1.0; do
        # for percentage_train in 0.01 0.05; do
            for ent in linear; do
                for reps in 1 2; do
                    for mps_sim in 1; do
                        for method in small sscnet unet; do
                            echo "Training models for $qubit qubits mps_sim $mps_sim method $method $reps $ent $percentage_train"
                            python run_qkernels_parallel_percentage.py --N $qubit --mps_sim $mps_sim \
                                --compress_method $method --kernel_type $kernel_type --ZZ_reps $reps --ent_type $ent --compute_entropy 0 --percentage_train $percentage_train \
                                >> perc_logs/${method}_${qubit}_${kernel_type}_${ent}_${reps}_${mps_sim}_0_${percentage_train}.log 2>&1 #&
                        done
                    done
                done
            done
        done
    done
done