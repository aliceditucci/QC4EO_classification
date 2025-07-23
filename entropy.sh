# for qubit in 4 8 12 16
# do
#     echo "Computing entropy for $qubit qubits"
#     python run_compute_entropy.py --N $qubit --mps_sim 1 --compress_method small --kernel_type projected --ZZ_reps 1 --ent_type linear --compute_entropy 0 >> logs/entropy_small_${qubit}_projected_linear_1_1_0.log 2>&1 &
#     wait
# done

for kernel in fidelity; do
    for ent in full; do
        for reps in 1; do
            for N in 4 8 12 16; do
                for method in small sscnet unet; do
                    echo "Computing entropy for $N qubits with $method"
                    python run_compute_entropy.py --N $N --mps_sim 0 --compress_method $method --kernel_type $kernel \
                        --ZZ_reps $reps --ent_type $ent --compute_entropy 0 \
                        >> logs/entropy_${method}_${N}_${kernel}_${ent}_${reps}_1_0.log 2>&1 #&
                    # python run_compute_entropy.py --N $N --mps_sim 0 --compress_method $method --kernel_type $kernel \
                    #     --ZZ_reps $reps --ent_type $ent --compute_entropy 0 \
                    #     >> test.log 2>&1 &
                #wait
                done
            done
        done
    done
done

#wait