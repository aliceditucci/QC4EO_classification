for percentage_train in 0.01 0.05 0.1 0.3 0.5 0.7 1.0; do
# for percentage_train in 0.01; do
# for percentage_train in 0.7 1.0; do
    # for method in small sscnet unet; do
    for method in sscnet unet; do
        echo "Training models for method $method percentage $percentage_train"
        python -u run_classical_kernels.py --compress_method $method --percentage_train $percentage_train \
        >> ../../../Data/aditucci/classic_logs/${method}_${percentage_train}.log 2>&1 &
    done
    wait
done

