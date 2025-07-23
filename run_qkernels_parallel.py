import os
import sys
import pickle

import argparse

from Dataset_loader import Handler_quantum

#New imports
import numpy as np 
# from DatasetHandler import DatasetHandler #CHANGED!!
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import MinMaxScaler


from kernel_functions import *

def main(): 

    #region input arguments
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="Number of qubits", required=False, type=int, default=4)
    parser.add_argument("--numbands", help="Number of bands", required=False, type=int, default=3)
    parser.add_argument("--numclasses", help="Num of classes: 2 or all", required=False, type=int, default=2)
    parser.add_argument("--kernel_type", help="Type of kernel: 'fidelity', 'projected' ", required=False, type=str, default='fidelity')
    parser.add_argument("--ZZ_reps", help="ZZ map repetitions", required=False, type=int, default=1)
    parser.add_argument("--ent_type", help="Type of entanglement: 'linear', 'full' ", required=False, type=str, default='full')
    parser.add_argument("--compute_entropy", help="1 for compute entanglement entropy", required=False, type=int, default=0)
    parser.add_argument("--mps_sim", help="1 for mps simulation", required=False, type=int, default=0)
    parser.add_argument("--compress_method", help="Type compression: 'small', 'sscnet' or 'unet' ", required=False, type=str, default='small')
    parser.add_argument("--percentage_train", help="Percentage of training data", required=False, type=float, default=1.0)


    np.random.seed(1359)
    args = parser.parse_args()

    N_FEATURES = args.N
    
    num_bands = args.numbands

    if num_bands == 3:
        bands = [3,2,1]
    else:
        bands = list(range(num_bands))

    
    compress_method = args.compress_method

    #Set dataset handler
    # dataset_root = '../../../Data/aditucci/Latent/{}_q{}_b3'.format(compress_method, N_FEATURES)
    dataset_root = '../../../Data/aditucci/Latent_old/{}_q{}_b3'.format(compress_method, N_FEATURES)
    handler = Handler_quantum(dataset_root)

    num_classes = args.numclasses   #NON SERVE PER ORA
    if num_classes == 2:
        classes = ['River', 'SeaLake']
    else:
        classes = []
        for i, c in enumerate(handler.classes):
            cl = os.path.basename(c) #changed because windows
            classes.append(cl)
        classes.sort()
    
    kernel_type = args.kernel_type
    ZZ_reps = args.ZZ_reps
    ent_type = args.ent_type
    compute_entropy = args.compute_entropy
    mps_sim = args.mps_sim

    if compute_entropy ==1:
        compute_entropy = True
    else:
        compute_entropy = False

    if mps_sim ==1:
        mps_sim = True
        method='matrix_product_state'
        device='CPU'
    else:
        mps_sim = False
        method='statevector'
        device='GPU' 

    percentage_train = args.percentage_train
     
    print("\n\n#############################################################################")

    print("\nn_qubits and features: ", N_FEATURES)
    print("num bands: ", num_bands, 'bands: ', bands)
    print('classes: ', classes)
    print('kernel type: ', kernel_type)
    print('ZZ repetitions: ', ZZ_reps)
    print('ent_type: ', ent_type)
    print('compute entropy:', compute_entropy)
    print('mps simulation: ', mps_sim)
    print('compress_method: ', compress_method)
    print('percentage_train', percentage_train)
    #endregion

    #Load data path
    percentage_images = 1.0 

    
    train_root = dataset_root + '/train'
    #NB percentage_train #####################
    x_train, train_labels = handler.load_paths_labels(train_root, classes=classes, percentage_images=percentage_train)

    val_root = dataset_root + '/val'
    x_val, val_labels = handler.load_paths_labels(val_root, classes=classes, percentage_images=percentage_images)


    #Normalize data
    scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))  
    scaler.fit(x_train)
    train_features = scaler.transform(x_train)
    val_features = scaler.transform(x_val)

    #make data dir
    dir = '../../../Data/aditucci/Output_data_old/{}/{}/N_{}/'.format(compress_method, kernel_type,N_FEATURES)
    os.makedirs(dir, exist_ok=True)

    print('train: ', train_features.shape)
    print('validation: ', val_features.shape)

    #Compute kernels and train SVC  
    entanglement_entropy = 0 

    if kernel_type == 'fidelity':
        #independent_entries, score, confusion = fidelity_kernels_simulator(train_features, train_labels, val_features, val_labels, ZZ_reps, ent_type, method, device)
        #independent_entries, score, confusion = fidelity_kernels_parallel(train_features, train_labels, val_features, val_labels, ZZ_reps, ent_type, method, device)
        independent_entries, gram_matrix_test, score, score_train, confusion = fidelity_kernels_batch(train_features, train_labels, val_features, val_labels, ZZ_reps, ent_type, method, device)

    elif kernel_type == 'projected':
        independent_entries, score, confusion = projected_kernels(train_features, train_labels, val_features, val_labels, ZZ_reps, ent_type, method, device)


    save_data = {
            'kernel_entries': independent_entries,
            'gram_matrix_test': gram_matrix_test,
            'N_FEATURES': N_FEATURES,
            'scores': score,
            'score_train': score_train,
            'confusion': confusion,
            'entanglement_entropy': entanglement_entropy
                }

    results_file_path = dir + 'kernel_results_{}_{}_{}_{}_{}.pkl'.format(ent_type, ZZ_reps,compute_entropy, mps_sim, percentage_train)

    with open(results_file_path, 'wb') as f:
        pickle.dump(save_data, f)


    print('Results write sucessfully to ' + dir)

if __name__ == "__main__":
    main()