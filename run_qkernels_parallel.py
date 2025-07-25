import os
import sys
import pickle

import argparse

from Dataset_loader import Handler_quantum

#New imports
import numpy as np 
# from DatasetHandler import DatasetHandler #CHANGED!!
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kernel_functions import *

def main(): 

    #region input arguments
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="Number of qubits", required=False, type=int, default=4)
    parser.add_argument("--numclasses", help="Num of classes: 2 or all", required=False, type=int, default=2)
    parser.add_argument("--kernel_type", help="Type of kernel: 'fidelity', 'projected' ", required=False, type=str, default='fidelity')
    parser.add_argument("--ZZ_reps", help="ZZ map repetitions", required=False, type=int, default=1)
    parser.add_argument("--ent_type", help="Type of entanglement: 'linear', 'full' ", required=False, type=str, default='full')
    parser.add_argument("--mps_sim", help="1 for mps simulation", required=False, type=int, default=0)
    parser.add_argument("--compress_method", help="Type compression: 'small', 'sscnet' or 'unet' ", required=False, type=str, default='small')
    parser.add_argument("--percentage_train", help="Percentage of training data", required=False, type=float, default=1.0)
    parser.add_argument("--if_evaluatesvm", help="1 for evaluting svm", required=False, type=int, default=0)
    parser.add_argument("--data_scaling", help="Scaling factor of kernel input data", required=False, type=float, default=1.0)
    parser.add_argument("--gamma", help="Decay of the Gaussian-like projected kernel", required=False, type=float, default=1.0)


    np.random.seed(1359)
    args = parser.parse_args()

    N_FEATURES = args.N

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
    mps_sim = args.mps_sim

    if mps_sim ==1:
        mps_sim = True
        method='matrix_product_state'
        device='CPU'
    else:
        mps_sim = False
        method='statevector'
        device='GPU' 

    percentage_train = args.percentage_train
    
    if_evaluatesvm = args.if_evaluatesvm

    data_scaling = args.data_scaling
    gamma = args.gamma

    print("\n\n#############################################################################")

    print("\nn_qubits and features: ", N_FEATURES)
    print('classes: ', classes)
    print('kernel type: ', kernel_type)
    print('ZZ repetitions: ', ZZ_reps)
    print('ent_type: ', ent_type)
    print('mps simulation: ', mps_sim)
    print('compress_method: ', compress_method)
    print('percentage_train: ', percentage_train)
    print('data_scaling: ', data_scaling)
    print('gamma' , gamma)
    #endregion


    #Load data 
    percentage_images = 1.0 
    
    # train_root = dataset_root + '/train'
    # x_train, train_labels = handler.load_paths_labels(train_root, classes=classes, percentage_images=percentage_train)

    # val_root = dataset_root + '/val'
    # x_val, val_labels = handler.load_paths_labels(val_root, classes=classes, percentage_images=percentage_images)

    train_root = dataset_root + '/val'
    x_train, train_labels = handler.load_paths_labels(train_root, classes=classes, percentage_images=percentage_train)

    val_root = dataset_root + '/test'
    x_val, val_labels = handler.load_paths_labels(val_root, classes=classes, percentage_images=percentage_images)


    #Normalize data
    scaler = MinMaxScaler(feature_range=((0, np.pi)))
    # scaler = StandardScaler()
    scaler.fit(x_train)
    train_features = scaler.transform(x_train)
    val_features = scaler.transform(x_val)

    train_features *= data_scaling
    val_features *= data_scaling

    print('train: ', train_features.shape)
    print('validation: ', val_features.shape)



    #evaluate kernels
    ek = EvaluateKernels(method= method, device=device, ZZ_reps=ZZ_reps, ent_type=ent_type,
                n_qubits=N_FEATURES, kernel_type= kernel_type, gamma = gamma)

    independent_entries, gram_matrix_test = ek.compute_kernel_matrices(train_features, val_features)



    #make data dir and save
    dir = '../../../Data/aditucci/Output/{}/{}/N_{}/'.format(compress_method, kernel_type,N_FEATURES)  
    os.makedirs(dir, exist_ok=True)

    kernel_results_file_path = dir + 'kernel_results_{}_reps{}_mps{}_perc{}_scale{}_gamma{}'.format(ent_type, ZZ_reps, mps_sim, percentage_train, data_scaling, gamma)

    ek.save_kernel_data(f"{kernel_results_file_path}.npz", independent_entries, gram_matrix_test, train_labels, val_labels)


    if if_evaluatesvm == 1:
        # Evaluate using SVM
        svm_results_file_path = dir + 'svm_results_{}_reps{}_mps{}_perc{}_scale{}_gamma{}'.format(ent_type, ZZ_reps, mps_sim, percentage_train, data_scaling, gamma)
        score, score_train, confusion = evaluate_svm(independent_entries, train_labels, gram_matrix_test, val_labels)
        save_svm_data(f"{svm_results_file_path}.npz" , score, score_train, confusion)


if __name__ == "__main__":
    main()