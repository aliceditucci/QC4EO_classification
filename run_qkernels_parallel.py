import os
import sys
import pickle


import argparse

# from Dataset_loader import DatasetHandler

#New imports
from PIL import Image
import numpy as np 
from DatasetHandler import DatasetHandler #CHANGED!!
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm.auto import tqdm
import time

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from sklearn.svm import SVC

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

    np.random.seed(1359)

    args = parser.parse_args()

    N_FEATURES = args.N
    
    num_bands = args.numbands

    if num_bands == 3:
        bands = [3,2,1]
    else:
        bands = list(range(num_bands))


    num_classes = args.numclasses   #NON SERVE PER ORA
    if num_classes == 2:
        classes = ['River', 'SeaLake']
    else:
        classes = []
        for i, c in enumerate(handler.classes):
            # cl = c.split('/')[-1]
            cl = os.path.basename(c) #changed because windows
            classes.append(cl)
        classes.sort()
    
    kernel_type = args.kernel_type
    ZZ_reps = args.ZZ_reps
    ent_type = args.ent_type
    compute_entropy = args.compute_entropy


    print("\n\n#############################################################################")

    print("\nn_qubits and features: ", N_FEATURES)
    print("num bands: ", num_bands, 'bands: ', bands)
    print('classes: ', classes)
    print('kernel type: ', kernel_type)
    print('ZZ repetitions: ', ZZ_reps)
    print('ent_type: ', ent_type)
    print('compute entropy', compute_entropy)
    #endregion


    #Load data path
    # dataset_root = '../EuroSAT-split/train'
    # handler = DatasetHandler(dataset_root)
    # imgs_path, imgs_label = handler.load_paths_labels(dataset_root, classes=classes, percentage_images=percentage_images)

    dataset_root = '../EuroSAT_RGB/EuroSAT_RGB' #CHANGED!!
    handler = DatasetHandler(dataset_root)
    imgs_path, imgs_label = handler.load_paths_labels(dataset_root, classes=classes)

    # Reduce data set just as a test
    dataset_size = 60
    imgs_path = imgs_path[0:dataset_size]
    imgs_label = imgs_label[0:dataset_size]

    #Split data set 
    train_imgs, train_labels, val_images, val_labels = handler.train_validation_split(imgs_path, imgs_label, split_factor=0.2)
    
    #Rename labels since I consider only 2 classes
    for i in range(len(train_labels)):
        lr = train_labels[i]
        l = 0

        if lr == 8:
            l = 0
        elif lr == 9:
            l = 1
        
        train_labels[i] = l

    for i in range(len(val_labels)):
        lr = val_labels[i]
        l = 0

        if lr == 8:
            l = 0
        elif lr == 9:
            l = 1
        
        val_labels[i] = l

    #Load images
    x_train = []
    for filename in train_imgs:
        im=Image.open(filename)
        x_train.append(im)

    x_test = []
    for filename in val_images:
        im=Image.open(filename)
        x_test.append(im)
        
    x_train = np.asarray(x_train)
    y_train = np.asarray(train_labels)
    x_test = np.asarray(x_test)
    y_test = np.asarray(val_labels)

    x_train, x_test = x_train/255.0, x_test/255.0

    #Normalize data and set the number of features (Artur's way) - to be removed when we have latent space
    #Train Data
    # Flatten each image (e.g., 64 x 64 x 3 → 12288)
    X_flat = x_train.reshape(x_train.shape[0], -1)

    # Center and normalize data
    X_normalized = (X_flat - X_flat.min(axis=0))
    X_normalized = X_normalized/X_normalized.max(axis=0)

    # Reshuffle columns in descending variance order
    X_var = X_normalized.var(axis=0)
    X_var_ordered = X_normalized[:,np.argsort(-X_var)] # negated array to sort descending 

    #Test data 
    X_test_flat = x_test.reshape(x_test.shape[0], -1)

    X_test_normalized = (X_test_flat - X_test_flat.min(axis=0))
    X_test_normalized = X_test_normalized/X_test_normalized.max(axis=0)

    # Reshuffle columns in descending variance order
    X_test_var = X_test_normalized.var(axis=0)
    X_test_var_ordered = X_test_normalized[:,np.argsort(-X_test_var)] # negated array to sort descending 

    print(y_train)
    print(y_test)

    m = X_var_ordered.shape[0] #Number of datapoints

    #make data dir
    dir = './Data/{}/N_{}/'.format(kernel_type,N_FEATURES)
    os.makedirs(dir, exist_ok=True)

    #Compute kernels and train SVC
    
    np.random.seed(1359)

    #Change this later
    train_features = X_var_ordered[:,:N_FEATURES]
    train_labels = y_train

    test_features = X_test_var_ordered[:,:N_FEATURES]
    test_labels = y_test

    entanglement_entropy = 0 

    if kernel_type == 'fidelity':
        independent_entries, score, confusion = fidelity_kernels(train_features, train_labels, test_features, test_labels, ZZ_reps, ent_type)

    elif kernel_type == 'projected':
    
        if compute_entropy == 1:
            independent_entries, score, confusion, entanglement_entropy = projected_kernels(train_features, train_labels, test_features, test_labels, ZZ_reps, ent_type, compute_entropy=True)
        else:
            independent_entries, score, confusion = projected_kernels(train_features, train_labels, test_features, test_labels, ZZ_reps, ent_type)


    save_data = {
            'kernel_entries': independent_entries,
            'N_FEATURES': N_FEATURES,
            'scores': score,
            'confusion': confusion,
            'entanglement_entropy': entanglement_entropy
                }

    results_file_path = dir + 'kernel_results_{}_{}_{}.pkl'.format(ent_type, ZZ_reps,compute_entropy)

    with open(results_file_path, 'wb') as f:
        pickle.dump(save_data, f)


    print('Results write sucessfully to ' + dir)

if __name__ == "__main__":
    main()