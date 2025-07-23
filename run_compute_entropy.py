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
    parser.add_argument("--ent_type", help="Type of entanglement: 'linear', 'full' ", required=False, type=str, default='linear')
    parser.add_argument("--compute_entropy", help="1 for compute entanglement entropy", required=False, type=int, default=0)
    parser.add_argument("--mps_sim", help="1 for mps simulation", required=False, type=int, default=0)
    parser.add_argument("--compress_method", help="Type compression: 'small', 'sscnet' or 'unet' ", required=False, type=str, default='small')

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
    dataset_root = '../../../Data/aditucci/Latent/{}_q{}_b3'.format(compress_method, N_FEATURES)
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
    else:
        mps_sim = False

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
    #endregion


    #Load data path
    percentage_images = 1.0

    train_root = dataset_root + '/train'
    x_train, _ = handler.load_paths_labels(train_root, classes=classes, percentage_images=percentage_images)

    # Assuming x_train is a 2D array of shape (num_samples, num_features)
    # Randomly select 20 samples without replacement
    x_train = x_train[np.random.choice(x_train.shape[0], 20, replace=False)]

    #Normalize data
    scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))  
    scaler.fit(x_train)
    train_features = scaler.transform(x_train)

    #make data dir
    dir = '../../../Data/aditucci/Output_data/{}/{}/N_{}/'.format(compress_method, kernel_type,N_FEATURES)
    os.makedirs(dir, exist_ok=True)

    print('train: ', train_features.shape)

    #Compute kernels and train SVC  
    if kernel_type == 'projected':
        entanglement_entropy, entropies = compute_entanglement_entropy(train_features, ZZ_reps, ent_type)

    if kernel_type == 'fidelity':
        entanglement_entropy, entropies = fk_compute_entanglement_entropy(train_features, ZZ_reps, ent_type)

    print("Entanglement entropy: ", entanglement_entropy, len(entropies))

    save_data = {
            'N_FEATURES': N_FEATURES,
            'entanglement_entropy': entanglement_entropy,
            'entropies':  entropies
                }

    results_file_path = dir + 'kernel_entropy_{}_{}_{}.pkl'.format(ent_type, ZZ_reps,compute_entropy)

    with open(results_file_path, 'wb') as f:
        pickle.dump(save_data, f)


    print('Results write sucessfully to ' + dir)

if __name__ == "__main__":
    main()