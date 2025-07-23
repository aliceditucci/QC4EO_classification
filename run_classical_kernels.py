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

from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import random


def main(): 

    #region input arguments
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--numbands", help="Number of bands", required=False, type=int, default=3)
    parser.add_argument("--numclasses", help="Num of classes: 2 or all", required=False, type=int, default=2)
    parser.add_argument("--compress_method", help="Type compression: 'small', 'sscnet' or 'unet' ", required=False, type=str, default='small')
    parser.add_argument("--percentage_train", help="Percentage of training data", required=False, type=float, default=1.0)


    np.random.seed(1359)
    args = parser.parse_args()
    
    num_bands = args.numbands

    if num_bands == 3:
        bands = [3,2,1]
    else:
        bands = list(range(num_bands))

    
    compress_method = args.compress_method


    num_classes = args.numclasses   #NON SERVE PER ORA
    if num_classes == 2:
        classes = ['River', 'SeaLake']
    else:
        classes = []
        for i, c in enumerate(handler.classes):
            cl = os.path.basename(c) #changed because windows
            classes.append(cl)
        classes.sort()
    

    percentage_train = args.percentage_train
     
    print("\n\n#############################################################################")

    print("num bands: ", num_bands, 'bands: ', bands)
    print('classes: ', classes)
    print('compress_method: ', compress_method)
    print('percentage_train', percentage_train)
    #endregion

    features_list = [4,8,12,16, 20, 24, 28]

    linear_results = []
    rbf_results = []

    for N_FEATURES in features_list:

        print('\n#############################')
        print('N_FRATURES : ', N_FEATURES)
        # dataset_root = '../../../Data/aditucci/Latent/{}_q{}_b3'.format(compress_method, N_FEATURES)
        dataset_root = '../../../Data/aditucci/Latent_old/{}_q{}_b3'.format(compress_method, N_FEATURES)
        handler = Handler_quantum(dataset_root)

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

        
        print('train: ', train_features.shape)
        print('validation: ', val_features.shape)

        # Linear SVM
        linear_clf = svm.SVC(kernel='linear', C=1.0)  # You can tune C
        linear_clf.fit(train_features, train_labels)
        y_pred_linear = linear_clf.predict(val_features)

        linear_acc = accuracy_score(val_labels, y_pred_linear)
        linear_results.append((N_FEATURES, linear_acc))  # Save pair

        print("=== Linear SVM Results ===")
        print("Accuracy:", linear_acc)
        print(classification_report(val_labels, y_pred_linear))

        # RBF SVM with grid search for gamma and C
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        }

        rbf_clf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy')
        rbf_clf.fit(train_features, train_labels)
        y_pred_rbf = rbf_clf.predict(val_features)

        rbf_acc = accuracy_score(val_labels, y_pred_rbf)
        rbf_results.append((N_FEATURES, rbf_acc))  # Save pair

        print("\n=== RBF SVM Results ===")
        print("Best Parameters:", rbf_clf.best_params_)
        print("Accuracy:", rbf_acc)
        print(classification_report(val_labels, y_pred_rbf))

    save_data = {
        'linear_results': linear_results,
        'rbf_results': rbf_results
            }

    dir = '../../../Data/aditucci/Output_data_classical/'
    os.makedirs(dir, exist_ok=True)
    results_file_path = dir + 'classic_kernel_results_{}_{}.pkl'.format(compress_method, percentage_train)

    with open(results_file_path, 'wb') as f:
        pickle.dump(save_data, f)

    print('Results write sucessfully to ' + dir)

if __name__ == "__main__":
    main()