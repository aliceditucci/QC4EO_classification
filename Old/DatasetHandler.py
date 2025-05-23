import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import glob
import os

from pyosv.io.reader import load
from pyosv.pre.normalizer import minmax_scaler

class DatasetHandler:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.classes = glob.glob(os.path.join(dataset_path, '*'))

    def print_classes(self):
        print('Classes: ') 
        for i,c in enumerate(self.classes): 
            print('     Class ' + str(i) + ' ->', c)

    def load_paths_labels(self, root, classes):
        # Initialize imaages path and images label lists
        imgs_path = []
        imgs_label = []
        class_counter = 0
        #encoded_class = np.zeros((len(classes)))

        class_dict = {
            "AnnualCrop" : 0,#[0,0,0,0],
            "Forest" : 1, #[0,0,0,1],
            "HerbaceousVegetation" : 2, #[0,0,1,0],
            "Highway" : 3, #[0,0,1,1],
            "Industrial" : 4, #[0,1,0,0],
            "Pasture" : 5,# [0,1,0,1],
            "PermanentCrop" : 6, #[0,1,1,0],
            "Residential" : 7, #[0,1,1,1],
            "River" : 8, #[1,0,0,0],
            "SeaLake" : 9, #[1,0,0,1]
        }

        # For each class in the class list
        for c in classes:
            # List all the images in that class
            paths_in_c = glob.glob(os.path.join(root, c+'/*'))
            # For each image in that class
            # print(paths_in_c)
            # print(os.path.normpath(paths_in_c))
            for path in paths_in_c:
                # Append the path of the image in the images path list
                # imgs_path.append(path)
                # Normalize paths to prevent mixed slashes   (I added this because the line above was giving / and \)
                imgs_path.append(os.path.normpath(path))
                # One hot encode the label
                #encoded_class[class_counter] = 1
                # Append the label in the iamges label list
                imgs_label.append(class_dict[c])#encoded_class)
                # Reset the class
                #encoded_class = np.zeros((len(classes)))

            # Jump to the next class after iterating all the paths
            # class_counter = class_counter + 1

        # Shuffler paths and labels in the same way
        c = list(zip(imgs_path, imgs_label))
        random.shuffle(c)
        imgs_path, imgs_label = zip(*c)

        return np.array(imgs_path), np.array(imgs_label)
    
    # Split the dataset into training and validation dataset 
    def train_validation_split(self, images, labels, split_factor = 0.2):
        val_size = int(len(images)*split_factor)
        train_size = int(len(images) - val_size)
        return images[0:train_size], labels[0:train_size, ...], images[train_size:train_size+val_size], labels[train_size:train_size+val_size, ...]
    
    # Data genertor: given images paths and images labels yield a batch of images and labels
    def cnn_data_loader(self, imgs_path, imgs_label, batch_size = 16, img_shape = (64, 64, 3), n_classes = 2):
        # Initialize the vectors to be yield
        batch_in = np.zeros((batch_size, img_shape[0], img_shape[1], img_shape[2]))
        batch_out = np.zeros((batch_size, n_classes))
        # Repeat until the generator will be stopped
        while True:
            # Load a batch of images and labels
            for i in range(batch_size):
                # Select a random image and labels from the dataset
                index = random.randint(0, len(imgs_path)-1)
                # Fill the vectors with images and labels
                batch_in[i, ...] = plt.imread(imgs_path[index])/255.0
                #one-hot encode the 10 classes
                l = np.zeros((n_classes))
                l[imgs_label[index]] = 1
                batch_out[i, ...] = l

            # Yield/Return the image and labeld vectors
            yield batch_in, batch_out
    
    # Data genertor: given images paths and images labels yield a batch of images and labels
    def qcnn_data_loader(self, imgs_path, imgs_label, batch_size = 1, img_shape = (64, 64, 3), returnpath=False):
        # Initialize the vectors to be yield
        batch_in = np.zeros((batch_size, img_shape[2], img_shape[0], img_shape[1]))
        batch_out = np.zeros((batch_size))

        # Repeat until the generator will be stopped
        while True:
            # Load a batch of images and labels
            for i in range(batch_size):
                # Select a random image and labels from the dataset
                index = random.randint(0, len(imgs_path)-1)
                # Fill the vectors with images and labels
                batch_in[i, ...] = np.transpose(plt.imread(imgs_path[index])/255.0)
                #labels are integers. No one-hot encoding
                batch_out[i] = imgs_label[index]
            # Yield/Return the image and labeld vectors
            if returnpath:
              yield  torch.Tensor(batch_in),  torch.Tensor(batch_out).type(torch.LongTensor), imgs_path[index]
            else:
              yield  torch.Tensor(batch_in),  torch.Tensor(batch_out).type(torch.LongTensor)
    
        # Data genertor: given images paths and images labels yield a batch of images and labels
    def ms_data_loader(self, imgs_path, imgs_label, batch_size = 1, img_shape = (64, 64, 3), returnpath=False, bands=3):
        # Initialize the vectors to be yield
        batch_in = np.zeros((batch_size, img_shape[2], img_shape[0], img_shape[1]))
        batch_out = np.zeros((batch_size))

        # Repeat until the generator will be stopped
        while True:
            # Load a batch of images and labels
            for i in range(batch_size):
                # Select a random image and labels from the dataset
                index = random.randint(0, len(imgs_path)-1)
                # Fill the vectors with images and labels
                img, meta, _ = load(imgs_path[index])
                img = minmax_scaler(img, mmax = 10000)
                img = img[:,:, bands]
                batch_in[i, ...] = np.transpose(img)
                #labels are integers. No one-hot encoding
                batch_out[i] = imgs_label[index]
            # Yield/Return the image and labeld vectors
            if returnpath:
              yield  torch.Tensor(batch_in),  torch.Tensor(batch_out).type(torch.LongTensor), imgs_path[index]
            else:
              yield  torch.Tensor(batch_in),  torch.Tensor(batch_out).type(torch.LongTensor)


