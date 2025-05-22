import numpy as np
import random
import torch
import glob
import os

from pyosv.io.reader import load

class DatasetHandler:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.classes = glob.glob(os.path.join(dataset_path, '*'))

        self.mmin = None
        self.mmax = None
        self.clip = None


    def print_classes(self):
        print('Classes: ') 
        for i,c in enumerate(self.classes): 
            print('     Class ' + str(i) + ' ->', c)

    def load_paths_labels(self, root, classes, percentage_images=1):
        # Initialize imaages path and images label lists
        imgs_path = []
        imgs_label = []

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

            # Keep only a percentage of the images
            num_to_keep = int(len(paths_in_c) * percentage_images)
            paths_in_c = paths_in_c[:num_to_keep]


            for path in paths_in_c:
                # Append the path of the image in the images path list
                imgs_path.append(os.path.normpath(path)) # Normalize paths to prevent mixed slashes 
                # Append the label in the iamges label list
                imgs_label.append(class_dict[c])

        # # Shuffler paths and labels in the same way
        # c = list(zip(imgs_path, imgs_label))
        # random.shuffle(c)
        # imgs_path, imgs_label = zip(*c)

        #NO SHUFFLE HERE!!

        return np.array(imgs_path), np.array(imgs_label)
    
    # Split the dataset into training and validation dataset 
    def train_validation_split(self, images, labels, split_factor = 0.2):
        val_size = int(len(images)*split_factor)
        train_size = int(len(images) - val_size)

        # return images[0:train_size], labels[0:train_size, ...], images[train_size:train_size+val_size], labels[train_size:train_size+val_size, ...]

        #SHUFFLE ADDED HERE
        train_images = images[:train_size]
        train_labels = labels[:train_size]
        val_images = images[train_size:train_size+val_size]
        val_labels = labels[train_size:train_size+val_size]

        # Shuffle training set here
        train_data = list(zip(train_images, train_labels))
        random.shuffle(train_data)
        train_images, train_labels = zip(*train_data)

        # Shuffle validation set here
        val_data = list(zip(val_images, val_labels))
        random.shuffle(val_data)
        val_images, val_labels = zip(*val_data)

        return np.array(train_images), np.array(train_labels), np.array(val_images), np.array(val_labels)

    
    def compute_dataset_stats(self, imgs_path, bands=[0,1,2], max_images=100, percentiles : list = [None, None]):
        min_per_band = None
        max_per_band = None
        sum_per_band = None
        count = 0
        pixel_values_per_band = [[] for _ in bands]

        for idx, path in enumerate(imgs_path):
            img, meta, _ = load(path)
            img = img[:, :, bands]
            img = img.astype(np.float32)
            img_min = np.min(img, axis=(0,1))
            img_max = np.max(img, axis=(0,1))
            img_sum = np.sum(img, axis=(0,1))
            img_count = img.shape[0] * img.shape[1]

            if min_per_band is None:
                min_per_band = img_min
                max_per_band = img_max
                sum_per_band = img_sum
            else:
                min_per_band = np.minimum(min_per_band, img_min)
                max_per_band = np.maximum(max_per_band, img_max)
                sum_per_band += img_sum

            if idx < max_images:
                for i in range(len(bands)):
                    pixel_values_per_band[i].extend(img[:, :, i].ravel())

            count += img_count

        mean_per_band = sum_per_band / count

        # Compute percentiles for outlier detection
        pmin = []
        pmax = []

        for (i,pixels) in enumerate(pixel_values_per_band):
            pmin.append(np.percentile(pixels, percentiles[0][i]))
            pmax.append(np.percentile(pixels, percentiles[1][i]))


        self.mmin = pmin
        self.mmax = pmax

        return min_per_band, max_per_band, mean_per_band, pixel_values_per_band, pmin, pmax

    def minmaxscaler(self, img : np.ndarray, mmin : list = None, mmax : list = None, clip : list = [None, None]) -> np.ndarray: 
        '''
            Apply the min max scaler to the input img:  '''
        

        if len(img.shape) != 3:
            raise Exception('Error: lenght of image shape must be 3 - (space, space, channels)')
        
        if mmin is not None:
            if len(mmin) != img.shape[-1]:
                raise Exception('Error: lenght of mmin must be equals to number of bands')
        
        if mmax is not None:
            if len(mmax) != img.shape[-1]:
                raise Exception('Error: lenght of mmax must be equals to number of bands')
        
        if mmin == None: raise Exception('Error: not normalizing over the dataset')
        if mmax == None: raise Exception('Error: not normalizing over the dataset')

        for i in range(img.shape[-1]): 

            if (clip[0] is not None) and (clip[1] is not None): 
                img[:,:,i] = np.clip(img[:,:,i], clip[0][i], clip[-1][i])
                mmin[i] = clip[0][i]
                mmax[i] = clip[-1][i]

            img[:,:,i] = np.clip(img[:,:,i], mmin[i], mmax[i])

            num = img[:,:,i] - mmin[i]
            den = mmax[i] - mmin[i]
            
            img[:, :, i] = np.divide(num, den, out=np.zeros_like(num)+0.5, where=den != 0)
    
        return img

    def ms_data_loader(self, imgs_path, imgs_label, batch_size = 1, img_shape = (64, 64, 3), returnpath=False, bands=[3,2,1]):

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
                img = img[:, :, bands]  # Select relevant bands
                img = img.astype(np.float32) 
                img = self.minmaxscaler(img, self.mmin, self.mmax)
                batch_in[i, ...] = np.transpose(img)

                batch_out[i] = imgs_label[index]
                
            # Yield/Return the image and labeld vectors
            if returnpath:
              yield  torch.Tensor(batch_in),  torch.Tensor(batch_out).type(torch.LongTensor), imgs_path[index]
            else:
              yield  torch.Tensor(batch_in),  torch.Tensor(batch_out).type(torch.LongTensor)



    #JUST TO DOUBLE CHECK 

    def compute_stats_tocheck(self, loader, bands=[0,1,2], max_images=100):
        min_per_band = None
        max_per_band = None
        sum_per_band = None
        count = 0

        pixel_values_per_band = [[] for _ in bands]

        processed_images = 0

        for _ in range(5):  # fixed number of batches
            imgs, _ = next(loader)  # imgs: [B, C, H, W]

            if min_per_band is None:
                min_per_band = torch.amin(imgs, dim=(0, 2, 3))
                max_per_band = torch.amax(imgs, dim=(0, 2, 3))
                sum_per_band = torch.sum(imgs, dim=(0, 2, 3))
            else:
                min_per_band = torch.minimum(min_per_band, torch.amin(imgs, dim=(0, 2, 3)))
                max_per_band = torch.maximum(max_per_band, torch.amax(imgs, dim=(0, 2, 3)))
                sum_per_band += torch.sum(imgs, dim=(0, 2, 3))

            count += imgs.shape[0] * imgs.shape[2] * imgs.shape[3]

            # Collect pixels for histogram (up to max_images)
            if processed_images < max_images:
                n = min(imgs.shape[0], max_images - processed_images)
                for i, b in enumerate(bands):
                    pixel_values_per_band[i].extend(imgs[:n, b, :, :].flatten().tolist())
                processed_images += n

        mean_per_band = sum_per_band / count
        return min_per_band, max_per_band, mean_per_band, pixel_values_per_band
    
