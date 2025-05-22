import os
import sys
import pickle


import argparse
import torch.optim as optim
from torchinfo import summary
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError


from Dataset_loader import DatasetHandler
from autoencoder_model import *


def main(): 

    #region input arguments
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", help="Number of qubits", required=False, type=int, default=12)
    parser.add_argument("--numbands", help="Number of bands", required=False, type=int, default=3)
    parser.add_argument("--epochs", help="Number of epochs", required=False, type=int, default=5)
    parser.add_argument("--perc", help="Percentage of dataset", required=False, type=float, default=0.3)
    parser.add_argument("--method", help="Type of autoencoder: 'small' or 'sscnet' ", required=False, type=str, default='small')
    parser.add_argument("--numclasses", help="Num of classes: 2 or all", required=False, type=int, default=2)
    parser.add_argument("--batchsize", help="Batchsize", required=False, type=int, default=16)
    parser.add_argument("--learning_rate", help="Learning rate", required=False, type=float, default=0.001)
    parser.add_argument("--maxp", help="Percentile max", required=False, type=float, default=99.0)
    parser.add_argument("--minp", help="Percentile min", required=False, type=float, default=0.1)


    args = parser.parse_args()

    num_qubits = args.N
    
    num_bands = args.numbands

    if num_bands == 3:
        bands = [3,2,1]
    else:
        bands = list(range(num_bands))

    num_epochs = args.epochs
    percentage_images = args.perc
    method = args.method

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
    
    batch_size = args.batchsize
    learning_rate = args.learning_rate
    maxp = args.maxp
    minp = args.minp

    print("\n\n#############################################################################")

    print("\nn_qubits: ", num_qubits)
    print("num bands: ", num_bands, 'bands: ', bands)
    print("epochs: ", num_epochs)
    print("percentage: ", percentage_images)
    print("method: ", method)
    print('classes: ', classes)
    print('batchsize: ', batch_size)
    print('learning rate: ', learning_rate)
    print('maxp: ', maxp, 'minp: ', minp)
    #endregion


    #Load data path
    dataset_root = '../EuroSAT-split/train'
    handler = DatasetHandler(dataset_root)
    imgs_path, imgs_label = handler.load_paths_labels(dataset_root, classes=classes, percentage_images=percentage_images)

    #Split data set 
    train_imgs, train_labels, val_images, val_labels = handler.train_validation_split(imgs_path, imgs_label, split_factor=0.2)
    
    num_batches = len(train_imgs) // batch_size
    maxpercentiles = [maxp]*num_bands
    minpercentiles = [minp]*num_bands
    
    
    #make data dir
    dir = './Data/{}_qubits/'.format(num_qubits)
    os.makedirs(dir, exist_ok=True)


    #Compute and save dataset statistics
    min_per_band, max_per_band, mean_per_band, pixel_values_per_band, pmin, pmax = handler.compute_dataset_stats(train_imgs, bands = bands, max_images=100, percentiles = [minpercentiles, maxpercentiles])

    stats_file_path = dir + 'stats_{}.pkl'.format(method)

    save_data = {
            'min_per_band': min_per_band,
            'max_per_band': max_per_band,
            'mean_per_band': mean_per_band, 
            'pmin': pmin,
            'pmax': pmax,
                }

    with open(stats_file_path, 'wb') as f:
        pickle.dump(save_data, f)

    print('Statistics write sucessfully to ' + dir)

    #Load and normalize data
    train_loader = iter(handler.ms_data_loader(train_imgs, train_labels, batch_size = batch_size, img_shape = (64,64,num_bands), bands = bands)) # 3 for grb reduciton
    test_loader = iter(handler.ms_data_loader(val_images, val_labels, batch_size = batch_size, img_shape = (64,64,num_bands), bands = bands ))

    #region NN training

    #Initialize model
    if method == 'small':
        network = Autoencoder_small(num_bands, num_qubits)
    elif method == 'sscnet':
        network = Autoencoder_sscnet(num_bands, num_qubits)
    else:
        raise Exception('Error with architecture')
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    rmse_metric = MeanSquaredError(squared=False)
    ssim_metric = StructuralSimilarityIndexMeasure()
    psnr_metric = PeakSignalNoiseRatio() 

    #Train model and save results
    train_loss_list = []
    val_loss_list = []

    train_rmse_list = []
    val_rmse_list = []
    train_ssim_list = []
    val_ssim_list = []
    train_psnr_list = []
    val_psnr_list = []

    for epoch in range(num_epochs):

        total_loss = []
        
        train_targets = []
        train_predictions = []

        network.train()
        # for batch_idx in range(len(train_labels)):
            # Itera direttamente sui batch del DataLoader
        for batch_idx in range(num_batches):
            images, _ = next(train_loader) 
            outputs = network(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

            
            train_targets.append(images.detach())
            train_predictions.append(outputs.detach())

            print('\r Epoch %d ~ Batch %d (%d) ~ Loss %f ' % (
                epoch, batch_idx, num_batches-1, loss.item()), end='\t\t')
        
        network.eval()
        with torch.no_grad():
            val_loss = []
            targets = []
            predictions = []
            for batch_idx in range(len(val_images)):
                data, _ = next(test_loader)
                output = network(data)
                loss = criterion(output, data)

                val_loss.append(loss.item())

                targets.append(data)
                predictions.append(output)

        train_loss_list.append(sum(total_loss)/len(total_loss))
        val_loss_list.append(sum(val_loss)/len(val_loss))
    
        print('Training [{:.0f}%]\t Training Loss: {:.4f} Validation Loss: {:.4f}'.format(
            100. * (epoch + 1) / num_epochs, train_loss_list[-1], val_loss_list[-1]))
        
        if epoch % 3 == 1:

                # Concatenate all batches into single tensors
            targets = torch.cat(targets, dim=0)
            predictions = torch.cat(predictions, dim=0)

            train_targets = torch.cat(train_targets, dim=0)
            train_predictions = torch.cat(train_predictions, dim=0)

            rt = rmse_metric(train_predictions, train_targets)
            rv = rmse_metric(predictions, targets)
            st = ssim_metric(train_predictions, train_targets)
            sv = ssim_metric(predictions, targets)
            pt = psnr_metric(train_predictions, train_targets)
            pv = psnr_metric(predictions, targets)      

            print('RMSE: ', 'train: ' , rt,'val: ' , rv)
            print('SSIM: ', 'train: ' , st, 'val: ' , sv)
            print('PSNR: ', 'train: ' , pt, 'val: ' ,  pv)

            train_rmse_list.append(rt)
            val_rmse_list.append(rv)
            train_ssim_list.append(st)
            val_ssim_list.append(sv)
            train_psnr_list.append(pt)
            val_psnr_list.append(pv)

    save_data = {
                'train_loss_list': train_loss_list,
                'val_loss_list':  val_loss_list,
                'train_rmse_list' : train_rmse_list,
                'val_rmse_list' : val_rmse_list,
                'train_ssim_list' : train_ssim_list,
                'val_ssim_list' : val_ssim_list,
                'train_psnr_list' : train_psnr_list,
                'val_psnr_list' : val_psnr_list
                    }

    results_file_path = dir + 'results_{}.pkl'.format(method)

    with open(results_file_path, 'wb') as f:
        pickle.dump(save_data, f)

    torch.save(network.state_dict(), dir + "model_weights_{}.pth".format(method))

    print('Results write sucessfully to ' + dir)

    #endregion

if __name__ == "__main__":
    main()