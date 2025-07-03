import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from glob import glob
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

import argparse


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--N",             help="Number of qubits", required=False, type=int, default=12)
    parser.add_argument("--numbands",      help="Number of bands", required=False, type=int, default=3)
    parser.add_argument("--method",        help="Type of autoencoder: 'small' or 'sscnet' ", required=False, type=str, default='small', choices=['small', 'sscnet', 'unet'])
    parser.add_argument("--split",         help="Dataset split", required=False, type=str, default='small', choices=['train', 'val', 'test'])


    args = parser.parse_args()

    # Base directory containing the classes
    method = args.method
    qubits = args.N
    bands  = args.numbands
    split  = args.split


    base_path = os.path.join(f'{method}_q{qubits}_b{bands}', split)

    classes = ['River', 'SeaLake']
    data    = []
    labels  = []

    # Load latent_space vectors from each .npz file
    for label, cls in enumerate(classes):
        npz_files = glob(os.path.join(base_path, cls, '*.npz'))
        for f in npz_files:
            loaded = np.load(f)
            latent = loaded['latent_space']
            data.append(latent)
            labels.append(label)

    # Convert to NumPy arrays
    X = np.vstack(data)
    y = np.array(labels)

    colors = ['blue', 'green']

    # === 2D PCA Plot ===
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X)

    save_results = 'clustering_results'
    os.makedirs(save_results, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(classes):
        plt.scatter(X_pca_2d[y == i, 0], X_pca_2d[y == i, 1], label=cls, alpha=0.7, color=colors[i])

    plt.title('2D PCA of Latent Space Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save 2D plot
    safe_base = base_path.replace(os.sep, '_').strip('_')
    filename_2d = os.path.join(save_results, f'pca2d_plot_{safe_base}.pdf')
    plt.savefig(filename_2d)
    print(f'2D PCA figure saved as: {filename_2d}')
    plt.show()

    # === 3D PCA Plot ===
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for i, cls in enumerate(classes):
        ax.scatter(X_pca_3d[y == i, 0], X_pca_3d[y == i, 1], X_pca_3d[y == i, 2],
                label=cls, alpha=0.7, color=colors[i])

    ax.set_title('3D PCA of Latent Space Features')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.legend()
    plt.tight_layout()


    # Save 3D plot
    filename_3d = os.path.join(save_results, f'pca3d_plot_{safe_base}.pdf')
    plt.savefig(filename_3d)
    print(f'3D PCA figure saved as: {filename_3d}')
    plt.show()
