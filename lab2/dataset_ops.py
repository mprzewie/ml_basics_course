import numpy as np
import matplotlib.pyplot as plt
import imageio as io
from typing import Tuple
import cv2

# for loading datasets from .png files

def load_dataset(
    img_path: str, 
    space_size: Tuple[int, int]=(100, 100)
) -> Tuple[np.ndarray, np.ndarray]:
    
    ds = io.imread(img_path)
    ds = cv2.resize(ds, dsize=space_size, interpolation=cv2.INTER_NEAREST)
    flat_ds = ds.reshape(-1, ds.shape[2])
    colors = np.unique(flat_ds, axis=0)[:-1]
    X = []
    y = []

    for (i, c) in enumerate(colors):
        X_of_color = np.argwhere((ds == c).prod(axis=2))[:, 0:2]
        X.extend(X_of_color)
        y.extend([i] * X_of_color.shape[0])
        X_tmp = np.array(X)
        y_tmp =np.array(y) 

    X = np.array(X)
    y = np.array(y)
    return X, y


def visualize_dataset(X: np.ndarray, y: np.ndarray):
    n_classes = len(np.unique(y))
    colors = ['k', 'b', 'r', 'c', 'm', 'y', 'k', 'w']
    assert(n_classes <= len(colors))
    
    for c in range(n_classes):
        X_c = X[y==c]
        plt.scatter(X_c[:, 0], X_c[:, 1], color=colors[c], marker=".")
    plt.show()