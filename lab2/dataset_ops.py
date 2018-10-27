import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import cv2

# for loading datasets from .png files

def load_dataset(
    img_path: str, 
    space_size: Tuple[int, int]=(100, 100),
    dropout: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    
    ds = cv2.imread(img_path)
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
    keep = np.random.rand(X.shape[0]) > dropout
    return X[keep], y[keep]

def sliced_dataset_name(prefix: str, n_slices: int, x0: int, x1: int) -> str:
    return f"{prefix}.{n_slices}.{x0}.{x1}"

def slice_dataset(X: np.ndarray, y: np.ndarray, n_slices: int, space_size: Tuple[int, int]=(100, 100)) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    d0 = space_size[0] / n_slices
    d1 = space_size[1] / n_slices
    n_classes = len(np.unique(y))
    result = {}
    for x0 in range(n_slices):
        x0_low = x0 * d0
        x0_high = (x0 + 1) * d0
        for x1 in range(n_slices):
            x1_low = x1 * d1
            x1_high = (x1 + 1) * d1
            condition = (
                (x0_low <= X[:, 0]) * 
                (X[:, 0] <= x0_high) *
                (x1_low <= X[:, 1]) * 
                (X[:, 1] <= x1_high)
            )
            result[(x0, x1)] = X[condition], y[condition]
    return result

def visualize_dataset(X: np.ndarray, y: np.ndarray, n_classes: Optional[int]=None, space_size: Tuple[Tuple[int,int], Tuple[int,int]]=((0,100), (0,100))):
    n_classes = len(np.unique(y)) if n_classes is None else n_classes
    colors = ['k', 'b', 'r', 'c', 'm', 'y', 'k', 'w']
    assert(n_classes <= len(colors))
    
    for c in range(n_classes):
        X_c = X[y==c]
        plt.scatter(X_c[:, 0], X_c[:, 1], color=colors[c], marker=".")
    plt.xlim(space_size[0][0], space_size[0][1])
    plt.ylim(space_size[1][0], space_size[1][1])
#     plt.show()