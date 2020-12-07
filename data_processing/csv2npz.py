from data_processing.challenge_dataset import OzeDataset as COD
import numpy as np
from data_processing.dataset import OzeDataset

# x_train_path = '/home/bayrakrg/Downloads/x_train_LsAZgHU.csv'
# y_train_path = '/home/bayrakrg/Downloads/y_train_EFo1WyE.csv'
#
# data = COD(x_train_path, y_train_path)
# np.savez('dataset.npz', R=data.R, Z=data.Z, X=data.X)

## TEST ##
tmp = OzeDataset('dataset.npz')
pass