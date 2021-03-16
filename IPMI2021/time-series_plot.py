import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sn
import pandas as pd
from scipy.io import loadmat
out_dir = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/out_lemon/lemon-results/Bi-LSTM_all4_lr_0.001_l1_0.5/test/test_fold_0-old/'
files = ['rv_pred.csv', 'rv_target.csv']
# id_corr = loadmat('/home/bayrakrg/neurdy/pycharm/neuroimg2020/RV/out/results/cnn_findlab90/id_corr.mat')
id_corr = 'pred_scans'

all_data = []
for file in files:
    path = os.path.join(out_dir, file)
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lst = line.rstrip('\n').split(',')
            tmp = []
            tmp.append([(float(i)) for i in lst])
            data.append(tmp)
    for j, d in enumerate(data):
        # tmp = (np.array(data[j][0]) - np.array(data[j][0]).mean(axis=0)) / np.array(data[j][0]).std(axis=0)  # z-score normalization
        all_data.append(d)

fig, axs = plt.subplots(7, 1, figsize=(20,8))

axs[0].plot(all_data[0][0], label='r = -0.0326')
axs[0].plot(all_data[7][0], linestyle='--')
axs[1].plot(all_data[1][0], label='r = 0.4635')
axs[1].plot(all_data[8][0], linestyle='--')
axs[2].plot(all_data[2][0], label='r = -0.2310')
axs[2].plot(all_data[9][0], linestyle='--')
axs[3].plot(all_data[3][0], label='r = 0.4325')
axs[3].plot(all_data[10][0], linestyle='--')
axs[4].plot(all_data[4][0], label='r = 0.0771')
axs[4].plot(all_data[11][0], linestyle='--')
axs[5].plot(all_data[5][0], label='r = 0.0028')
axs[5].plot(all_data[12][0], linestyle='--')

axs[0].tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=True,         # ticks along the top edge are off
    labelbottom=False,
    labeltop=True) # labels along the bottom edge are off
axs[0].grid(color='gray', linestyle=':', linewidth=.8)
axs[0].legend(loc='upper left', bbox_to_anchor=(0.0, 1.00), ncol=2)

for i in range(6):
    axs[i].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    axs[i].grid(color='gray', linestyle=':', linewidth=.8)
    axs[i].legend(loc='upper left', bbox_to_anchor=(0.00, 1.00), ncol=2)

axs[6].plot(all_data[6][0], label='r = -0.04318')
axs[6].plot(all_data[13][0], linestyle='--')
axs[6].legend(loc='lower left', bbox_to_anchor=(0.00, 0.00), ncol=2)
axs[6].grid(color='gray', linestyle=':', linewidth=.8)

plt.show()
pass
