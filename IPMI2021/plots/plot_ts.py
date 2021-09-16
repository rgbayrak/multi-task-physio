import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sn
import pandas as pd
from scipy.io import loadmat
out_dir = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/out_nih/results/Bi-LSTM_schaefertractsegtianaan_lr_0.001_l1_0.5/test/test_fold_0/'
files = ['rv_pred.csv', 'rv_target.csv', 'hr_pred.csv', 'hr_target.csv']
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

fig, axs = plt.subplots(4, 1, figsize=(20,8))

# # select scan to plot
# a = loadmat('/bigdata/HCP_rest/power+xifra/resting_min+prepro/bpf-ds/physio/HR_filt_ds/194645_rfMRI_REST1_RL_hr_filt_ds.mat')
# mu = np.mean(a['hr_filt_ds'])

n = 0
axs[0].plot(np.arange(len(all_data[0+n][0]))*1.44, all_data[0+n][0],  linestyle='--', label='predicted')
axs[0].set_ylabel('r = 0.748')
axs[0].plot(np.arange(len(all_data[22+n][0]))*1.44, all_data[22+n][0], label='target')
# axs[1].plot(list(np.array(all_data[635][0]) + mu), label='r = 0.748')
# axs[1].plot(list(np.array(all_data[935][0]) + mu), linestyle='--')
axs[1].plot(np.arange(len(all_data[9+n][0]))*1.44, all_data[9+n][0], linestyle='--', label='predicted')
axs[1].set_ylabel('r = 0.713')
axs[1].plot(np.arange(len(all_data[31+n][0]))*1.44, all_data[31+n][0], label='target')
axs[2].plot(np.arange(len(all_data[13+n][0]))*1.44, all_data[13+n][0], linestyle='--', label='predicted')
axs[2].set_ylabel('r = 0.568')
axs[2].plot(np.arange(len(all_data[35+n][0]))*1.44, all_data[35+n][0], label='target')
axs[3].plot(np.arange(len(all_data[20+n][0]))*1.44, all_data[20+n][0], linestyle='--', label='predicted')
axs[3].set_ylabel('r = 0.432')
axs[3].plot(np.arange(len(all_data[42+n][0]))*1.44, all_data[42+n][0], label='target')

for i in range(4):
    axs[i].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        labelbottom=False,
        labeltop=True) # labels along the bottom edge are off
    axs[i].grid(color='gray', linestyle=':', linewidth=.8)
    axs[i].legend(loc='upper left', bbox_to_anchor=(0.0, 1.00), ncol=2)

# for i in range(4):
#     axs[i].tick_params(
#         axis='x',          # changes apply to the x-axis
#         which='both',      # both major and minor ticks are affected
#         bottom=False,      # ticks along the bottom edge are off
#         top=False,         # ticks along the top edge are off
#         labelbottom=False) # labels along the bottom edge are off
#     axs[i].grid(color='gray', linestyle=':', linewidth=.8)
#     axs[i].legend(loc='upper left', bbox_to_anchor=(0.00, 1.00), ncol=2)
#
# axs[2].plot(all_data[6][0], label='r = -0.04318')
# axs[6].plot(all_data[13][0], linestyle='--')
# axs[2].legend(loc='lower left', bbox_to_anchor=(0.00, 0.00), ncol=2)
# axs[2].grid(color='gray', linestyle=':', linewidth=.8)

plt.show()
pass
