import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sn
import pandas as pd
from scipy.io import loadmat
out_dir = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/transformer/out/rresults/'
model = 'Att_schaefertractsegtianaan_lr_0.0001_l1_0.3_loss_pearson'
# files = ['measured_wide.csv', 'predictions_wide.csv']
files = ['rv_target.csv', 'rv_pred.csv']
id_filename = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/transformer/out/rresults/test/pred_scans.csv'

norm_data = []
for file in files:
    path = os.path.join(out_dir, model, 'test/test_fold_0', file)
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lst = line.rstrip('\n').split(',')
            tmp = []
            tmp.append([(float(i)) for i in lst])
            data.append(tmp)
    for j, d in enumerate(data):
        tmp = (np.array(data[j][0]) - np.array(data[j][0]).mean(axis=0)) / np.array(data[j][0]).std(axis=0)  # z-score normalization
        norm_data.append(tmp)

# with open('/home/bayrakrg/Data/preproced_HCP_social/id.csv', 'r') as f:
#     x = f.readlines()
#     y = [a.rstrip('\n') for a in x]

for n, scan in enumerate(norm_data):
    if n < 600:
        fig, axs = plt.subplots(figsize=(12,2))

        axs.plot(norm_data[n], label='Measured')
        axs.plot(norm_data[300+n], label='Predicted')

        axs.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True,
            labeltop=False) # labels along the bottom edge are off
        axs.grid(color='gray', linestyle=':', linewidth=.8)
        axs.legend(loc='upper left', ncol=2)
        # plt.title(str(z[n]))

        # plt.show()

        # figname = '/home/bayrakrg/Data/preproced_HCP_social/modeled_RV_QA_figs/' + str(y[n]) + '-' + str(round(float(z[n]), 3)) + '.png'
        # plt.savefig(figname, dpi=300)
        pass
