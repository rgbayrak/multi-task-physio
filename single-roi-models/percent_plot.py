import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sn
import pandas as pd
import matplotlib.ticker as plticker

warnings.filterwarnings("ignore")

# folds together
net_dir = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/single-roi-models/out/results_delete/'
results_dir = os.listdir(net_dir)
all_results = sorted(results_dir, reverse=False)

# list_loss = ['l1_0', 'l1_0.0001', 'l1_0.3', 'l1_0.5', 'l1_0.7', 'l1_0.9999', 'l1_1']
allhr_data = []
allrv_data = []
labels = []
lrate = []
loss = []
rois = []
percent = []

list = []
for folder in all_results:
    # if 'l1_0.5' in folder:
    parts = folder.split('_')
    labels.append(parts[0])
    rois.append(parts[2])
    percent.append(parts[2])
    lrate.append(parts[4])
    loss.append(parts[6])
    fold_dir = os.path.join(net_dir, folder, 'test')
    rv_data = []
    hr_data = []
    for o in os.listdir(fold_dir):
        if os.path.isdir(os.path.join(fold_dir, o)):
            files = os.listdir(os.path.join(fold_dir, o))
            for file in files:
                if 'pred_scans' in file:
                    path = os.path.join(fold_dir, o, file)
                    with open(path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            rv_data.append(float(line.strip('\n').split(',')[2]))
                            hr_data.append(float(line.strip('\n').split(',')[3]))

    allrv_data.append(rv_data)
    allhr_data.append(hr_data)


rv_means = np.array([np.mean(ri) for ri in allrv_data])
hr_means = np.array([np.mean(hi) for hi in allhr_data])

rv_medians = np.array([np.median(ri) for ri in allrv_data])
hr_medians = np.array([np.median(hi) for hi in allhr_data])

rv_std = np.array([np.std(ri) for ri in allrv_data])
hr_std = np.array([np.std(hi) for hi in allhr_data])

# # sort based on rv
# idx = rv_means.argsort()
#
# # # sort based on hr
# idx = hr_means.argsort()

# sort by validation loss
# idx = np.array(val_loss).argsort()

# # sort based on the difference
# idx = (rv_means-hr_means).argsort()

# sort based on percentage
idx = np.array([int(x.strip('%')) for x in percent]).argsort()

# val_loss = np.array(val_loss)[idx]
rv_means = np.array(rv_means)[idx]
hr_means = np.array(hr_means)[idx]
# labels = np.array(labels)[idx]
# ids = np.array(ids)[idx]
# atlas = np.array(atlas)[idx]
percent = np.array(percent)[idx]
# x = np.log([1, 2, 4, 10, 20, 100])
# x = [1, 5, 10, 20, 40, 100]
x = [1, 2, 4, 10, 20, 100]
plt.scatter(x, rv_means, marker='*', s=45, c='#0D5901')
plt.scatter(x, hr_means, marker='*', s=45, c='#B3000C')
plt.plot(x, rv_means, c='#0D5901')
plt.plot(x, hr_means, c='#B3000C')
# plt.xticks(np.arange(len(rv_means)), labels=labels, rotation='horizontal', fontsize=9)
plt.xticks([1,10,20,100], labels=['1%', '10%', '20%', '100%'], rotation='horizontal', fontsize=10)
# plt.ylim([-0.01, 0.3])
plt.ylim([0.0, 0.8])
plt.ylabel(r'${\mu}$' + ' Pearson Correlation')
plt.legend(['rv', 'hr'], loc='lower right')
plt.grid()
plt.show()

pass