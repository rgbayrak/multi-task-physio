import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sn
import pandas as pd
from matplotlib import rcParams
import matplotlib.ticker as plticker

warnings.filterwarnings("ignore")

# folds together
net_dir = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/out_select/'
results_dir = os.listdir(net_dir)
all_results = sorted(results_dir, reverse=False)

allhr_data = []
allrv_data = []
labels = []
lrate = []
loss = []
percent = []
rois = []

list = []
for folder in all_results:
    parts = folder.split('_')
    labels.append(parts[0])
    rois.append(parts[1])
    lrate.append(parts[3])
    loss.append(parts[5])
    # percent.append(parts[7])
    # fold_dir = os.path.join(net_dir, folder)
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
# plt.violinplot(allrv_data)
# plt.violinplot(allhr_data)

vecrv = np.reshape(allrv_data, [-1])
vechr = np.reshape(allhr_data, [-1])
vec = np.concatenate((vecrv, vechr), axis=None)

frame_labels = []
frame_lr = []
frame_loss = []
frame_type = []
# frame_percent = []
frame_roi = []
# for i, l in enumerate(labels):
#     frame_labels.extend([l]*len(vecrv))
#     frame_lr.extend([lrate[i]]*len(vecrv))
#     frame_loss.extend([loss[i]]*len(vecrv))
#     frame_type.extend(['Respiration']*len(vecrv))
# del i, l
# for i, l in enumerate(labels):
#     frame_labels.extend([l]*len(vechr))
#     frame_lr.extend([lrate[i]]*len(vechr))
#     frame_loss.extend([loss[i]]*len(vechr))
#     frame_type.extend(['Heart Rate']*len(vechr))

for i, l in enumerate(labels):
    frame_labels.extend([l] * len(allrv_data[0]))
    frame_lr.extend([lrate[i]] * len(allrv_data[0]))
    frame_loss.extend([loss[i]] * len(allrv_data[0]))
    # frame_percent.extend([percent[i]] * len(allrv_data[0]))
    frame_type.extend(['Respiration'] * len(allrv_data[0]))
    frame_roi.extend([rois[i]] * len(allrv_data[0]))
del i, l
for i, l in enumerate(labels):
    frame_labels.extend([l] * len(allhr_data[0]))
    frame_lr.extend([lrate[i]] * len(allhr_data[0]))
    frame_loss.extend([loss[i]] * len(allhr_data[0]))
    # frame_percent.extend([percent[i]] * len(allrv_data[0]))
    frame_type.extend(['Heart Rate'] * len(allhr_data[0]))
    frame_roi.extend([rois[i]] * len(allhr_data[0]))



df_data = {'Pearson Correlation': vec, 'Model Architectures': frame_labels, 'Learning Rate': frame_lr, 'Lambda RV': frame_loss,
           'Physio Type': frame_type, 'Input': frame_roi}
# df_data = {'Pearson Correlation': vec, 'Model Architectures': frame_labels, 'Learning Rate': frame_lr, 'Lambda RV': frame_loss,
#            'Physio Type': frame_type, 'Missing %': frame_percent}
df = pd.DataFrame(data=df_data)

sn.set(style="whitegrid")

from matplotlib import rc,rcParams

plt.rcParams["figure.figsize"] = (4,4)
# plt.ylim(-0.2, 1)
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 17
# plt.rc('font', weight='bold')
# plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE, color='black')    # fontsize of the tick labels
plt.rc('font', weight='bold')
# ax.set_facecolor((0.212, 0.212, 0.227))
# dark mode
# plt.style.use('dark_background')
# plt.rcParams['axes.facecolor'] = (0.212, 0.212, 0.227)
# plt.rcParams['savefig.facecolor'] = (0.212, 0.212, 0.227)
sn.stripplot(x='Model Architectures', y='Pearson Correlation', data=df,
                   palette=["gray"])

# list_loss = ['l1_0', 'l1_0.0001', 'l1_0.3', 'l1_0.5', 'l1_0.7', 'l1_0.9999', 'l1_1']
# df = df[~df['Lambda RV'].isin(list_loss)] did not work
#
ax1 = sn.swarmplot(x='Input', y='Pearson Correlation', data=df, split=True,
                   hue='Physio Type', palette=['#0D5901', '#B3000C'], size=8, dodge=True)
# ax = sn.boxplot(x='Input', y='Pearson Correlation', data=df, hue='Physio Type', palette=['#0D5901', '#B3000C'],
#         showcaps=False,boxprops={'facecolor':'None'},
#         showfliers=False,whiskerprops={'linewidth':0})

#
# handles, labels = ax1.get_legend_handles_labels()
# df_hr = df[df['Physio Type']=='Heart Rate']
# ax = sn.lineplot(x='Missing %', y='Pearson Correlation', data=df_hr, style='Lambda RV', hue='Lambda RV',
#                  estimator=np.median, markers=True, ci=75, palette=['#0D5901', '#B3000C'])
# ax = sn.lineplot(x='Missing %', y='Pearson Correlation', data=df_hr, style='Lambda RV',
#                  estimator=np.median, markers=True, ci=75, palette=['#0D5901', '#B3000C'])
# handles, labels = ax.get_legend_handles_labels()
# #
# plt.xticks(np.arange(len(percent)), labels=percent, rotation='horizontal', fontsize=12)
plt.yticks(weight='bold')
# ax1.xaxis.set_tick_params(labelsize=2)
plt.ylabel('Pearson Correlation', fontsize=15)
# plt.xlabel(r'${\lambda}_{RV}$')
# plt.xlabel('Missing Data by %')
# ['dodgerblue', 'palevioletred', 'mediumseagreen', 'gold', 'slateblue']

# ax.grid(color='gray', linestyle='-', linewidth=.8)
# plt.legend(loc='lower left', ncol=10, handles=handles, bbox_to_anchor=(0.0, 1.00))
plt.legend(loc='lower left', ncol=10, bbox_to_anchor=(0.0, 1.00))
# ax.set_title('RV')
# plt.grid()
# loc = plticker.MultipleLocator(base=0.1) # this locator puts ticks at regular intervals
# ax.yaxis.set_major_locator(loc)

# plt.savefig('/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/figures/nih.png', bbox_inches='tight', dpi=300)
plt.show()
print()
