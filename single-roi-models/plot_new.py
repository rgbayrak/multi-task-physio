import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import csv
import seaborn as sn
import pandas as pd
import matplotlib.ticker as plticker

warnings.filterwarnings("ignore")

# folds together
net_dir = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/single-roi-models/out/results/'
results_dir = os.listdir(net_dir)
all_results = sorted(results_dir, reverse=False)

# list_loss = ['l1_0', 'l1_0.0001', 'l1_0.3', 'l1_0.5', 'l1_0.7', 'l1_0.9999', 'l1_1']
allhr_data = []
allrv_data = []
labels = []
lrate = []
loss = []
rois = []
num_roi = []

list = []
for folder in all_results:
    # if 'l1_0.5' in folder:
    parts = folder.split('_')
    labels.append(parts[0])
    rois.append(parts[1])
    num_roi.append(parts[3])
    lrate.append(parts[5])
    loss.append(parts[7])
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
# plt.violinplot(all_data)

rv_means = np.mean(allrv_data, axis=1)
hr_means = np.mean(allhr_data, axis=1)

with open('findlab90_roisnames.txt', 'r') as fd:
    reader = csv.reader(fd)
    labels_roifindlab = []
    for row in reader:
        labels_roifindlab.append(row[0])

# for i in range(12):
#     labels_roifindlab.append(i+1)

wm_labels = ['Cingulum and associated tracts', ' Uncinate and middle temporal lobe tracts', 'Sensorimotor superficial '
             'white-matter system', 'Forceps minor system', 'Superior longitudinal fasciculus system',
             'visual superficial white matter system', 'inferior longitudinal fasciculus system',
             'inferior corticospinal tract', 'posterior cerebellar tracts', 'dorsal frontoparietal tracts',
             'deep frontal white matter', 'ventral frontoparietal tracts']

for wm in wm_labels:
    labels_roifindlab.append(wm)

# sort
desired_array = [int(numeric_string) for numeric_string in num_roi]
new = np.array(desired_array).argsort()
new_num_roi = np.array(desired_array)[new]
new_rv_means = np.array(rv_means)[new]
new_hr_means = np.array(hr_means)[new]

plt.scatter(np.arange(len(new_rv_means)), new_rv_means)
plt.scatter(np.arange(len(new_hr_means)), new_hr_means)
plt.xticks(np.arange(291), labels=new_num_roi, rotation='vertical', fontsize=6)
plt.grid()
plt.ylabel(r'${\mu}$' + ' Pearson Correlation')
plt.legend(['rv', 'hr'])
plt.show()
pass
# vecrv = np.reshape(allrv_data, [-1])
# vechr = np.reshape(allhr_data, [-1])
# vec = np.concatenate((vecrv, vechr), axis=None)
#
# frame_labels = []
# frame_lr = []
# frame_loss = []
# frame_type = []
# for i, l in enumerate(labels):
#     frame_labels.extend([l]*len(allrv_data[0]))
#     frame_lr.extend([lrate[i]]*len(allrv_data[0]))
#     frame_loss.extend([loss[i]]*len(allrv_data[0]))
#     frame_type.extend(['Respiration']*len(allrv_data[0]))
# del i, l
# for i, l in enumerate(labels):
#     frame_labels.extend([l]*len(allhr_data[0]))
#     frame_lr.extend([lrate[i]]*len(allhr_data[0]))
#     frame_loss.extend([loss[i]]*len(allhr_data[0]))
#     frame_type.extend(['Heart Rate']*len(allhr_data[0]))
#
#
# df_data = {'Pearson Correlation': vec, 'Model Architectures': frame_labels, 'Learning Rate': frame_lr, 'Lambda RV': frame_loss,
#            'Physio Type': frame_type}
# df = pd.DataFrame(data=df_data)
#
# sn.set(style="whitegrid")
#
# # plt.rcParams["figure.figsize"] = (15,5)
# plt.rcParams["figure.figsize"] = (8,4)
# # plt.ylim(-0.5, 1)
# SMALL_SIZE = 10
# MEDIUM_SIZE = 14
# BIGGER_SIZE = 17
# plt.rc('font', weight='bold')
# # plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
# plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
# plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE, color='black')    # fontsize of the tick labels
#
# # ax.set_facecolor((0.212, 0.212, 0.227))
# # dark mode
# # plt.style.use('dark_background')
# # plt.rcParams['axes.facecolor'] = (0.212, 0.212, 0.227)
# # plt.rcParams['savefig.facecolor'] = (0.212, 0.212, 0.227)
# # sn.stripplot(x='Model Architectures', y='Pearson Correlation', data=df,
# #                    palette=["gray"])
#
# # df = df.loc[df['Lambda RV'] == '0.5']
# #
# # ax1 = sn.swarmplot(x='Lambda RV', y='Pearson Correlation', data=df, split=True, hue='Physio Type',
# #                     size=4, dodge=True)
# #
# # handles, labels = ax1.get_legend_handles_labels()
# # #
# ax = sn.lineplot(x='Lambda RV', y='Pearson Correlation', data=df, style='Model Architectures', hue='Physio Type', estimator=np.median, markers=True, ci='sd')
# handles, labels = ax.get_legend_handles_labels()
# #
# # ax2 = sn.violinplot(x='Lambda RV', y='Pearson Correlation', data=df, split=False, hue='Physio Type',
# #                     scale="count")
# #
# # handles, labels = ax2.get_legend_handles_labels()
#
# # ['dodgerblue', 'palevioletred', 'mediumseagreen', 'gold', 'slateblue']
#
# # ax.grid(color='gray', linestyle='-', linewidth=.8)
# plt.legend(loc='lower left', ncol=5, handles=handles, bbox_to_anchor=(0.0, 1.00))
# # ax.set_title('RV')
#
# # loc = plticker.MultipleLocator(base=0.1) # this locator puts ticks at regular intervals
# # ax.yaxis.set_major_locator(loc)
# # plt.show()
# # plt.savefig('/home/bayrakrg/neurdy/pycharm/multi-task-physio/single-roi-models/figures/contribution.png', bbox_inches='tight', dpi=300)
# plt.show()
# print()