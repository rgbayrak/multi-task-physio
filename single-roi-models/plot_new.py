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
net_dir = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/single-roi-models/out/results_delete/'
all_results = os.listdir(net_dir)

# get the file names in the order that they were fed in to the networks
path_labels = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/single-roi-models/all4.txt'
with open(path_labels, 'r') as f:
    fname = f.readlines()

for k in range(len(fname)):
    fname[k] = [str(x) for x in fname[k].strip('\n').split(',')]

# list_loss = ['l1_0', 'l1_0.0001', 'l1_0.3', 'l1_0.5', 'l1_0.7', 'l1_0.9999', 'l1_1']
allhr_data = []
allrv_data = []
labels = []
atlas = []
ids = []

list = []
val_loss = []
for j, z in enumerate(fname):
    for folder in all_results:
        if folder.split('roi_')[1] == z[0]:
            print(z[0] + ': ' + folder)
            # load validation file  to get the best model accuracy
            val_path = net_dir + folder + '/validate_loss_split_train_fold_0.txt'
            with open(val_path, 'r') as f:
                value = f.readlines()

            for i in range(len(value)):
                value[i] = [float(x) for x in value[i].strip('\n').split(',')]

            parts = folder.split('_')
            labels.append(folder.replace('Bi-LSTM_'+parts[1]+'_roi_', ''))
            atlas.append(parts[1])
            val_loss.append(np.nanmax(value))
            ids.append(j)

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

rv_means = np.array([np.mean(ri) for ri in allrv_data])
hr_means = np.array([np.mean(hi) for hi in allhr_data])

# # sort based on rv
# idx = rv_means.argsort()
#
# # # sort based on hr
# idx = hr_means.argsort()

# sort by validation loss
# idx = np.array(val_loss).argsort()

# sort based on the difference
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

plt.scatter(np.arange(len(rv_means)), rv_means, marker='*', s=64, c='#0D5901')
plt.scatter(np.arange(len(hr_means)), hr_means, marker='*', s=64, c='#B3000C')
plt.plot(rv_means)
plt.plot(hr_means)
plt.xticks(np.arange(len(rv_means)), labels=labels, rotation='vertical', fontsize=6)
plt.xticks(np.arange(len(rv_means)), labels=percent, rotation='vertical', fontsize=6)
# plt.ylim([-0.01, 0.5])
plt.ylim([-0.01, 1])
plt.ylabel(r'${\mu}$' + ' Pearson Correlation')
plt.legend(['rv', 'hr'])
plt.show()
plt.grid()
pass

# dict = []
# for ii, label in enumerate(labels):
#     dict.append({'id': ids[ii], 'label': label, 'atlas': atlas[ii], 'rv': str(rv_means[ii]), 'hr': str(hr_means[ii]), 'val': str(val_loss[ii])})
#
# csv_columns = ['id', 'label', 'atlas', 'rv', 'hr', 'val']
# csv_file = "info.csv"
# try:
#     with open(csv_file, 'w') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
#         writer.writeheader()
#         for data in dict:
#             writer.writerow(data)
# except IOError:
#     print("I/O error")

# pass


#######################################################################################################################
# Data Frame

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