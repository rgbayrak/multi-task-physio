import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sn
import pandas as pd
warnings.filterwarnings("ignore")

plot_these = ['hr_corr_splits_test_fold_0.txt', 'hr_corr_splits_test_fold_1.txt',
              'hr_corr_splits_test_fold_2.txt', 'hr_corr_splits_test_fold_3.txt',
              'hr_corr_splits_test_fold_4.txt']

# folds together
net_dir = '/home/bayrakrg/neurdy/pycharm/HR/out/results/'
results_dir = os.listdir(net_dir)
all_results = sorted(results_dir, reverse=False)

all_data = []
labels = []
parcellation = []
for folder in all_results:
    labels.append(folder.split('_')[0])
    parc_part = folder.split('_')[3]
    parcellation.append(parc_part)
    models = os.path.join(net_dir, folder)
    data = []
    for file in plot_these:
        path = os.path.join(models, file)
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data.append(float(line.rstrip('\n')))
    all_data.append(data)
# plt.boxplot(all_data)

vec = np.reshape(all_data, [-1])
frame_labels = []
frame_parc = []
for i, l in enumerate(labels):
    frame_labels.extend([l]*len(all_data[0]))
    frame_parc.extend([parcellation[i]]*len(all_data[0]))

df_data = {'Pearson Correlation': vec, 'Model Architectures': frame_labels, 'parc': frame_parc}
df = pd.DataFrame(data=df_data)

# sn.set(style="whitegrid")

plt.rcParams["figure.figsize"] = (8,7)
SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 17
plt.rc('font', weight='bold')
# plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE, color='black')    # fontsize of the tick labels

# ax.set_facecolor((0.212, 0.212, 0.227))
# dark mode
plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = (0.212, 0.212, 0.227)
plt.rcParams['savefig.facecolor'] = (0.212, 0.212, 0.227)
# sn.stripplot(x='Model Architectures', y='Pearson Correlation', data=df,
#                    palette=["gray"])

ax = sn.violinplot(x='Model Architectures', y='Pearson Correlation', data=df,  kind='violin', hue='parc',  scale="count", inner="quartile",
                   palette=['dodgerblue', 'palevioletred', 'mediumseagreen'], order=["Bi-LSTM"])



# ['dodgerblue', 'palevioletred', 'mediumseagreen', 'gold', 'slateblue']

# ax.grid(color='gray', linestyle='-', linewidth=.8)
ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.00), ncol=5)


plt.savefig('/home/bayrakrg/neurdy/pycharm/HR/hr.png', bbox_inches='tight', dpi=300)
plt.show()
print()