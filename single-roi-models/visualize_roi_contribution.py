'''
Author: Roza G. Bayrak
THIS SCRIPT IS USED TO VISUALIZE SINGLE ROI MODELS FOR IPMI 2021 SUBMISSION
'''

import os
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import warnings
import csv

# folds together
net_dir = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/single-roi-models/out/results/'
results_dir = os.listdir(net_dir)

rois = []
atlases = []
for folder in results_dir:
    if 'tianschaefer' in folder:
        parts = folder.split('_')
        if not atlases:
            atlases.append(parts[1])
        rois.append(parts[3])
pass

