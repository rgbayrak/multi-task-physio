'''
Author: Roza G. Bayrak
THIS SCRIPT IS USED TO VISUALIZE SINGLE ROI MODELS AS SQUARES
FOR IPMI 2021 SUBMISSION
'''

import os
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import warnings
import csv

path_labels = '/home/bayrakrg/neurdy/pycharm/atlas_processing/schaefer_gradient_info.csv'
with open(path_labels, 'r') as f:
    content = f.readlines()



