import copy
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from skimage import io, transform

def get_roi_len(dirs):
    roi_path = "/data/rfMRI/"
    roi_len = 0
    for dir in dirs:
        files = os.listdir(roi_path + dir + '_mat')
        roi = loadmat(roi_path + dir + '_mat/' + files[0])
        key = [x for x in list(roi.keys()) if x.startswith('roi_dat')][0]
        roi_len += roi[key].shape[1]
    return roi_len

def get_sub(path):
    fp = open(path, 'r')
    sublines = fp.readlines()
    gm_fold = []
    hr_fold = []
    ngm_fold = []

    for subline in sublines:
        gm_fold.append(subline)
        ngm_fold.append(subline.replace('rois', 'rois_wmcsf'))
        hr_fold.append(subline.replace('.mat', '_hr_filt_ds.mat').replace('rois_', ''))
    fp.close()
    return gm_fold, hr_fold, ngm_fold


def get_dictionary(fold):
    roi_path = os.path.join("/data/rfMRI/")
    hr_path = "/data/HR_filt_ds/"
    fold_path = os.path.join("/home/bayrakrg/neurdy/pycharm/HR/k_fold_files/", fold)
    gm_fold, hr_fold, ngm_fold = get_sub(fold_path)


    # # LOOK AT YOUR DATA
    # x = os.path.join(rv_path, 'RV_filtds_983773_3T_rfMRI_REST1_RL.mat')
    # rv = loadmat(x)
    # rv.keys()
    # type(ROI['roi_dat']), ROI['roi_dat'].shape
    # type(ROI['roi_inds']), ROI['roi_inds'].shape
    # type(rv['rv_filt_ds']), rv['rv_filt_ds'].shape
    # type(rv['tax']), rv['tax'].shape

    data = {}
    for i, d in enumerate(gm_fold):
        subdir_parts = gm_fold[i].rstrip(".mat").split('_')
        subject_id = subdir_parts[1]
        # print("{}".format(subject_id))

        clust_list = os.listdir(roi_path)
        if subject_id not in data:
            data[subject_id] = {clust_list[1]: [roi_path + clust_list[1] + '/' + ngm_fold[i].rstrip('\n')],
                                clust_list[0]: [roi_path + clust_list[0] + '/' + d.rstrip('\n')],
                                'HR_filt_ds': [hr_path + hr_fold[i].rstrip('\n')]}
        else:
            if clust_list[0] and clust_list[1] not in data[subject_id]:
                data[subject_id][clust_list[1]] = [roi_path + clust_list[1] + '/' + ngm_fold[i].rstrip('\n')]
                data[subject_id][clust_list[0]] = [roi_path + clust_list[0] + '/' + d.rstrip('\n')]
                data[subject_id]['HR_filt_ds'] = [hr_path + hr_fold[i].rstrip('\n')]
            else:
                data[subject_id][clust_list[1]].append(roi_path + clust_list[1] + '/' + ngm_fold[i].rstrip('\n'))
                data[subject_id][clust_list[0]].append(roi_path + clust_list[0] + '/' + d.rstrip('\n'))
                data[subject_id]['HR_filt_ds'].append(hr_path + hr_fold[i].rstrip('\n'))

    # get the paths
    subj_excl = []
    for subj in data:
        paths = data[subj]['findlab_mat']
        # keep tract of the subjects that do not have all 4 scans
        if len(paths) == 4:
            subj_excl.append(subj)

        scan_order = []
        for path in paths:
            scan_order.append(path.lstrip('/data/rfMRI/findlab_mat/rois_').rstrip('.mat'))

        for k in data[subj]:
            new_paths = []
            for scan_id in scan_order:
                for path in data[subj][k]:
                    if scan_id in path:
                        new_paths.append(path)
                        break
            data[subj][k] = new_paths

    # print(list(data.keys())) # subject_ids
    return data


class data_to_tensor():
    """ From pytorch example"""

    def __init__(self, data, roi_clust, transform=None):
        # go through all the data and load them in
        # start with one worker
        # as soon as I pass to the data loader it is gonna create a copy depending on the workers (threads)
        # copy of the data for each worker (for more heavy duty data)
        # random data augmentation usually needs multiple workers
        self.data = copy.deepcopy(data)
        self.paths = copy.deepcopy(data)
        self.idx_list = []

        for subj in self.data.keys():
            for folder in self.data[subj]:
                for i, val in enumerate(self.data[subj][folder]):
                    self.data[subj][folder][i] = loadmat(val)

        # make sure in get_item that we see all data by
        for subj in self.data.keys():
            for i, val in enumerate(self.data[subj]['HR_filt_ds']):
                self.idx_list.append([subj, i])

        self.keys = list(self.data.keys())  # so, we just do it once
        self.transform = transform
        self.roi_clust = roi_clust

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # load on the fly
        single = self.data[self.idx_list[idx][0]]  # passing the subject string to get the other dictionary
        single_paths = self.paths[self.idx_list[idx][0]]
        hr_path = single_paths['HR_filt_ds'][self.idx_list[idx][1]]
        gm = single[self.roi_clust + '_mat'][self.idx_list[idx][1]]['roi_dat'][20:580, :]
        ngm = single['wmcsf_mat'][self.idx_list[idx][1]]['roi_dat_wmcsf'][20:580, :]
        hr = single['HR_filt_ds'][self.idx_list[idx][1]]['hr_filt_ds'][20:580, :]  # trimmed

        # # TO DO multi-head

        hr_norm = (hr - hr.mean(axis=0)) / hr.std(axis=0)  # z-score normalization
        gm_norm = (gm - gm.mean(axis=0)) / gm.std(axis=0)  # z-score normalization
        ngm_norm = (ngm - ngm.mean(axis=0)) / ngm.std(axis=0)  # z-score normalization
        roi_norm = np.hstack((gm_norm, ngm_norm))

        # plt.subplot(311)
        # plt.plot(gm_norm[:, 5], 'g')
        # plt.legend(['gm'])
        # plt.subplot(312)
        # plt.plot(hr_norm, 'b')
        # plt.legend(['hr'])
        # plt.subplot(313)
        # plt.plot(ngm_norm, 'r')
        # plt.legend(['ngm'])
        # plt.show()

        # swap axis because
        # numpy: W x C
        # torch: C X W
        roi_norm = roi_norm.transpose((1, 0))
        hr_norm = hr_norm.squeeze()

        sample = {'roi': roi_norm, 'hr': hr_norm}

        # if self.transform:
        #     sample = self.transform(sample)

        sample = ToTensor()(sample)
        sample['hr_path'] = hr_path
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        roi, hr = sample['roi'], sample['hr']

        return {'roi': torch.from_numpy(roi).type(torch.FloatTensor),
                 'hr': torch.from_numpy(hr).type(torch.FloatTensor)}
