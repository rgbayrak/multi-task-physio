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
    roi_path = "/bigdata/eegfmri_nih/bpf-rs/"
    roi_len = 0
    for dir in dirs:
        files = os.listdir(roi_path + dir)
        roi = loadmat(roi_path + dir + '/' + files[0])
        key = [x for x in list(roi.keys()) if x.startswith('roi_dat_rs')][0]
        roi_len += roi[key].shape[1]
    return roi_len


def get_sub(path):
    fp = open(path, 'r')
    sublines = fp.readlines()
    roi_fold = []
    rv_fold = []
    hr_fold = []

    for subline in sublines:
        roi_fold.append(subline.replace('\n', ''))
        rv_fold.append(subline.replace('rois_', 'RV_filtrs_').replace('\n', ''))
        hr_fold.append(subline.replace('rois_', 'HR_filtrs_').replace('\n', ''))
    fp.close()
    return roi_fold, rv_fold, hr_fold

def get_dictionary(opt):

    if opt.mode == 'train':
        fold = opt.train_fold
    elif opt.mode == 'test':
        fold = opt.test_fold

    roi_path = os.path.join("/bigdata/eegfmri_nih/bpf-rs/")
    rv_path = "/bigdata/eegfmri_nih/RV_filt_rs/"
    hr_path = "/bigdata/eegfmri_nih/HR_filt_rs/"
    fold_path = os.path.join("/IPMI2021/nih_files/", fold)
    roi_fold, rv_fold, hr_fold = get_sub(fold_path)

    # # LOOK AT YOUR DATA
    # x = os.path.join(rv_path, 'RV_filtds_983773_3T_rfMRI_REST1_RL.mat')
    # rv = loadmat(x)
    # rv.keys()
    # type(ROI['roi_dat']), ROI['roi_dat'].shape
    # type(ROI['roi_inds']), ROI['roi_inds'].shape
    # type(rv['rv_filt_ds']), rv['rv_filt_ds'].shape
    # type(rv['tax']), rv['tax'].shape

    data = {}
    for i, d in enumerate(roi_fold):
        subdir_parts = roi_fold[i].rstrip(".mat").split('_')
        subject_id = subdir_parts[1]
        # print("{}".format(subject_id))

        clust_list = ['schaefer', 'tractseg', 'tian', 'aan']
        if subject_id not in data:
            data[subject_id] = {clust_list[0]: [roi_path + clust_list[0] + '/' + d.rstrip('\n')],
                                clust_list[1]: [roi_path + clust_list[1] + '/' + d.rstrip('\n')],
                                clust_list[2]: [roi_path + clust_list[2] + '/' + d.rstrip('\n')],
                                clust_list[3]: [roi_path + clust_list[3] + '/' + d.rstrip('\n')],
                                'RV_filt_rs': [rv_path + rv_fold[i].rstrip('\n')],
                                'HR_filt_rs': [hr_path + hr_fold[i].rstrip('\n')]}
        else:
            if clust_list[0] and clust_list[1] and clust_list[2] and clust_list[3] not in data[subject_id]:
                data[subject_id][clust_list[0]] = [roi_path + clust_list[0] + '/' + d.rstrip('\n')]
                data[subject_id][clust_list[1]] = [roi_path + clust_list[1] + '/' + d.rstrip('\n')]
                data[subject_id][clust_list[2]] = [roi_path + clust_list[2] + '/' + d.rstrip('\n')]
                data[subject_id][clust_list[3]] = [roi_path + clust_list[3] + '/' + d.rstrip('\n')]
                data[subject_id]['RV_filt_rs'] = [rv_path + rv_fold[i].rstrip('\n')]
                data[subject_id]['HR_filt_rs'] = [hr_path + hr_fold[i].rstrip('\n')]
            else:
                data[subject_id][clust_list[0]].append(roi_path + clust_list[0] + '/' + d.rstrip('\n'))
                data[subject_id][clust_list[1]].append(roi_path + clust_list[1] + '/' + d.rstrip('\n'))
                data[subject_id][clust_list[2]].append(roi_path + clust_list[2] + '/' + d.rstrip('\n'))
                data[subject_id][clust_list[3]].append(roi_path + clust_list[3] + '/' + d.rstrip('\n'))
                data[subject_id]['RV_filt_rs'].append(rv_path + rv_fold[i].rstrip('\n'))
                data[subject_id]['HR_filt_rs'].append(hr_path + hr_fold[i].rstrip('\n'))

    # get the paths
    subj_excl = []
    for subj in data:
        paths = data[subj]['schaefer']

        scan_order = []
        for path in paths:
            scan_order.append(path.replace('/bigdata/eegfmri_nih/bpf-rs/schaefer/rois_', '').replace('.mat', ''))

        # for k in data[subj]:
        #     new_paths = []
        #     for scan_id in scan_order:
        #         for path in data[subj][k]:
        #             if scan_id in path:
        #                 new_paths.append(path)
        #                 break
        #     data[subj][k] = new_paths

    # print(list(data.keys())) # subject_ids
    return data


class data_to_tensor():
    """ From pytorch example"""

    def __init__(self, data, roi_list, transform=None):
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
            for i, val in enumerate(self.data[subj]['RV_filt_rs']):
                self.idx_list.append([subj, i])

        self.keys = list(self.data.keys())  # so, we just do it once
        self.transform = transform
        self.roi_list = roi_list

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # load on the fly
        single = self.data[self.idx_list[idx][0]]  # passing the subject string to get the other dictionary
        id = self.keys[idx]
        print(id)
        single_paths = self.paths[self.idx_list[idx][0]]
        rv_path = single_paths['RV_filt_rs'][self.idx_list[idx][1]]
        schaefer = single[self.roi_list[0]][self.idx_list[idx][1]]['roi_dat_rs']
        tractseg = single[self.roi_list[1]][self.idx_list[idx][1]]['roi_dat_rs']
        tian = single[self.roi_list[2]][self.idx_list[idx][1]]['roi_dat_rs']
        aan = single[self.roi_list[3]][self.idx_list[idx][1]]['roi_dat_rs']
        rv = single['RV_filt_rs'][self.idx_list[idx][1]]['rv_filt_ds']
        rv = np.concatenate(rv, axis=0)
        hr = single['HR_filt_rs'][self.idx_list[idx][1]]['hr_filt_ds']
        hr = np.concatenate(hr, axis=0)

        # # TO DO multi-head
        # z-score
        rv_norm = (rv - rv.mean()) / rv.std()
        hr_norm = (hr - hr.mean()) / hr.std()
        schaefer_norm = (schaefer - np.nanmean(schaefer, axis=0)) / np.nanstd(schaefer, axis=0)
        tractseg_norm = (tractseg - np.nanmean(tractseg, axis=0)) / np.nanstd(tractseg, axis=0)
        # print(np.sum(np.isnan(tractseg_norm)))
        tractseg_norm[np.isnan(tractseg_norm)] = 0
        tian_norm = (tian - tian.mean(axis=0)) / tian.std(axis=0)  # z-score normalization
        aan_norm = (aan - aan.mean(axis=0)) / aan.std(axis=0)  # z-score normalization
        roi_norm = np.hstack((schaefer_norm, tractseg_norm, tian_norm, aan_norm))

        plt.subplot(411)
        plt.plot(roi_norm[:, 1], 'b')
        plt.subplot(412)
        plt.plot(roi_norm[:, 10], 'b')
        plt.subplot(413)
        plt.plot(rv_norm, 'g')
        plt.subplot(414)
        plt.plot(hr_norm, 'r')
        plt.show()

        # swap axis because
        # numpy: W x C
        # torch: C X W
        roi_norm = roi_norm.transpose((1, 0))
        rv_norm = rv_norm.squeeze()
        hr_norm = hr_norm.squeeze()

        sample = {'roi': roi_norm, 'rv': rv_norm, 'hr': hr_norm}

        # if self.transform:
        #     sample = self.transform(sample)

        sample = ToTensor()(sample)
        sample['rv_path'] = rv_path
        sample['id'] = id
        sample['task'] = 'TR_test'
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        roi, rv, hr = sample['roi'], sample['rv'], sample['hr']

        return {'roi': torch.from_numpy(roi).type(torch.FloatTensor),'rv': torch.from_numpy(rv).type(torch.FloatTensor)
                ,'hr': torch.from_numpy(hr).type(torch.FloatTensor)}
