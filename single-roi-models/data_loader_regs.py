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


def get_reg_len():
    reg_path = "/bigdata/HCP_1200/power+xifra/resting_min+prepro/bpf-ds/regs/movement"
    reg_len = 0
    files = os.listdir(reg_path)
    reg = loadmat(reg_path + '/' + files[0])
    key = [x for x in list(reg.keys()) if x.startswith('m_filt_ds')][0]
    reg_len += reg[key].shape[1]
    return reg_len


def get_sub(path):
    fp = open(path, 'r')
    sublines = fp.readlines()
    reg_fold = []
    hr_fold = []
    rv_fold = []

    for subline in sublines:
        reg_fold.append(subline.replace('rois_', 'mv_filt_ds_'))
        hr_fold.append(subline.replace('.mat', '_hr_filt_ds.mat').replace('rois_', ''))
        rv_fold.append(subline.replace('.mat', '_rv_filt_ds.mat').replace('rois_', ''))
    fp.close()
    return reg_fold, hr_fold, rv_fold


def get_dictionary(fold):
    reg_path = os.path.join("/bigdata/HCP_1200/power+xifra/resting_min+prepro/bpf-ds/regs/movement/")
    hr_path = "/data/HR_filt_ds/"
    rv_path = "/data/RV_filt_ds/"
    fold_path = os.path.join("/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/k_fold_files/", fold)
    reg_fold, hr_fold, rv_fold = get_sub(fold_path)

    # # LOOK AT YOUR DATA
    x = os.path.join(rv_path, '983773_rfMRI_REST1_RL_rv_filt_ds.mat')
    y = os.path.join(hr_path, '983773_rfMRI_REST1_RL_hr_filt_ds.mat')
    z = os.path.join(reg_path, 'mv_filt_ds_983773_rfMRI_REST1_RL.mat')
    rv = loadmat(x)
    hr = loadmat(y)
    mv = loadmat(z)
    rv.keys()
    # type(mv['m_filt_ds']), mv['m_filt_ds'].shape
    # type(rv['rv_filt_ds']), rv['rv_filt_ds'].shape
    # type(rv['tax']), rv['tax'].shape

    data = {}
    for i, d in enumerate(reg_fold):
        subdir_parts = reg_fold[i].rstrip(".mat").split('_')
        subject_id = subdir_parts[3]
        # print("{}".format(subject_id))

        if subject_id not in data:
            data[subject_id] = {'MV_filt_ds': [reg_path + reg_fold[i].rstrip('\n')],
                                'HR_filt_ds': [hr_path + hr_fold[i].rstrip('\n')],
                                'RV_filt_ds': [rv_path + rv_fold[i].rstrip('\n')]}

    # get the paths
    subj_excl = []
    for subj in data:
        paths = data[subj]['MV_filt_ds']
        # keep tract of the subjects that do not have all 4 scans
        if len(paths) == 4:
            subj_excl.append(subj)

        scan_order = []
        for path in paths:
            scan_order.append(path.lstrip('/bigdata/HCP_1200/power+xifra/resting_min+prepro/regs/movement/bpf-ds/mv_filt_ds_').rstrip('.mat'))

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

    def __init__(self, data, transform=None):
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

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # load on the fly
        single = self.data[self.idx_list[idx][0]]  # passing the subject string to get the other dictionary
        single_paths = self.paths[self.idx_list[idx][0]]
        hr_path = single_paths['HR_filt_ds'][self.idx_list[idx][1]]
        regs = single['MV_filt_ds'][self.idx_list[idx][1]]['m_filt_ds'][0:600, :]
        hr = single['HR_filt_ds'][self.idx_list[idx][1]]['hr_filt_ds'][0:600, :]  # trimmed
        rv = single['RV_filt_ds'][self.idx_list[idx][1]]['rv_filt_ds'][0:600, :]  # trimmed

        # # TO DO multi-head

        hr_norm = (hr - hr.mean(axis=0)) / hr.std(axis=0)  # z-score normalization
        rv_norm = (rv - rv.mean(axis=0)) / rv.std(axis=0)  # z-score normalization
        regs_norm = (regs - regs.mean(axis=0)) / regs.std(axis=0)  # z-score normalization

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
        regs_norm = regs_norm.transpose((1, 0))
        hr_norm = hr_norm.squeeze()
        rv_norm = rv_norm.squeeze()

        sample = {'regs': regs_norm, 'hr': hr_norm, 'rv': rv_norm}

        # if self.transform:
        #     sample = self.transform(sample)

        sample = ToTensor()(sample)
        sample['hr_path'] = hr_path
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        regs, hr, rv = sample['regs'], sample['hr'], sample['rv']

        return {'regs': torch.from_numpy(regs).type(torch.FloatTensor),
                'hr': torch.from_numpy(hr).type(torch.FloatTensor), 'rv': torch.from_numpy(rv).type(torch.FloatTensor)}
