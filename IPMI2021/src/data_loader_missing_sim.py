import copy
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
import pickle
from skimage import io, transform

# data loader for simulating missing data

def get_roi_len(dirs):
    roi_path = "/bigdata/HCP_1200/power+xifra/resting_min+prepro/bpf-ds/"
    roi_len = 0
    for dir in dirs:
        files = os.listdir(roi_path + dir)
        roi = loadmat(roi_path + dir + '/' + files[0])
        key = [x for x in list(roi.keys()) if x.startswith('roi_dat')][0]
        roi_len += roi[key].shape[1]
    return roi_len


def get_sub(path):
    fp = open(path, 'r')
    sublines = fp.readlines()
    roi_fold = []
    hr_fold = []
    rv_fold = []

    for subline in sublines:
        roi_fold.append(subline)
        hr_fold.append(subline.replace('.mat', '_hr_filt_ds.mat').replace('rois_', ''))
        rv_fold.append(subline.replace('.mat', '_rv_filt_ds.mat').replace('rois_', ''))
    fp.close()
    return roi_fold, hr_fold, rv_fold

def get_dictionary(opt):
    if opt.mode == 'train':
        fold = opt.train_fold
    elif opt.mode == 'test':
        fold = opt.test_fold
    percent = opt.percent
    roi_path = os.path.join("/bigdata/HCP_1200/power+xifra/resting_min+prepro/bpf-ds/")
    hr_path = "/data/HR_filt_ds/"
    rv_path = "/data/RV_filt_ds/"
    fold_path = os.path.join("/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/k_fold_files/", fold)
    roi_fold, hr_fold, rv_fold = get_sub(fold_path)

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
                                'HR_filt_ds': [hr_path + hr_fold[i].rstrip('\n')],
                                'RV_filt_ds': [rv_path + rv_fold[i].rstrip('\n')]}
        else:
            if clust_list[0] and clust_list[1] and clust_list[2] and clust_list[3] not in data[subject_id]:
                data[subject_id][clust_list[0]] = [roi_path + clust_list[0] + '/' + d.rstrip('\n')]
                data[subject_id][clust_list[1]] = [roi_path + clust_list[1] + '/' + d.rstrip('\n')]
                data[subject_id][clust_list[2]] = [roi_path + clust_list[2] + '/' + d.rstrip('\n')]
                data[subject_id][clust_list[3]] = [roi_path + clust_list[3] + '/' + d.rstrip('\n')]
                data[subject_id]['HR_filt_ds'] = [hr_path + hr_fold[i].rstrip('\n')]
                data[subject_id]['RV_filt_ds'] = [rv_path + rv_fold[i].rstrip('\n')]
            else:
                data[subject_id][clust_list[0]].append(roi_path + clust_list[0] + '/' + d.rstrip('\n'))
                data[subject_id][clust_list[1]].append(roi_path + clust_list[1] + '/' + d.rstrip('\n'))
                data[subject_id][clust_list[2]].append(roi_path + clust_list[2] + '/' + d.rstrip('\n'))
                data[subject_id][clust_list[3]].append(roi_path + clust_list[3] + '/' + d.rstrip('\n'))
                data[subject_id]['HR_filt_ds'].append(hr_path + hr_fold[i].rstrip('\n'))
                data[subject_id]['RV_filt_ds'].append(rv_path + rv_fold[i].rstrip('\n'))

    # get the paths
    subj_excl = []
    for subj in data:
        paths = data[subj]['schaefer']
        # keep tract of the subjects that do not have all 4 scans
        if len(paths) == 4:
            subj_excl.append(subj)

        scan_order = []
        for path in paths:
            scan_order.append(path.replace('/bigdata/HCP_1200/power+xifra/resting_min+prepro/bpf-ds/schaefer/rois_', '').replace('.mat', ''))

        for k in data[subj]:
            new_paths = []
            for scan_id in scan_order:
                for path in data[subj][k]:
                    if scan_id in path:
                        new_paths.append(path)
                        break
            data[subj][k] = new_paths

    # remove subject physio to simulate missing data
    num_sub = int(np.round(len(data)*percent))
    subs = list(data.keys())[:num_sub]
    missing_sub_log = "/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/logs/" + opt.uni_id
    if not os.path.isdir(missing_sub_log):
        os.makedirs(missing_sub_log)
    with open(missing_sub_log + "/missing-physio-" + str(opt.percent) + "-percent.txt", "w") as fp:  # Pickling
        fp.write('\n'.join(sorted(subs)))
    for sub in subs:
        data[sub].pop('HR_filt_ds')
        # data[sub].pop('RV_filt_ds')

    # print(list(data.keys())) # subject_ids
    return data, num_sub


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
            for i, val in enumerate(self.data[subj]['schaefer']):
                self.idx_list.append([subj, i])

        self.keys = list(self.data.keys())  # so, we just do it once
        self.transform = transform
        self.roi_list = roi_list

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # create a dictionary that will have keys depending on the availability of input data
        sample = {}

        # load on the fly
        single = self.data[self.idx_list[idx][0]]  # passing the subject string to get the other dictionary
        single_paths = self.paths[self.idx_list[idx][0]]

        # ROI block
        schaefer_path = single_paths['schaefer'][self.idx_list[idx][1]]
        schaefer = single[self.roi_list[0]][self.idx_list[idx][1]]['roi_dat'][0:600, :]
        tractseg = single[self.roi_list[1]][self.idx_list[idx][1]]['roi_dat'][0:600, :]
        tian = single[self.roi_list[2]][self.idx_list[idx][1]]['roi_dat'][0:600, :]
        aan = single[self.roi_list[3]][self.idx_list[idx][1]]['roi_dat'][0:600, :]

        schaefer_norm = (schaefer - schaefer.mean(axis=0)) / schaefer.std(axis=0)  # z-score normalization
        tractseg_norm = (tractseg - tractseg.mean(axis=0)) / tractseg.std(axis=0)  # z-score normalization
        tian_norm = (tian - tian.mean(axis=0)) / tian.std(axis=0)  # z-score normalization
        aan_norm = (aan - aan.mean(axis=0)) / aan.std(axis=0)  # z-score normalization

        # add all up
        roi_norm = np.hstack((schaefer_norm, tractseg_norm, tian_norm, aan_norm))

        # Physio block
        if 'HR_filt_ds' in single:
            hr = single['HR_filt_ds'][self.idx_list[idx][1]]['hr_filt_ds'][0:600, :]  # trimmed
            hr_norm = (hr - hr.mean(axis=0)) / hr.std(axis=0)  # z-score normalization
            hr_mask = 1
        else:
            hr_norm = np.zeros((600,))
            hr_mask = 0

        if 'RV_filt_ds' in single:
            rv = single['RV_filt_ds'][self.idx_list[idx][1]]['rv_filt_ds'][0:600, :]  # trimmed
            rv_norm = (rv - rv.mean(axis=0)) / rv.std(axis=0)  # z-score normalization
            rv_mask = 1
        else:
            rv_norm = np.zeros((600,))
            rv_mask = 0

        # plt.subplot(911)
        # plt.plot(roi_norm[:, 1], 'b')
        # plt.subplot(912)
        # plt.plot(roi_norm[:, 10], 'b')
        # plt.subplot(913)
        # plt.plot(roi_norm[:, 100], 'b')
        # plt.subplot(914)
        # plt.plot(roi_norm[:, 300], 'b')
        # plt.subplot(915)
        # plt.plot(roi_norm[:, 405], 'b')
        # plt.subplot(916)
        # plt.plot(roi_norm[:, 467], 'b')
        # plt.subplot(917)
        # plt.plot(roi_norm[:, 482], 'b')
        # plt.subplot(918)
        # plt.plot(roi_norm[:, 425], 'b')
        # plt.subplot(919)
        # plt.plot(roi_norm[:, 495], 'b')
        # plt.show()

        # swap axis because
        # numpy: W x C
        # torch: C X W
        roi_norm = roi_norm.transpose((1, 0))
        sample['roi'] = roi_norm

        hr_norm = hr_norm.squeeze()
        sample['hr'] = hr_norm
        sample['hr_mask'] = np.array(hr_mask, dtype=np.bool)

        rv_norm = rv_norm.squeeze()
        sample['rv'] = rv_norm
        sample['rv_mask'] = np.array(rv_mask, dtype=np.bool)

        # sample = {'roi': roi_norm, 'hr': hr_norm, 'rv': rv_norm}

        # if self.transform:
        #     sample = self.transform(sample)

        sample = ToTensor()(sample)
        sample['schaefer'] = schaefer_path
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        sample['roi'] = torch.from_numpy(sample['roi']).type(torch.float32)
        sample['hr'] = torch.from_numpy(sample['hr']).type(torch.float32)
        sample['rv'] = torch.from_numpy(sample['rv']).type(torch.float32)

        return sample
