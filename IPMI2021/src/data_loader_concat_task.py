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
    roi_path = "/bigdata/HCP_task/LANGUAGE/bpf-ds/"
    roi_len = 0
    for dir in dirs:
        files = os.listdir(roi_path + dir)
        roi = loadmat(roi_path + dir + '/' + files[0])
        key = [x for x in list(roi.keys()) if x.startswith('roi_dat')][0]
        roi_len += roi[key].shape[1]
    return roi_len

def get_dictionary(opt):
    if opt.mode == 'train':
        fold = opt.train_fold
    elif opt.mode == 'test':
        fold = opt.test_fold

    # # LOOK AT YOUR DATA
    # x = os.path.join(rv_path, 'RV_filtds_983773_3T_rfMRI_REST1_RL.mat')
    # rv = loadmat(x)
    # rv.keys()
    # type(ROI['roi_dat']), ROI['roi_dat'].shape
    # type(ROI['roi_inds']), ROI['roi_inds'].shape
    # type(rv['rv_filt_ds']), rv['rv_filt_ds'].shape
    # type(rv['tax']), rv['tax'].shape

    list_of_tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'SOCIAL', 'WM']
    clust_list = ['schaefer', 'tractseg', 'tian', 'aan']
    path_to_good_lists = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/noverlap_scans'

    data = {}

    for task in list_of_tasks:
        task_file = path_to_good_lists + '/' + task + '.txt'

        roi_path = '/bigdata/HCP_task/' + task + '/bpf-ds'
        rv_path = '/bigdata/HCP_task/' + task + '/RV_filt_ds'
        hr_path = '/bigdata/HCP_task/' + task + '/HR_filt_ds'

        with open(task_file) as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split('_')
                subject_id = parts[0]
                if subject_id not in data:
                    data[subject_id] = {parts[2]: {}}
                    data[subject_id][parts[2]] = {
                        clust_list[0]: [roi_path + '/' + clust_list[0] + '/rois_' + line.strip(' \n') + '.mat'],
                        clust_list[1]: [roi_path + '/' + clust_list[1] + '/rois_' + line.strip(' \n') + '.mat'],
                        clust_list[2]: [roi_path + '/' + clust_list[2] + '/rois_' + line.strip(' \n') + '.mat'],
                        clust_list[3]: [roi_path + '/' + clust_list[3] + '/rois_' + line.strip(' \n') + '.mat'],
                        'HR_filt_ds': [hr_path + '/HR_filtds_' + line.strip(' \n') + '.mat'],
                        'RV_filt_ds': [rv_path + '/RV_filtds_' + line.strip(' \n') + '.mat']}
                else:
                    if parts[2] not in data[subject_id]:
                        data[subject_id][parts[2]] = {
                            clust_list[0]: [roi_path + '/' + clust_list[0] + '/rois_' + line.strip(' \n') + '.mat'],
                            clust_list[1]: [roi_path + '/' + clust_list[1] + '/rois_' + line.strip(' \n') + '.mat'],
                            clust_list[2]: [roi_path + '/' + clust_list[2] + '/rois_' + line.strip(' \n') + '.mat'],
                            clust_list[3]: [roi_path + '/' + clust_list[3] + '/rois_' + line.strip(' \n') + '.mat'],
                            'HR_filt_ds': [hr_path + '/HR_filtds_' + line.strip(' \n') + '.mat'],
                            'RV_filt_ds': [rv_path + '/RV_filtds_' + line.strip(' \n') + '.mat']}
                    else:
                        data[subject_id][parts[2]][clust_list[0]].append(
                            roi_path + '/' + clust_list[0] + '/rois_' + line.strip(' \n') + '.mat')
                        data[subject_id][parts[2]][clust_list[1]].append(
                            roi_path + '/' + clust_list[1] + '/rois_' + line.strip(' \n') + '.mat')
                        data[subject_id][parts[2]][clust_list[2]].append(
                            roi_path + '/' + clust_list[2] + '/rois_' + line.strip(' \n') + '.mat')
                        data[subject_id][parts[2]][clust_list[3]].append(
                            roi_path + '/' + clust_list[3] + '/rois_' + line.strip(' \n') + '.mat')
                        data[subject_id][parts[2]]['HR_filt_ds'].append(
                            hr_path + '/HR_filtds_' + line.strip(' \n') + '.mat')
                        data[subject_id][parts[2]]['RV_filt_ds'].append(
                            rv_path + '/RV_filtds_' + line.strip(' \n') + '.mat')

    fold_path = os.path.join("/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/k_fold_task_files/", fold)
    ids_in_fold = get_sub(fold_path)

    # create a smaller dictionary using the keys from each fold
    from funcy import project
    data = project(data, ids_in_fold)

    return data

def get_sub(path):
    fp = open(path, 'r')
    sublines = fp.readlines()
    ids_in_fold = []

    for subline in sublines:
        ids_in_fold.append(subline.strip('\n'))
    fp.close()
    return ids_in_fold


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
        # self.idx_list = []

        self.data_rl = {}
        self.data_lr = {}
        self.all_data = []

        for subj in self.data.keys():
            for task in self.data[subj]:
                for folder in self.data[subj][task]:
                    for i, val in enumerate(self.data[subj][task][folder]):
                        path = self.data[subj][task][folder][i]
                        vec = loadmat(val)
                        if path.endswith('LR.mat'):
                            if subj + '_LR' not in self.data_lr:
                                self.data_lr[subj + '_LR'] = {}
                            if task not in self.data_lr[subj + '_LR']:
                                self.data_lr[subj + '_LR'][task] = {}
                            self.data_lr[subj + '_LR'][task][folder] = vec

                        elif path.endswith('RL.mat'):
                            if subj + '_RL' not in self.data_rl:
                                self.data_rl[subj + '_RL'] = {}
                            if task not in self.data_rl[subj + '_RL']:
                                self.data_rl[subj + '_RL'][task] = {}
                            self.data_rl[subj + '_RL'][task][folder] = vec
                        else:
                            print('Path does not specify phase: {}'.format(path))

        self.all_data = copy.deepcopy(self.data_lr)
        self.all_data.update(self.data_rl)

        del self.data_rl, self.data_lr

        self.keys = list(self.all_data.keys())  # so, we just do it once
        self.transform = transform
        self.roi_list = roi_list

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        # if you want to load data on the fly, do it here
        single = self.all_data[self.keys[idx]]  # passing the subject string to get the other dictionary
        id = self.keys[idx]
        tasks = list(single.keys())
        input_data, label_rv, label_hr = [], [], []

        # First z-score normalize and then concat all the tasks. Resulting matrix size > N-ROIs x len(N-task) matrix
        for task in single:
            roi_data, hr_data, rv_data = [], [], []
            for jj in range(len(single[task])-2):
                roi = single[task][self.roi_list[jj]]['roi_dat']
                roi_norm = (roi - roi.mean(axis=0)) / roi.std(axis=0)  # z-score normalization
                roi_data.append(roi_norm)

            hr = single[task]['HR_filt_ds']['hr_filt_ds']
            rv = single[task]['RV_filt_ds']['rv_filt_ds']

            hr_norm = (hr - hr.mean(axis=0)) / hr.std(axis=0)  # z-score normalization
            rv_norm = (rv - rv.mean(axis=0)) / rv.std(axis=0)  # z-score normalization

            hr_data.append(hr_norm)
            rv_data.append(rv_norm)

            roi_data = np.concatenate(roi_data, axis=1)
            hr_data = np.concatenate(hr_data, axis=1)
            rv_data = np.concatenate(rv_data, axis=1)
            input_data.append(roi_data)
            label_hr.append(hr_data)
            label_rv.append(rv_data)

        input_data = np.concatenate(input_data, axis=0)
        label_hr = np.concatenate(label_hr, axis=0)
        label_rv = np.concatenate(label_rv, axis=0)

        # # mask some specific ROIs to test their effect
            # mask = np.ones(tractseg.shape[1]) > 0
            # mask[[15, 16, 17, 18, 21, 22, 27, 32, 33, 34, 35]] = 0
            # tractseg = tractseg[:, mask]

        # plt.subplot(211)
        # plt.plot(input_data)
        # plt.subplot(212)
        # plt.plot(label_rv, 'g')
        # plt.plot(label_hr, 'r')
        # # plt.ylim([0, 100])
        # plt.show()

        # swap axis because
        # numpy: W x C
        # torch: C X W
        input_data = input_data.transpose((1, 0))
        label_rv = label_rv.squeeze()
        label_hr = label_hr.squeeze()

        sample = {'roi': input_data, 'hr': label_hr, 'rv': label_rv}

        # if self.transform:
        #     sample = self.transform(sample)

        sample = ToTensor()(sample)
        sample['id'] = id
        sample['task'] = tasks
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        roi, hr, rv = sample['roi'], sample['hr'], sample['rv']

        return {'roi': torch.from_numpy(roi).type(torch.FloatTensor),
                'hr': torch.from_numpy(hr).type(torch.FloatTensor), 'rv': torch.from_numpy(rv).type(torch.FloatTensor)}
