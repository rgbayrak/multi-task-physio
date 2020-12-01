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
import seaborn as sns


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


def get_dictionary(fold):
    roi_path = os.path.join("/bigdata/HCP_1200/power+xifra/resting_min+prepro/bpf-ds/")
    hr_path = "/data/HR_filt_ds/"
    rv_path = "/data/RV_filt_ds/"
    fold_path = os.path.join("/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/k_fold_files/", fold)
    roi_fold, hr_fold, rv_fold = get_sub(fold_path)

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
            scan_order.append(path.lstrip('/bigdata/HCP_1200/power+xifra/resting_min+prepro/schaefer/bpf-ds-mat/rois_').rstrip('.mat'))

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

    def __init__(self, data, opt, transform=None):
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
            for i, val in enumerate(self.data[subj]['RV_filt_ds']):
                for j in range(0, 600-opt.window_size+1):  # 537 is the number of samples from a single scan
                    self.idx_list.append([subj, i, j])

        self.keys = list(self.data.keys())  # so, we just do it once
        self.transform = transform
        self.roi_list = opt.roi_list
        # self.opt = opt
        self.w = opt.window_size

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # load on the fly
        # rand_idx = random.randint(0, len(self.idx_list)*537 - 1)
        single = self.data[self.idx_list[idx][0]]  # passing the subject string to get the other dictionary
        single_paths = self.paths[self.idx_list[idx][0]]
        rv_path = single_paths['RV_filt_ds'][self.idx_list[idx][1]]
        schaefer = single[self.roi_list[0]][self.idx_list[idx][1]]['roi_dat']
        tractseg = single[self.roi_list[1]][self.idx_list[idx][1]]['roi_dat']
        tian = single[self.roi_list[2]][self.idx_list[idx][1]]['roi_dat']
        aan = single[self.roi_list[3]][self.idx_list[idx][1]]['roi_dat']
        rv = single['RV_filt_ds'][self.idx_list[idx][1]]['rv_filt_ds']
        hr = single['HR_filt_ds'][self.idx_list[idx][1]]['hr_filt_ds']
        # plt.figure(1)
        # plt.plot(roi[:, 1])  # look at your data
        # plt.plot(rv)  # look at your data

        # z-score normalization
        schaefer_norm = (schaefer - schaefer.mean(axis=0)) / schaefer.std(axis=0)  # z-score normalization
        tractseg_norm = (tractseg - tractseg.mean(axis=0)) / tractseg.std(axis=0)  # z-score normalization
        tian_norm = (tian - tian.mean(axis=0)) / tian.std(axis=0)  # z-score normalization
        aan_norm = (aan - aan.mean(axis=0)) / aan.std(axis=0)  # z-score normalization
        roi_norm = np.hstack((schaefer_norm, tractseg_norm, tian_norm, aan_norm))

        # windowing w=64
        roi = roi_norm[self.idx_list[idx][2]:(self.idx_list[idx][2] + self.w)]
        rv = rv[self.idx_list[idx][2] + int(self.w/2)]
        hr = hr[self.idx_list[idx][2] + int(self.w/2)]

        # # Sanity check: Visualize the input
        # plt.scatter(self.idx_list[idx][2] + 32, rv)
        # plt.plot(np.arange(self.idx_list[idx][2], self.idx_list[idx][2]+roi[:, 1].shape[0]), roi[:, 1])
        # plt.legend(['roi', 'rv', 'window_roi', 'rv_middle_point'])
        # plt.grid(color='b', linestyle='-', linewidth=.0)
        # plt.show()

        # swap axis because
        # numpy: W x C
        # torch: C X W
        roi = roi.transpose((1, 0))
        # rv = rv.squeeze()

        sample = {'roi': roi, 'rv': rv, 'hr': hr}

        # if self.transform:
        #     sample = self.transform(sample)

        sample = ToTensor()(sample)
        return sample


class val_to_tensor:
    """ From pytorch example"""

    def __init__(self, data, opt, transform=None):
        self.data = copy.deepcopy(data)
        self.paths = copy.deepcopy(data)
        self.idx_list = []
        for subj in self.data.keys():
            for folder in self.data[subj]:
                for i, val in enumerate(self.data[subj][folder]):
                    self.data[subj][folder][i] = loadmat(val)

        # make sure in get_item that we see all data by
        for subj in self.data.keys():
            for i, val in enumerate(self.data[subj]['RV_filt_ds']):
                for j in range(0, 600-opt.window_size+1):  # 537 is the number of samples from a single scan
                    self.idx_list.append([subj, i, j])

        self.keys = list(self.data.keys())  # so, we just do it once
        self.transform = transform
        self.roi_list = opt.roi_list
        self.opt = opt
        self.w = opt.window_size


    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        # load on the fly
        # rand_idx = random.randint(0, len(self.idx_list)*537 - 1)
        single = self.data[self.idx_list[idx][0]]  # passing the subject string to get the other dictionary
        single_paths = self.paths[self.idx_list[idx][0]]
        rv_path = single_paths['RV_filt_ds'][self.idx_list[idx][1]]
        schaefer = single[self.roi_list[0]][self.idx_list[idx][1]]['roi_dat']
        tractseg = single[self.roi_list[1]][self.idx_list[idx][1]]['roi_dat']
        tian = single[self.roi_list[2]][self.idx_list[idx][1]]['roi_dat']
        aan = single[self.roi_list[3]][self.idx_list[idx][1]]['roi_dat']
        rv = single['RV_filt_ds'][self.idx_list[idx][1]]['rv_filt_ds']
        hr = single['HR_filt_ds'][self.idx_list[idx][1]]['hr_filt_ds']
        # plt.figure(1)
        # plt.plot(roi[:, 1])  # look at your data
        # plt.plot(rv)  # look at your data

        # z-score normalization
        schaefer_norm = (schaefer - schaefer.mean(axis=0)) / schaefer.std(axis=0)  # z-score normalization
        tractseg_norm = (tractseg - tractseg.mean(axis=0)) / tractseg.std(axis=0)  # z-score normalization
        tian_norm = (tian - tian.mean(axis=0)) / tian.std(axis=0)  # z-score normalization
        aan_norm = (aan - aan.mean(axis=0)) / aan.std(axis=0)  # z-score normalization
        roi_norm = np.hstack((schaefer_norm, tractseg_norm, tian_norm, aan_norm))

        # windowing w=64
        roi = roi_norm[self.idx_list[idx][2]:(self.idx_list[idx][2] + self.w)]
        rv = rv[self.idx_list[idx][2] + int(self.w/2)]
        hr = hr[self.idx_list[idx][2] + int(self.w/2)]

        # # Sanity check: Visualize the input
        # plt.scatter(self.idx_list[idx][2] + 32, rv)
        # plt.plot(np.arange(self.idx_list[idx][2], self.idx_list[idx][2]+roi[:, 1].shape[0]), roi[:, 1])
        # plt.legend(['roi', 'rv', 'window_roi', 'rv_middle_point'])
        # plt.grid(color='b', linestyle='-', linewidth=.0)
        # plt.show()

        # swap axis because
        # numpy: W x C
        # torch: C X W
        roi = roi.transpose((1, 0))
        # print(roi.shape)
        # print(rv.shape)

        sample = {'roi': roi, 'rv': rv, 'hr': hr}
        sample = ToTensor()(sample)

        return sample


class test_to_tensor:
    """ From pytorch example"""

    def __init__(self, data, opt, transform=None):
        self.data = copy.deepcopy(data)
        self.paths = copy.deepcopy(data)
        self.idx_list = []

        for subj in self.data.keys():
            for folder in self.data[subj]:
                for i, val in enumerate(self.data[subj][folder]):
                    self.data[subj][folder][i] = loadmat(val)

        # make sure in get_item that we see all data by
        for subj in self.data.keys():
            self.scans = 0
            for i, val in enumerate(self.data[subj]['RV_filt_ds']):
                self.scans += 1
                self.idx_list.append([subj, i])

        self.transform = transform
        self.roi_clust = opt.roi_clust
        self.w = opt.window_size

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        single = self.data[self.idx_list[idx][0]]  # passing the subject string to get the other dictionary
        single_paths = self.paths[self.idx_list[idx][0]]
        rv_path = single_paths['RV_filt_ds'][self.idx_list[idx][1]]
        roi = single[self.roi_clust][self.idx_list[idx][1]]['roi_dat']
        rv = single['RV_filt_ds'][self.idx_list[idx][1]]['rv_filt_ds']
        hr = single['HR_filt_ds'][self.idx_list[idx][1]]['hr_filt_ds']

        single_roi = []
        single_rv = []
        single_hr = []
        idx_tracker = np.arange(0, len(roi) + 1 - self.w)
        for i in idx_tracker:
            single_roi.append(roi[i:i + self.w])  # (64 x #ROI)
            single_rv.append(rv[i + int(self.w/2)])  # (64 x 1)
            single_hr.append(hr[i + int(self.w/2)])  # (64 x 1)

        single_roi = np.array(single_roi)
        single_roi = single_roi.transpose(0, 2, 1)
        single_rv = np.array(single_rv)
        single_hr = np.array(single_hr)

        sample = {'roi': single_roi, 'rv': single_rv, 'hr': single_hr}

        sample = ToTensor()(sample)
        sample['rv_path'] = rv_path

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        roi, rv, hr= sample['roi'], sample['rv'], sample['hr']

        return {'roi': torch.from_numpy(roi).type(torch.FloatTensor),
                'rv': torch.from_numpy(rv).type(torch.FloatTensor),
                'hr': torch.from_numpy(hr).type(torch.FloatTensor)}

#
# # w=128
# import copy
# import csv
# import os
# import random
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import torch
# from scipy.io import loadmat
# from skimage import io, transform
# import seaborn as sns
#
#
# def get_sub(path):
#     fp = open(path, 'r')
#     sublines = fp.readlines()
#     roi_fold = []
#     rv_fold = []
#
#     for subline in sublines:
#         rv_info = subline.replace('rois', 'RV_filtds')
#         roi_fold.append(subline)
#         rv_fold.append(rv_info)
#     fp.close()
#     return roi_fold, rv_fold
#
#
# def get_dictionary(fold, clust):
#     roi_path = "/home/bayrakrg/Data/RV/neuroimg_data/"
#     rv_path = "/home/bayrakrg/Data/RV/RV_filt_ds"
#     fold_path = os.path.join("/home/bayrakrg/neurdy/pycharm/neuroimg2020/RV/lso_fold_files/", fold)
#     print()
#     roi_fold, rv_fold = get_sub(fold_path)
#
#     # # LOOK AT YOUR DATA
#     # x = os.path.join(rv_path, 'RV_filtds_983773_3T_rfMRI_REST1_RL.mat')
#     # rv = loadmat(x)
#     # rv.keys()
#     # type(ROI['roi_dat']), ROI['roi_dat'].shape
#     # type(ROI['roi_inds']), ROI['roi_inds'].shape
#     # type(rv['rv_filt_ds']), rv['rv_filt_ds'].shape
#     # type(rv['tax']), rv['tax'].shape
#
#     data = {}
#     for i, d in enumerate(roi_fold):
#         subdir_parts = roi_fold[i].rstrip(".mat").split('_')
#         subject_id = subdir_parts[1]
#         # print("{}".format(subject_id))
#
#         if subject_id not in data:
#             data[subject_id] = {clust: [roi_path + clust + '/' + d.rstrip('\n')],
#                                 'RV_filt_ds': [rv_path + '/' + rv_fold[i].rstrip('\n')]}
#         else:
#             if clust not in data[subject_id]:
#                 data[subject_id][clust] = [roi_path + clust + '/' + d.rstrip('\n')]
#                 data[subject_id]['RV_filt_ds'] = [rv_path + '/' + rv_fold[i].rstrip('\n')]
#             else:
#                 data[subject_id][clust].append(roi_path + clust + '/' + d.rstrip('\n'))
#                 data[subject_id]['RV_filt_ds'].append(rv_path + '/' + rv_fold[i].rstrip('\n'))
#
#     # print(list(data.keys())) # subject_ids
#     return data
#
#
# class data_to_tensor():
#     """ From pytorch example"""
#
#     def __init__(self, data, opt, transform=None):
#         # go through all the data and load them in
#         # start with one worker
#         # as soon as I pass to the data loader it is gonna create a copy depending on the workers (threads)
#         # copy of the data for each worker (for more heavy duty data)
#         # random data augmentation usually needs multiple workers
#         self.data = copy.deepcopy(data)
#         self.paths = copy.deepcopy(data)
#         self.idx_list = []
#         for subj in self.data.keys():
#             for folder in self.data[subj]:
#                 for i, val in enumerate(self.data[subj][folder]):
#                     self.data[subj][folder][i] = loadmat(val)
#
#         # make sure in get_item that we see all data by
#         for subj in self.data.keys():
#             for i, val in enumerate(self.data[subj]['RV_filt_ds']):
#                 for j in range(0, 473):  # 473 is the number of samples from a single scan
#                     self.idx_list.append([subj, i, j])
#
#         self.keys = list(self.data.keys())  # so, we just do it once
#         self.transform = transform
#         self.roi_clust = opt.roi_clust
#         self.opt = opt
#
#     def __len__(self):
#         return len(self.idx_list)
#
#     def __getitem__(self, idx):
#         # load on the fly
#         # rand_idx = random.randint(0, len(self.idx_list)*473 - 1)
#         single = self.data[self.idx_list[idx][0]]  # passing the subject string to get the other dictionary
#         single_paths = self.paths[self.idx_list[idx][0]]
#         rv_path = single_paths['RV_filt_ds'][self.idx_list[idx][1]]
#         roi = single[self.roi_clust][self.idx_list[idx][1]]['roi_dat']
#         rv = single['RV_filt_ds'][self.idx_list[idx][1]]['rv_filt_ds']
#         # plt.figure(1)
#         # plt.plot(roi[:, 1])  # look at your data
#         # plt.plot(rv)  # look at your data
#
#         # normalization or not normalization
#         # # method1
#         # roi = (roi - roi.min()) / (roi.max() - roi.min())
#         # rv = (rv - rv.min()) / (rv.max() - rv.min())
#
#         # # method2
#         # rv_norm = (rv - rv.mean(axis=0)) / rv.std(axis=0)  # z-score normalization
#         # roi_norm = (roi - roi.mean(axis=0)) / roi.std(axis=0)  # z-score normalization
#
#         # Jorge's
#         roi = roi[self.idx_list[idx][2]:(self.idx_list[idx][2] + 128)]
#         rv = rv[self.idx_list[idx][2] + 64]
#
#         # # # Sanity check: Visualize the input
#         # plt.scatter(self.idx_list[idx][2]+64, rv)
#         # plt.plot(np.arange(self.idx_list[idx][2], self.idx_list[idx][2]+roi[:, 1].shape[0]), roi[:, 1])
#         # plt.legend(['roi', 'rv', 'window_roi', 'rv_middle_point'])
#         # plt.grid(color='b', linestyle='-', linewidth=.0)
#         # plt.show()
#
#         # swap axis because
#         # numpy: W x C
#         # torch: C X W
#         roi = roi.transpose((1, 0))
#         # rv = rv.squeeze()
#
#         sample = {'roi': roi, 'rv': rv}
#
#         # if self.transform:
#         #     sample = self.transform(sample)
#
#         sample = ToTensor()(sample)
#         return sample
#
#
# class val_to_tensor:
#     """ From pytorch example"""
#
#     def __init__(self, data, opt, transform=None):
#         self.data = copy.deepcopy(data)
#         self.paths = copy.deepcopy(data)
#         self.idx_list = []
#         for subj in self.data.keys():
#             for folder in self.data[subj]:
#                 for i, val in enumerate(self.data[subj][folder]):
#                     self.data[subj][folder][i] = loadmat(val)
#
#         # make sure in get_item that we see all data by
#         for subj in self.data.keys():
#             for i, val in enumerate(self.data[subj]['RV_filt_ds']):
#                 for j in range(0, 473):  # 473 is the number of samples from a single scan
#                     self.idx_list.append([subj, i, j])
#
#         self.keys = list(self.data.keys())  # so, we just do it once
#         self.transform = transform
#         self.roi_clust = opt.roi_clust
#         self.opt = opt
#
#     def __len__(self):
#         return len(self.idx_list)
#
#     def __getitem__(self, idx):
#         # load on the fly
#         # rand_idx = random.randint(0, len(self.idx_list)*473 - 1)
#         single = self.data[self.idx_list[idx][0]]  # passing the subject string to get the other dictionary
#         single_paths = self.paths[self.idx_list[idx][0]]
#         rv_path = single_paths['RV_filt_ds'][self.idx_list[idx][1]]
#         roi = single[self.roi_clust][self.idx_list[idx][1]]['roi_dat']
#         rv = single['RV_filt_ds'][self.idx_list[idx][1]]['rv_filt_ds']
#
#         # Jorge's
#         # Jorge's
#         roi = roi[self.idx_list[idx][2]:(self.idx_list[idx][2] + 128)]
#         rv = rv[self.idx_list[idx][2] + 64]
#
#         # swap axis because
#         # numpy: W x C
#         # torch: C X W
#         roi = roi.transpose((1, 0))
#
#         sample = {'roi': roi, 'rv': rv}
#         sample = ToTensor()(sample)
#
#         return sample
#
#
# class test_to_tensor:
#     """ From pytorch example"""
#
#     def __init__(self, data, opt, transform=None):
#         self.data = copy.deepcopy(data)
#         self.paths = copy.deepcopy(data)
#         self.idx_list = []
#
#         for subj in self.data.keys():
#             for folder in self.data[subj]:
#                 for i, val in enumerate(self.data[subj][folder]):
#                     self.data[subj][folder][i] = loadmat(val)
#
#         # make sure in get_item that we see all data by
#         for subj in self.data.keys():
#             self.scans = 0
#             for i, val in enumerate(self.data[subj]['RV_filt_ds']):
#                 self.scans += 1
#                 self.idx_list.append([subj, i])
#
#         self.transform = transform
#         self.roi_clust = opt.roi_clust
#
#     def __len__(self):
#         return len(self.idx_list)
#
#     def __getitem__(self, idx):
#         single = self.data[self.idx_list[idx][0]]  # passing the subject string to get the other dictionary
#         single_paths = self.paths[self.idx_list[idx][0]]
#         rv_path = single_paths['RV_filt_ds'][self.idx_list[idx][1]]
#         roi = single[self.roi_clust][self.idx_list[idx][1]]['roi_dat']
#         rv = single['RV_filt_ds'][self.idx_list[idx][1]]['rv_filt_ds']
#
#         single_roi = []
#         single_rv = []
#         idx_tracker = np.arange(0, len(roi) + 1 - 128)
#         for i in idx_tracker:
#             single_roi.append(roi[i:i + 128])  # (128 x #ROI)
#             single_rv.append(rv[i + 64])  # (64 x 1)
#
#         single_roi = np.array(single_roi)
#         single_roi = single_roi.transpose(0, 2, 1)
#         single_rv = np.array(single_rv)
#
#         sample = {'roi': single_roi, 'rv': single_rv}
#
#         sample = ToTensor()(sample)
#         sample['rv_path'] = rv_path
#
#         return sample
#
#
# class ToTensor(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         roi, rv = sample['roi'], sample['rv']
#
#         return {'roi': torch.from_numpy(roi).type(torch.FloatTensor),
#                 'rv': torch.from_numpy(rv).type(torch.FloatTensor)}
