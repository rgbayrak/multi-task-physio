# Author: Roza G. Bayrak. PhD Student
# Date: June 2021
#
# Purpose: to create a dictionary from all the task files we have, the files were selected based on physio quality
# So, we don't have all the available HCP task data, rather subsets
# But existing subject scans have all 4 atlas ts inputs and 2 physio (hr, rv) labels
# Here creating a dictionary of dictionaries

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
                data[subject_id][parts[2]] = {clust_list[0]: [roi_path + '/' + clust_list[0] + '/rois_' + line.strip('\n') + '.mat'],
                                    clust_list[1]: [roi_path + '/' + clust_list[1] + '/rois_' + line.strip('\n') + '.mat'],
                                    clust_list[2]: [roi_path + '/' + clust_list[2] + '/rois_' + line.strip('\n') + '.mat'],
                                    clust_list[3]: [roi_path + '/' + clust_list[3] + '/rois_' + line.strip('\n') + '.mat'],
                                    'HR_filt_ds': [hr_path + '/HR_filtds_' + line.strip('\n') + '.mat'],
                                    'RV_filt_ds': [rv_path + '/RV_filtds_' + line.strip('\n') + '.mat']}
            else:
                if parts[2] not in data[subject_id]:
                    data[subject_id][parts[2]] = {clust_list[0]: [roi_path + '/' + clust_list[0] + '/rois_' + line.strip('\n') + '.mat'],
                                        clust_list[1]: [roi_path + '/' + clust_list[1] + '/rois_' + line.strip('\n') + '.mat'],
                                        clust_list[2]: [roi_path + '/' + clust_list[2] + '/rois_' + line.strip('\n') + '.mat'],
                                        clust_list[3]: [roi_path + '/' + clust_list[3] + '/rois_' + line.strip('\n') + '.mat'],
                                        'HR_filt_ds': [hr_path + '/HR_filtds_' + line.strip('\n') + '.mat'],
                                        'RV_filt_ds': [rv_path + '/RV_filtds_' + line.strip('\n') + '.mat']}
                else:
                    data[subject_id][parts[2]][clust_list[0]].append(roi_path + '/' + clust_list[0] + '/rois_' + line.strip('\n') + '.mat')
                    data[subject_id][parts[2]][clust_list[1]].append(roi_path + '/' + clust_list[1] + '/rois_' + line.strip('\n') + '.mat')
                    data[subject_id][parts[2]][clust_list[2]].append(roi_path + '/' + clust_list[2] + '/rois_' + line.strip('\n') + '.mat')
                    data[subject_id][parts[2]][clust_list[3]].append(roi_path + '/' + clust_list[3] + '/rois_' + line.strip('\n') + '.mat')
                    data[subject_id][parts[2]]['HR_filt_ds'].append(hr_path + '/HR_filtds_' + line.strip('\n') + '.mat')
                    data[subject_id][parts[2]]['RV_filt_ds'].append(rv_path + '/RV_filtds_' + line.strip('\n') + '.mat')




pass




