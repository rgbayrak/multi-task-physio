import os
from random import shuffle

roi_path = "/bigdata/HCP_1200/power+xifra/resting_min+prepro/findlab90/bpf-ds-mat/"
out_dir = '/home/bayrakrg/neurdy/pycharm/ISBI2020/k_fold_files/'
os.mkdir(out_dir)

train_files = []
id_list = []
test_files = []

files = os.listdir(roi_path)
all_files = os.listdir(roi_path)

# subject_number_fold
data = {}
for i, d in enumerate(files):
    subdir_parts = files[i].rstrip(".mat").split('_')
    subject_id = subdir_parts[1]

    if subject_id not in id_list:
        id_list.append(subject_id)

# k-fold
k = 5
parc = round(len(files) / k)

for i in range(0, k):
    test_files = []
    for l, id in enumerate(id_list):
        if len(test_files) < parc:
            test_files.extend([x for x in files if id in x])

    train_files = [item for item in all_files if item not in test_files]
    files = [item for item in files if item not in test_files]

    test_fname = os.path.join(out_dir, 'test_fold_{}.txt'.format(i))
    train_fname = os.path.join(out_dir, 'train_fold_{}.txt'.format(i))

    with open(test_fname, 'w') as f:
        for j in test_files:
            # id = j.split('_')[1]
            f.write('{}\n'.format(j))

    with open(train_fname, 'w') as f:
        for k in train_files:
            # id = k.split('_')[1]
            f.write('{}\n'.format(k))

