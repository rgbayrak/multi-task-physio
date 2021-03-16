import os
import numpy as np
resting_folds = os.listdir('IPMI2021/k_fold_files')
current_subj = 'boo'

task_subject = []
for fold in resting_folds:
    if 'test' in fold:
        f = open("/home/bayrakrg/Desktop/resting_task_overlap_byfold/" + fold, "r")
        FOLD = f.readlines()
        # task_subject_lst.append(np.unique(np.array(FOLD)))
        task_subject.extend([x.strip() for x in FOLD])
        f.close()

f = open("/home/bayrakrg/Desktop/task_test.txt", "r")
lst = f.readlines()
h = open("/home/bayrakrg/Desktop/non_overlapping.txt", "a")
for F in lst:
    for tsubj in task_subject:
        if tsubj in F:
            if F != current_subj:
                current_subj = F
                h.write(F)
h.close()