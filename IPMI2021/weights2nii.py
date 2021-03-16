import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt

path_atlas = '/data/gm-atlases/Schaefer/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'
# path_atlas = '/data/wm-atlases/pandora_2mm/thresholded/TractSeg-th0.95-2mm_HCP.nii.gz'
# path_atlas = '/data/AAN_brainstem_2mm/AAN_MNI152_2mm.nii.gz'
# path_atlas = '/data/gm-atlases/Tian2020MSA/7T_2mm/Tian_Subcortex_S1_7T_2mm.nii.gz'

atlas = nib.load(path_atlas)
atlas_img = atlas.get_fdata()

atlas_id = 'schaefer'
d4 = False
# exclude = [4, 12]
exclude = []
path_labels = 'info.csv'
with open(path_labels, 'r') as f:
    content = f.readlines()

hr = []
rv = []
ids = []
roi_labels = []
id = 0
for i in range(len(content)):
    _, label, atlas_name, rv_val, hr_val, val = content[i].strip().split(',')
    if i != 0 and atlas_name == atlas_id:
        # hr.append(float(hr_val))
        # rv.append(float(rv_val))
        ids.append(int(id))
        roi_labels.append(label)
        id += 1

# rv_copy = np.zeros(atlas_img.shape)
hr_copy = np.zeros(atlas_img.shape)

if not d4:
    for id in ids:
        # rv_copy[atlas_img == (id+1)] = rv[id]
        # hr_copy[atlas_img == (id+1)] = hr[id]
        hr_copy[atlas_img == (id+1)] = s_att3[0][id]
        print(s_att3[0][id])
else:
    hr = np.array(hr)
    # rv = np.array(rv)

    #####TRACTSEG or AAN#####
    # rv_copy = atlas_img * rv[None, None, None, ...]
    hr_copy = atlas_img * hr[None, None, None, ...]

    for ex in exclude:
        # rv_copy = np.delete(rv_copy, ex, 3)
        hr_copy = np.delete(hr_copy, ex, 3)

    # rv_copy = np.max(rv_copy, axis=3)
    hr_copy = np.max(hr_copy, axis=3)


# save new img
# rv_img = nib.Nifti1Image(rv_copy, atlas.affine, atlas.header)
# rv_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/single-roi-models/{}_rv_mean_pearson.nii.gz'.format(atlas_id)
# nib.save(rv_img, rv_path)
hr_img = nib.Nifti1Image(hr_copy, atlas.affine, atlas.header)
hr_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/IPMI2021/{}_rv_mean_pearson.nii.gz'.format(atlas_id)
nib.save(hr_img, hr_path)


