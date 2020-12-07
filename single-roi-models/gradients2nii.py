import nibabel as nib
import os
import numpy as np

path_atlas = '/data/gm-atlases/Schaefer/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'
atlas = nib.load(path_atlas)
atlas_img = atlas.get_fdata()

path_labels = '/home/bayrakrg/neurdy/pycharm/atlas_processing/schaefer_gradient_info.csv'
with open(path_labels, 'r') as f:
    content = f.readlines()

hr = []
rv = []
ids = []
for i in range(len(content)):
    if i != 0:
        hr.append(float((content[i].split(','))[4]))
        rv.append(float((content[i].split(','))[3]))
        ids.append(int((content[i].split(','))[0]))

rv_copy = atlas_img.copy()
hr_copy = atlas_img.copy()

for id in ids:
    rv_copy[rv_copy == (id+1)] = rv[id]
    hr_copy[hr_copy == (id+1)] = hr[id]

# save new img
rv_img = nib.Nifti1Image(rv_copy, atlas.affine, atlas.header)
rv_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/single-roi-models/Schaefer_rv_mean_pearson.nii.gz'
nib.save(rv_img, rv_path)
hr_img = nib.Nifti1Image(hr_copy, atlas.affine, atlas.header)
hr_path = '/home/bayrakrg/neurdy/pycharm/multi-task-physio/single-roi-models/Schaefer_hr_mean_pearson.nii.gz'
nib.save(hr_img, hr_path)


