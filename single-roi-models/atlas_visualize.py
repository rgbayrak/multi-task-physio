import nibabel as nib
from nilearn import plotting

path_schaefer = '/data/gm-atlases/Schaefer/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'
path_tian = '/data/gm-atlases/Tian2020MSA/7T_2mm/Tian_Subcortex_S1_7T_2mm.nii.gz'

# # # plot 3D atlas
img_schaefer = nib.load(path_schaefer)
display = plotting.plot_roi(img_schaefer, cut_coords=(8, -4, 9), draw_cross=False, cmap='RdYlGn')
plotting.show()
#
# # plot 4D atlas
path4D = '/data/wm-atlases/pandora_2mm/thresholded/TractSeg-th0.95-2mm_HCP.nii.gz'
img4D = nib.load(path4D)
plotting.plot_prob_atlas(img4D, cut_coords=(8, -4, 9), draw_cross=False, cmap='RdYlGn')
plotting.show()
#
img_tian = nib.load(path_tian)
plotting.plot_roi(img_tian, cut_coords=(8, -4, 9), draw_cross=False, cmap='RdYlGn')
plotting.show()

pass