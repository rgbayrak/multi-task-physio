from nilearn import plotting
from nilearn import datasets
from nilearn import surface

path_schaefer = '/data/gm-atlases/Schaefer/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz'
fsaverage = datasets.fetch_surf_fsaverage()
texture_rh = surface.vol_to_surf(path_schaefer, fsaverage.pial_right)
texture_lh = surface.vol_to_surf(path_schaefer, fsaverage.pial_left)

# plotting.view_img_on_surf(texture)
plotting.plot_surf_roi(fsaverage.infl_right, texture_rh, hemi='right', cmap='RdYlGn',
                            title='Surface right hemisphere', colorbar=True,
                            view='lateral', threshold=1.)
plotting.plot_surf_roi(fsaverage.infl_left, texture_lh, hemi='left', cmap='RdYlGn',
                            title='Surface left hemisphere', colorbar=True,
                            view='lateral', threshold=1.)
plotting.show()