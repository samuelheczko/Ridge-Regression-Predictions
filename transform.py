import sys
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import glob
import os
import mne
import connectome
import pandas as pd

from nilearn import image
from nilearn import plotting
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import MultiNiftiLabelsMasker
from nilearn.maskers import MultiNiftiMasker
from nilearn import image
import ants

##transform the atlaes
path_atlases = sys.argv[1]

##1. load the atlases
atlas_list = glob.glob(path_atlases + '*2x2x2*.nii.gz')##load the altases with 1mm res
print(os.path.basename(atlas_list[0]))

##2. load the the transforms
transform = '/kyb/agks/sheczko/Downloads/MastersThesis/neuroparc/atlases/transforms/MNI152NLin6_2_MNI152NLin2009cAsym.h5' ##load the transform downloeaded from https://figshare.com/articles/dataset/MNI_T1_6thGen_NLIN_to_MNI_2009b_NLIN_ANTs_transform/3502238
template = '/kyb/agks/sheczko/Downloads/MastersThesis/code/data/templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii' ##use the ICBM T1 template
fixed = ants.ants_image_io.image_read(template) ##load the template as ants image
transform_ants = ants.ants_transform_io.read_transform(transform) ##load the transform into ants

##3. loop over the atlases, create new file registred into the ICBM space
for atlas_i, atlas_n in enumerate(atlas_list):
    moving_t = ants.ants_image_io.image_read(atlas_n) ##load the image to be transformed, the atlas in this case
    moved = ants.apply_transforms(fixed = fixed,moving = moving_t, transformlist=transform,interpolator = 'genericLabel') ##apply the transform using the interpolator that precieces the labels, genericlabel
    moved.to_file(path_atlases + f'ICBM/{os.path.basename(atlas_n)}')
    print(f'{atlas_i + 1} of {len(atlas_list)} done')




