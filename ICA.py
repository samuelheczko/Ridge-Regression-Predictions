import glob
import os
import connectome
import nibabel as nib
import pandas as pd
import regresson

import sys; sys.path
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import time
from nilearn.decomposition import CanICA


cluster = True
if cluster:
    path = '/home/sheczko/ptmp/data/' #Cluster
else:
    path = 'data/' ##local

##add the data
if cluster:
    imgs_paths = glob.glob(path + 'func_images/AOMIC/prep_nifti/*.nii') ##load up a subset of the subejct images
else:
    imgs_paths = glob.glob(path + 'func_images/AOMIC/prep_nifti/*000*.nii') ##load up all the subejct images

comp_list = np.array([100,200,400]) #define the amaount components used in this 

for n_comp in comp_list:

    canica = CanICA(
        n_components=n_comp, 
        memory='nilearn_cache',
        memory_level=2,
        verbose=10,
        mask_strategy='background',
        random_state=0,
        standardize = True,
        n_jobs = -1)
    canica.fit(imgs_paths)

    canica.components_img_.to_filename(path + f'results/ICA/canICA_space-IBCM_ncomp_{n_comp}.nii.gz')
