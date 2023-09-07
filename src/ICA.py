import glob
import os
import connectome
import nibabel as nib
import pandas as pd
#import regresson

import sys; sys.path
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import time
from nilearn.decomposition import CanICA


cluster = True

path = '..data/' ##local

##add the data

imgs_paths = glob.glob(path + 'func_images/AOMIC/prep_nifti/*.nii') ##load up all subejct images


comp_list = np.array([100,200,400]) #define the amaount components used in this 

for n_comp in comp_list:

    canica = CanICA(
        mask = path + '/atlases/templates/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii',
        n_components=n_comp, 
        memory='nilearn_cache',
        memory_level=2,
        verbose=10,
        mask_strategy='whole-brain-template',
        random_state=0,
        standardize = True,
        n_jobs = 1)

    canica.fit(imgs_paths)

    canica.components_img_.to_filename(path + f'results/ICA/canICA_space-IBCM_WB_mask2_ncomp_{n_comp}.nii.gz')
