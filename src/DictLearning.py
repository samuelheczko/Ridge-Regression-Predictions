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
from nilearn.decomposition import DictLearning


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

comp_list = np.array([100,200,400]) #define the amaount components to be fit

for n_comp in comp_list:

    dictlearn = DictLearning(
        n_components=n_comp,
        memory_level=2,
        verbose=1,
        random_state = 0,
        n_epochs = 1,
        mask_strategy='whole-brain-template',
        standardize = True,
        n_jobs = -1)

    dictlearn.fit(imgs_paths)

    dictlearn.components_img_.to_filename(path + f'results/DictMap/DictLearn_space-IBCM_WB_mask2_ncomp_{n_comp}.nii.gz')
