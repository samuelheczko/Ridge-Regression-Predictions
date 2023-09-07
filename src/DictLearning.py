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

#script to create the dict learn componetns


path = 'data/' ##local

##add the data


comp_list = np.array([100,200,400]) #define the amaount components to be fit

for n_comp in comp_list:

    dictlearn = DictLearning(
        n_components=n_comp,
        memory_level=2,
        verbose=1,
        random_state = 0,
        n_epochs = 1,
        mask_strategy='whole-brain-template', #included in nilearn
        standardize = True,
        n_jobs = -1)

    dictlearn.fit(imgs_paths)

    dictlearn.components_img_.to_filename(path + f'results/DictMap/DictLearn_space-IBCM_WB_mask2_ncomp_{n_comp}.nii.gz') #save the componets 
