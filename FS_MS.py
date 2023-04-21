##to be run on the cluster
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import glob
import os
import sklearn
import nibabel as nib

from nilearn import image
from nilearn import plotting
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import MultiNiftiLabelsMasker
from nilearn.maskers import MultiNiftiMasker
cluster = False
if cluster:    
    path = '/home/sheczko/ptmp/AOMIC/prep_nifti/'
else:
    path = '/kyb/agks/sheczko/Downloads/MastersThesis/code/data/'
print(path)

imgs_paths = glob.glob(path + 'prep_00*.nii')

nib_image = nib.load(imgs_paths[0])

print(nib_image.shape)

