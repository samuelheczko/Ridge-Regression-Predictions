##to be run on nyx
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import glob
import os
import sklearn

from nilearn import image
from nilearn import plotting
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import MultiNiftiLabelsMasker
from nilearn.maskers import MultiNiftiMasker
from nilearn import image

#HYPERPARAMETERS TO DEFINE
cluster = False
atlas = 'Hammersmith'
correlation_measure = 'correlation' #can be also tangent, partial
res = 4

##CODE 
if cluster:    
    path = '/home/sheczko/ptmp/data/'
else:
    path = '/kyb/agks/sheczko/Downloads/MastersThesis/code/data/'

##construct the path for the desired atlas
res = f'{res}x{res}x{res}'
atlas_path = path + f'/atlases/lawrance2021/label/Human/' + atlas + '_space-MNI152NLin6_res-' + res + '.nii.gz'
imgs_paths = glob.glob(path + 'func_images/AOMIC/prep_nifti/*000*.nii')
print(len(imgs_paths))

##nibabel test
nib_image = nib.load(imgs_paths[2])
print(imgs_paths[2])
print(nib_image.shape)
print(nib_image.header)
""" 
##pass the chosen atlas to the multi masker that masks each brain region for each participant and extracts the time series
masker = MultiNiftiLabelsMasker(labels_img = atlas_path, standardize= True, memory='nilearn_cache', n_jobs = -1)
##extract the time series from the data
time_series = masker.fit_transform(imgs_paths)


connectome_measure = ConnectivityMeasure(kind = correlation_measure)
correlation_matrices = connectome_measure.fit_transform(time_series)
print(correlation_matrices.shape)
 """