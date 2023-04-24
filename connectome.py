import glob

from nilearn import image
from nilearn import plotting
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import MultiNiftiLabelsMasker
from nilearn.maskers import MultiNiftiMasker
from nilearn import image

path = '/kyb/agks/sheczko/Downloads/MastersThesis/code/data/' ##change this for the cluster version

def calculate_correalations(atlas,res,test = True):
    res = f'{res}x{res}x{res}'
    ## 1. choose atlas with desired resolution:
    atlas = path + f'/atlas/atlases/label/Human/' + atlas + '_space-MNI152NLin6_res-' + res + '.nii.gz'
    if test: ##choose test if needed
        imgs_paths = glob.glob(path + 'prep_000*.nii')
    else:
        imgs_paths = glob.glob(path + '*.nii')
    ##pass the chosen atlas to the multi masker that masks each brain region for each participant and extracts the time series
    masker = MultiNiftiLabelsMasker(labels_img = atlas, standardize= True, memory='nilearn_cache', n_jobs = -1)
    ##extract the time series from the data
    time_series = masker.fit_transform(imgs_paths)
    return time_series

def connectome(time_series,correlation_measure = 'correlation'):
    connectome_measure = ConnectivityMeasure(correlation_measure)
    correlation_matrices = connectome_measure.fit_transform(time_series)
    return correlation_matrices




