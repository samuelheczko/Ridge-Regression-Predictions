import glob
import os
import pandas as pd
import numpy as np


from nilearn import image
from nilearn import plotting
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import MultiNiftiLabelsMasker
from nilearn.maskers import MultiNiftiMasker
from nilearn.maskers import MultiNiftiMapsMasker



#path = '/kyb/agks/sheczko/Downloads/MastersThesis/code/data/' ##change this for the cluster version


def calculate_time_series(atlas_path,imgs_paths,standardise = True,map_atlas = False):
    ##input: the atlas, images (paths), outputs: time series for each subject of each brain region (standardised)
    ##pass the chosen atlas to the multi masker that masks each brain region for each participant and extracts the time series
    if map_atlas:
        masker = MultiNiftiMapsMasker(maps_img = atlas_path, standardize = standardise, n_jobs = -1)
    else:
        masker = MultiNiftiLabelsMasker(labels_img = atlas_path, standardize = standardise, memory='nilearn_cache', n_jobs = -1)
    ##extract the time series from the data
    time_series = masker.fit_transform(imgs_paths)[:]
    return time_series

def connectome(time_series,correlation_measure = 'correlation'):
    ##INPUT: the time series on each brain region as list(?) of arrays (for each participant) given by nilearn, the desired correlation type (choose from linear correaltin, partial correlation, tangent correlation)
    ##OUTPUT: signal (standartised)
    connectome_measure = ConnectivityMeasure(kind = correlation_measure)
    correlation_matrices = connectome_measure.fit_transform(time_series) 
    return correlation_matrices,connectome_measure

def save_connectomes_df(correlation_matrices,anatomical_label_presence, anatomic_labels,path_to_save, atlas_name, n_subjects, correlation_measure,subject_ixds):
    ##INPUT: correlation matrices (all subjects, shape subjects_n x brain_areas x brain_areas), path where to save the csv file, and details to name the file
    ##OUTPUT: the dataframe with eachs subjects connetion strenghts in columns and the brain areas involved
    df_ = pd.DataFrame()
    triangle_index = np.triu_indices(correlation_matrices.shape[1], k = 1) ## get the indecies of the matrix upper triangle (shape : 2 x (n(n-1))/2)
    for subject_ixd in range(correlation_matrices.shape[0]):
        df_[f'subject_{subject_ixds[subject_ixd]}'] = correlation_matrices[subject_ixd][np.triu_indices(correlation_matrices.shape[1], k = 1)] #take the upper triangle from the subject specific matrix, to go back to matrix https://stackoverflow.com/questions/17527693/transform-the-upper-lower-triangular-part-of-a-symmetric-matrix-2d-array-into/58806626#58806626
    
    if anatomical_label_presence: ##if we have anatomical labels for the brain areas the atlas considered
        df_['brain_area_1'] = list(anatomic_labels['anatomical_label'].iloc[triangle_index[0]])
        df_['brain_area_2'] = list(anatomic_labels['anatomical_label'].iloc[triangle_index[1]])
    
    else: ##if we have anatomical labels for the brain areas the atlas considered
        df_['brain_area_1 (index)'] = list(triangle_index[0] + 1)
        df_['brain_area_2 (index)'] = list(triangle_index[1] + 1)
    
    df_.to_csv(path_or_buf = path_to_save + f'n_sub-{n_subjects}_correlationType-{correlation_measure}_atlas-{atlas_name}.csv')

    return df_

    

#def plot_connectome(atlas,connectome:
#coords = plotting.find_parcellation_cut_coords(labels_img = atlas))






