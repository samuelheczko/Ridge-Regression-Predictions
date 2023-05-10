import glob
import os
import connectome
import nibabel as nib
import pandas as pd

import sys; sys.path
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import explained_variance_score, r2_score
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
##defime global parameters

cluster = True
if cluster:
    path = '/home/sheczko/ptmp/data/' #Cluster
else:
    path = 'data/' ##local

correlation_measure = 'correlation' #can be also tangent, partial
res_n = 1
res = f'{res_n}x{res_n}x{res_n}'
correlation_measure='correlation' ##set for calculation of the brain connectome, choose from 'correlation', 'tangent', 'partial' as implemented by nilearn
#templateICBM = '/kyb/agks/sheczko/Downloads/MastersThesis/code/data/templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii' ##use the ICBM T1 template


##add the data and names of thigs
if cluster:
    imgs_paths = glob.glob(path + 'func_images/AOMIC/prep_nifti/*.nii') ##load up a subset of the subejct images
else:
    imgs_paths = glob.glob(path + 'func_images/AOMIC/prep_nifti/*000*.nii') ##load up all the subejct images
subjects_idxs = []
for s_n in imgs_paths:
    subjects_idxs.append(s_n.split('_')[-1].split('.')[0]) #split the path to extract only the number of the subject from path



atlases = glob.glob(path + '/atlases/lawrance2021/label/Human/ICBM/*.nii.gz') ##get the atlases
anatomical_labels = glob.glob(path + '/atlases/lawrance2021/label/Human/Anatomical-labels-csv/*.csv') #get the anatomical labels (where available)
anatomical_label_names = []
for a_l in anatomical_labels:
    anatomical_label_names.append(a_l.split('/')[-1].split('.')[0]) #split the path to extract only the name of the atlas


##loop over atlases


for atlas_path in atlases: ##loop over atlases
    atlas_name = (atlas_path.split('/')[-1].split('.')[0].split('_')[0]) #split the atlas path so we
    print(atlas_name)
    al_p = (any(n == atlas_name for n in anatomical_label_names)) ##find wheter we have the anatomical labellings for this atlas

    if al_p:
        anatomic_path = path + f'/atlases/lawrance2021/label/Human/Anatomical-labels-csv/' + atlas_name + '.csv'
        ana_labels = pd.read_csv(anatomic_path,names=colnames, header=None)
        ana_labels = ana_labels[ana_labels['idx'] != 0]
    else:
        ana_labels = None


    time_series = connectome.calculate_time_series(atlas_path = atlas_path,imgs_paths=imgs_paths)     ##get the time series of the from all subjects, using the atlas defined
    print(f'we have {len(time_series)} subjects and {time_series[0].shape} shaped time series')
    df_time_series = pd.DataFrame()
    
    ##save time series
    for array_i, array in enumerate(time_series):
        df_time_series = pd.concat([df_time_series,pd.DataFrame(data = array.flatten(), columns=['subj_' + str(array_i)] )], axis = 1)
        
    Brn_area_indecies = np.repeat(np.arange(time_series[0].shape[1]),time_series[0].shape[0])
    if al_p:
        df_labels = pd.DataFrame(data = ana_labels.iloc[Brn_area_indecies,1],columns=['anatomical_label'])
        df_labels = df_labels.reset_index()
        df_time_series = pd.concat([df_labels,df_time_series],axis = 1)
    else:
        df_time_series = pd.concat([pd.DataFrame(data = Brn_area_indecies,columns = ['brain_area_index']),df_time_series],axis = 1)

        
        


    
    df_time_series.to_csv(path_or_buf = path + f'/results/time_series/time_series_n_sub-{len(time_series)}_atlas-{atlas_name}.csv')




