import connectome
import pandas as pd
import numpy as np
import glob
import pandas as pd


from nilearn.maskers import NiftiSpheresMasker
import connectome



##defime global parameters
cluster = True
if cluster:
    path = '/home/sheczko/ptmp/data/' #Cluster
else:
    path = 'data/' ##local
correlation_measures = ['correlation','tangent'] #can be also tangent, partial

#templateICBM = '/kyb/agks/sheczko/Downloads/MastersThesis/code/data/templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii' ##use the ICBM T1 template


if cluster:
    imgs_paths = glob.glob(path + 'func_images/AOMIC/prep_nifti/*.nii') ##load up all subejct images
else:
    imgs_paths = glob.glob(path + 'func_images/AOMIC/prep_nifti/*000*.nii') ##load up a subset of the subejct imagesim

imgs_paths_df = pd.DataFrame(imgs_paths)
imgs_paths_df.to_csv(path + 'results/img_labels.csv') #save the order of the subjects in the images as csv

subjects_idxs = []
for s_n in imgs_paths:
    subjects_idxs.append(s_n.split('_')[-1].split('.')[0]) #split the path to extract only the number of the subject from path
dosenbach = datasets.fetch_coords_dosenbach_2010(ordered_regions = False,legacy_format=False) ##fetc
coords = np.vstack(
    (
    dosenbach.rois['x'],
    dosenbach.rois['y'],
    dosenbach.rois['z'],
    )
).T
print(coords.shape)

for correlation_measure in correlation_measures:
    spheres_masker = NiftiSpheresMasker(
    seeds = coords,
    radius = 4.5,
    standardize = True,
    )
    atlas_name = 'dosenbach160'
    ana_labels = pd.DataFrame(dosenbach.labels).T
    al_p = False
    time_series = []
    for image_i, image in enumerate(imgs_paths):
        time_series_image = spheres_masker.fit_transform(image)
        time_series.append(time_series_image)
    
    
    df_time_series = pd.DataFrame()

##save time series
    for array_i, array in enumerate(time_series):
        df_time_series = pd.concat([df_time_series,pd.DataFrame(data = array.flatten(), columns=['subj_' + str(array_i)] )], axis = 1)#get df with the activations for each subject 
        
    Brn_area_indecies = np.repeat(np.arange(time_series[0].shape[1]),time_series[0].shape[0])
    if al_p:
        df_labels = pd.DataFrame(data = ana_labels.iloc[Brn_area_indecies,1],columns=['anatomical_label'])
        df_labels = df_labels.reset_index()
        df_time_series = pd.concat([df_labels,df_time_series],axis = 1)
    else:
        df_time_series = pd.concat([pd.DataFrame(data = Brn_area_indecies,columns = ['brain_area_index']),df_time_series],axis = 1)

    df_time_series.to_csv(path_or_buf = path + f'/results/time_series/time_series_n_sub-{len(time_series)}_atlas-{atlas_name}.csv') ##SAVE

    correlation_matrices, _ =  connectome.connectome(time_series = time_series,correlation_measure=correlation_measure) #get the connectivity matrices

    df_ = connectome.save_connectomes_df(correlation_matrices,anatomical_label_presence = al_p, anatomic_labels = ana_labels, path_to_save = path + 'results/connectomes/', atlas_name = atlas_name, n_subjects = correlation_matrices.shape[0], correlation_measure = correlation_measure,subject_ixds = subjects_idxs)
        
