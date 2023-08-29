import connectome
import pandas as pd
import numpy as np
import glob



##defime global parameters

path = 'data/' ##define the location of path here
#correlation_measure = 'correlation' #can be also tangent, partial
res_n = 1
res = f'{res_n}x{res_n}x{res_n}'
#templateICBM = '/kyb/agks/sheczko/Downloads/MastersThesis/code/data/templates/mni_icbm152_nlin_asym_09c_nifti/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii' ##use the ICBM T1 template



##add the data and names of things
imgs_paths = glob.glob(path + 'func_images/AOMIC/prep_nifti/*.nii') ##load up all subejct images, specific to AOMIC -- change for other data
#print(imgs_paths)
subjects_idxs = []
for s_n in imgs_paths:
    subjects_idxs.append(s_n.split('_')[-1].split('.')[0]) #split the path to extract only the number of the subject from path
#print(subjects_idxs)



atlases = glob.glob(path + 'atlases/300_ROI_Set/*.nii') ##get the atlases -- change for other data
print (atlases)
anatomical_labels = glob.glob(path + '/atlases/lawrance2021/label/Human/Anatomical-labels-csv/*.csv') #get the anatomical labels (where available) - specific to neuroparc atlas collection, change if iother data used
anatomical_label_names = []
for a_l in anatomical_labels:
    anatomical_label_names.append(a_l.split('/')[-1].split('.')[0]) #split the path to extract only the name of the atlas

colnames=['idx','anatomical_label'] 

correlation_measures = ['tangent','correlation'] ##define the corrleation measures to be analysed

##loop over correlation measures
for correlation_measure in correlation_measures:
##loop over atlases 

    for atlas_path in atlases: ##loop over atlases
        atlas_name = (atlas_path.split('/')[-1].split('.')[0].split('_')[0]) #split the atlas path so we have the name
        
        print(atlas_name)

        al_p = (any(n == atlas_name for n in anatomical_label_names)) ##find wheter we have the anatomical labellings for this atlas

        if al_p:
            anatomic_path = path + f'/atlases/lawrance2021/label/Human/Anatomical-labels-csv/' + atlas_name + '.csv'
            ana_labels = pd.read_csv(anatomic_path,names=colnames, header=None)
            ana_labels = ana_labels[ana_labels['idx'] != 0]
        else:
            ana_labels = None

	#make the time series
        time_series = connectome.calculate_time_series(atlas_path = atlas_path,imgs_paths=imgs_paths,map_atlas = False)     ##get the time series of the from all subjects, using the atlas defined
        

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
	##save the connectomes
        df_ = connectome.save_connectomes_df(correlation_matrices,anatomical_label_presence = al_p, anatomic_labels = ana_labels, path_to_save = path + 'results/connectomes/', atlas_name = atlas_name, n_subjects = correlation_matrices.shape[0], correlation_measure = correlation_measure,subject_ixds = subjects_idxs)
        

    print('round ran succesfully:)')
