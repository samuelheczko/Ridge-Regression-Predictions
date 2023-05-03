#to get the fc in correct format

import pandas as pd
import numpy as np
import glob

def load_data(path):
    #INPUT: the path to the desired atlas
    #OUTPUT: the subjects x features matrix of funtional connectivity
    fc_panda = pd.read_csv(path) #load in the desired datafile
    sorted_df = fc_panda.sort_index(axis=1) ##sort for subjects in order
    #print(sorted_df.columns)
    #print(f'currently the shape of the array is {sorted_df.shape}. We need to cut the columns which dont contain subject connectome data)')
    fc = sorted_df.loc[:,sorted_df.columns.str.startswith('sub')].T.values # derive the pure connectome as np array where each subjerct has a row, to fit the sklearn convention
    return fc


def load_regressors(path):
    #get the cognitive traits in correct format
    regressors_path = glob.glob(path) ##how to extract the target variables
    #print(regressors_path)
    regressors_df = pd.DataFrame()
    for regressors in regressors_path:
        name_r = regressors.split('/')[-1].split('.')[0]
        #print(name_r)
        regressors_df[f'{name_r}'] = pd.read_csv(regressors,header=None) ##in this format each subject has a row, to test the model we can try to predict the subject id (presumably random)

    ##extract the traget variables

    subj_ids = np.zeros(regressors_df.shape[0])
    #extract subject IDs from subj_data table
    for subj_id_i, subj_id in enumerate(regressors_df['subj'].astype('str')):
        subj_ids[subj_id_i] = int(subj_id.split('-')[-1])
    regressors_df['subj_ids'] = subj_ids

    return regressors_df  #subjects as row, regressors as colums (one of them being the subject id)
