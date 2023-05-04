
import glob
import os
import connectome
import nibabel as nib
import pandas as pd
import regresson

import sys; sys.path
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import explained_variance_score, r2_score
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

cluster = True
if cluster:
    path = '/home/sheczko/ptmp/data/' #global path for cluster
else:
    path = 'data/' ##local path for local computations




##let's find the paths to the the csv connectivity data we ahve
data_paths = glob.glob(path + '/results/connectomes/*.csv')

##load up regresors
regressors_df =  regresson.load_regressors(path + 'func_images/AOMIC/regressors/*.txt')
##choose the target variables, take as np arrays
GCA = regressors_df.regressor_iq.values
bmi = regressors_df.regressor_bmi.values

 #concatenate cognitive metrics into single variable
cognition = ['GCA','bmi']
#cognition = ['PMAT Correct', 'PMAT Response Time']
cog_metric = np.transpose(np.asarray([GCA, bmi]))

#set the number of permutations you want to perform
perm = 100
    #set the number of cross-validation loops you want to perform
cv_loops = 5
#set the number of folds you want in the inner and outer folds of the nested cross-validation
k = 3
#set the proportion of data you want in your training set
train_size = .8
#set the number of variable you want to predict to be the number of variables stored in the cognition variablse
n_cog = np.size(cognition)
#set regression model type
regr = Ridge(fit_intercept = True, max_iter=1000000)
#set hyperparameter grid space you want to search through for the model
alphas = np.linspace(10, 10000, num = 100, endpoint=True, dtype=None, axis=0)
#set y to be the cognitive metrics you want to predict. They are the same for every atlas (each subject has behavioural score regradless of parcellation)
Y = cog_metric


for data_path_i, data_path in enumerate(data_paths): ##loop over atlases
    
    cuurent_path = data_paths[data_path_i+1]
    current_atlas = cuurent_path.split('/')[-1].split('-')[-1].split('.')[0]
    print(f'current atlas: ' + current_atlas)

    fc = regresson.load_data(cuurent_path) ##set the imput variable to the current atlas connectome, gives subjects x features matrix

    #set x data to be the input variable you want to use (we use always fc)
    
    X = fc
    X[X<0] = 0 #filter the negative values from the correlations


    #set the number of features 
    n_feat = X.shape[1]
    r2, preds, var, corr, featimp, cogtest,opt_alphas = regresson.regression(X = X, Y = Y, perm = perm, cv_loops = cv_loops, k = k, train_size = 0.8, n_cog = n_cog, regr = regr, alphas = alphas,n_feat = n_feat,cognition = cognition)
    
    ##save data:
    result_df = pd.DataFrame() ## empty dataframe, filled with colums where each row is one permutation
    preds_df = pd.DataFrame()

    for cog_i, cog_names in enumerate(cognition):
            result_df[f'{cog_names}' + '_r2'] = r2[:,cog_i] #fill in the r2 values
            result_df[f'{cog_names}' + '_var'] = var[:,cog_i] #fill in explained variation values
            result_df[f'{cog_names}' + '_opt_alphas'] = opt_alphas[:,cog_i] #fill in optimal alpha values
            for perm_n in range(preds.shape[0]): 
                preds_df[f'{cog_names}_perm_{perm_n + 1}_preds'] = preds[perm_n,cog_i,:] #fill in preditions
                preds_df[f'{cog_names}_perm_{perm_n + 1}_real'] = cogtest[perm_n,cog_i,:] #fill in real_values
    result_df.to_csv(path + f'results/ridge_regression/ridge_results_atlas-{current_atlas}.csv')
    preds_df.to_csv(path + f'results/ridge_regression/ridge_preds_atlas-{current_atlas}.csv')