
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
from sklearn.preprocessing import normalize


from sklearn.metrics import explained_variance_score, r2_score
from sklearn.linear_model import Ridge

#from snapml import LinearRegression
#from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
from scipy.stats import loguniform


n_train = 375
n_test = 125

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

cluster = False
if cluster:
    path = '/home/sheczko/ptmp/data/' #global path for cluster
else:
    path = 'data/' ##local path for local computations

CT = 'tangent' #set the correlation type

Feature_selection = True ##set the whether to use the feature selection trick based on the education scores

prop = False ##set the whether to use the feature selection trick based on the education scores

bias_reduct = True #reduce bias ?
n_loops = 3


fit_intercept = True


csv_paths  = glob.glob(path + f'/results/connectomes/{CT}_gabys/*.csv')
print(csv_paths)

folds_gaby = pd.read_csv(path + f'/manual_folds/new/folds_{n_train}_{n_test}.txt')
folds_gaby = folds_gaby[~folds_gaby['#----------------------------------------'].str.contains('------------------------------------')]
folds_gaby2 = folds_gaby[~folds_gaby['#----------------------------------------'].str.contains('set')]
folds_gaby2 = folds_gaby[~folds_gaby['#----------------------------------------'].str.contains('#')]





##let's find the paths to the the csv connectivity data we ahve
#data_paths = glob.glob(path + '/results/connectomes/*tangent*.csv')

##load up regresors
regressors_df =  regresson.load_regressors(path + 'func_images/AOMIC/regressors/*.txt')
##choose the target variables, take as np arrays
GCA = regressors_df.regressor_iq.values
bmi = regressors_df.regressor_bmi.values
edu = regressors_df.regressor_edu.values
#print(edu)
 #concatenate cognitive metrics into single variable
cognition = ['GCA'] #nanems o fthe cog metrics used
#cognition = ['PMAT Correct', 'PMAT Response Time']
cog_metric = np.transpose(np.asarray([GCA, edu])) ##the df with the cog ntirics as columns
print(f'cog metric shape {cog_metric.shape}')
#set the number of permutations you want to perform
perm = 100
#set the number of cross-validation loops you want to perform
cv_loops = 5
#set the number of folds you want in the inner of the nested cross-validation
k = 3
#set the proportion of data you want in your training set
train_size = .8
#set the number of variable you want to predict to be the number of variables stored in the cognition variablse
n_cog = np.size(cognition)
#set regression model type
regr = Ridge(fit_intercept = fit_intercept, max_iter=1000000)
#regr = LinearRegression(fit_intercept = True, use_gpu=False, max_iter=1000000,dual=True,penalty='l2')
#set y to be the cognitive metrics you want to predict. They are the same for every atlas (each subject has behavioural score regradless of parcellation)
Y = cog_metric

 #set hyperparameter grid space you want to search through for the model
#alphas = np.linspace(max(n_feat*0.12 - 1000, 0.0001), n_feat*0.12 + 2000, num = 50, endpoint=True, dtype=None, axis=0) #set the range of alpahs being searhced based off the the amount of features
alphas = loguniform(10, 10e3)
n_iter = 50

column_names_pred = []
column_names_real = []    


for perm_ixd in range(perm):
    for cog in cognition:
        column_names_pred.append(f'{cog}_perm_{perm_ixd + 1}_pred')
        column_names_real.append(f'{cog}_perm_{perm_ixd + 1}_real')

#print(data_paths[:4])     
# 
for n_feat in np.array([2250]):   

    for data_path_i, data_path in enumerate(csv_paths): ##loop over atlases
            
        current_path = data_path
        #current_path = csv_paths[4]
        current_atlas = current_path.split('/')[-1].split('_')[-1].split('.')[0] #change this gives v short names for some of the altases
        print(f'current ' + current_atlas)

        if current_atlas in ('atlas-Schaefer1000','atlas-Slab1068'):

            print('skipping this atlas too many features:(')
            continue

        fc = regresson.load_data(current_path) ##set the imput variable to the current atlas connectome, gives subjects x features matrix

        #set x data to be the input variable you want to use (we use always fc)
        
        X = fc
        X[X<0] = 0 #filter the negative values from the correlations
        #set the number of features 
        if Feature_selection:
            if prop:
                n_feat = int(X.shape[1]/2)
            else:
                n_feat = n_feat
        else:
            n_feat = X.shape[1]

        print(f'n_feat input: {n_feat}')

        r2_iq_fMRI_preds, r2_iq_edu_preds, r2_iq_avg_preds, r2_iq_resid_preds, r2_preds_edu, corr_iq_fMRI_preds, corr_iq_edu_preds, corr_iq_avg_preds, corr_iq_resid_preds,corr_preds_edu, n_pred, cogtest, featimp,preds, preds2, preds3,var,opt_alpha = regresson.regression(X = X, Y = Y, perm = perm, cv_loops = cv_loops, k = k, train_size = 0.8, n_cog = n_cog, regr = regr, alphas = alphas,n_feat = n_feat,
        cognition = cognition, n_iter_search=n_iter,Feature_selection = Feature_selection,manual_folds = True,prop = prop,fold_list = folds_gaby2,n_test = n_test,n_train = n_train,z_score = False,bias_reduct = bias_reduct,n_loops = n_loops,fit_intercept = fit_intercept)

        
              ##save data:
#1 make df of the predicted values 
        df_preds = pd.DataFrame(preds.reshape(perm * n_cog,n_pred).T,columns = column_names_pred) ## we flatten the permutation axis 

        df_real = pd.DataFrame(cogtest.reshape(perm * n_cog,n_pred).T,columns = column_names_real)
        preds_real_df = pd.concat([df_preds,df_real],axis = 1, sort = True)

#2 make df of the statistical values

        result_r2 = pd.DataFrame(columns = [cog + '_r2' for cog in cognition], data = r2_iq_fMRI_preds)
        result_r2_edu = pd.DataFrame(columns = [cog + '_r2_using_only_edu' for cog in cognition], data = r2_iq_edu_preds)
        result_r2_2 = pd.DataFrame(columns = [cog + '_r2_averaged_FA' for cog in cognition], data = r2_iq_avg_preds)
        result_r2_resid = pd.DataFrame(columns = [cog + '_r2_after_controlling_residuals' for cog in cognition], data = r2_iq_resid_preds)

        result_r2_pred_edu = pd.DataFrame(columns = [cog + '_r2_to_edu_pred' for cog in cognition], data = r2_preds_edu)


        result_corr = pd.DataFrame(columns = [cog + '_corr' for cog in cognition], data = corr_iq_fMRI_preds)
        result_corr_edu = pd.DataFrame(columns = [cog + '_corr_using_only_edu' for cog in cognition], data = corr_iq_edu_preds)
        result_corr_2 = pd.DataFrame(columns = [cog + '_corr_averaged_FA' for cog in cognition], data = corr_iq_avg_preds)
        result_corr_resid = pd.DataFrame(columns = [cog + '_corr_after_controlling_residuals' for cog in cognition], data = corr_iq_resid_preds)
        
        result_corr_pred_edu = pd.DataFrame(columns = [cog + '_corr_edu_pred' for cog in cognition], data = corr_preds_edu)



        result_var = pd.DataFrame(columns = [cog + '_var' for cog in cognition], data = var)
        opt_alphas_df = pd.DataFrame(columns = [cog + '_opt_alphas' for cog in cognition], data =  opt_alpha)




        result_df = pd.concat([result_var,result_r2,result_r2_edu,result_r2_2,result_r2_resid,result_r2_pred_edu,result_corr,result_corr_edu,result_corr_2,result_corr_resid,result_corr_pred_edu,opt_alphas_df],axis = 1)

        result_df.to_csv(path + f'results/ridge_regression/{CT}/gaby_results/ridge_results_yFS_n_feat_{n_feat}_bias_reduct_{bias_reduct}_prop_{prop}_{CT}_{current_atlas}_fold_size_{n_train}.csv')
        preds_real_df.to_csv(path + f'results/ridge_regression/{CT}/gaby_results/ridge_preds_yFS_n_feat_{n_feat}_bias_reduct_{bias_reduct}_prop_{prop}_{CT}_{current_atlas}_fold_size_{n_train}.csv')