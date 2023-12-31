
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



from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold


path = '../data/' ##local path for local computations

CT = 'pearson'

Feature_selection = True ##set the whether to use the feature selection trick based on the education scores


csv_paths  = glob.glob(path + f'/results/connectomes/HCP/*{CT}*.csv') #find the csv file, in this case again the CT doesnt matter because we are using the same 
print(csv_paths)

n_features_array = np.arange(250,3250,250)



##let's find the paths to the the csv connectivity data we ahve
#data_paths = glob.glob(path + '/results/connectomes/*tangent*.csv')

##load up regresors
regressors_df =  regresson.load_regressors_HCP(path + '/results/connectomes/HCP/regressors/*.txt')
##choose the target variables, take as np arrays
GCA = regressors_df.regressor_iq.values
edu = regressors_df.regressor_edu.values
#print(edu)
 #concatenate cognitive metrics into single variable
cognition = ['GCA'] #nanems o fthe cog metrics used
#cognition = ['PMAT Correct', 'PMAT Response Time']
cog_metric = np.transpose(np.asarray([GCA, edu])) ##the df with the cog ntirics as columns
print(f'cog metric shape {cog_metric.shape}')
#set the number of permutations you want to perform
perm = 50
#set the number of cross-validation loops you want to perform
cv_loops = 5
#set the number of folds you want in the inner and outer folds of the nested cross-validation
k = 3
#set the proportion of data you want in your training set
train_size = .8
#set the number of variable you want to predict to be the number of variables stored in the cognition variablse
n_cog = np.size(cognition)
#set regression model type
##model parameters
fit_intercept = False #the hcp dataset is z scored, no need for intercept,
regr = Ridge(fit_intercept = fit_intercept, max_iter=1000000) # set regression model type

#regr = LinearRegression(fit_intercept = True, use_gpu=False, max_iter=1000000,dual=True,penalty='l2')
#set y to be the cognitive metrics you want to predict. They are the same for every atlas (each subject has behavioural score regradless of parcellation)
Y = cog_metric
#set hyperparameters for distribution you want to search through for the model

alphas = loguniform(10, 10e3) #define distribution
n_iter = 100 #amounf of random guesses
##model parameters
n_train = int(cog_metric.shape[0] * train_size)
n_test = int(cog_metric.shape[0] * (1 - train_size))

column_names_pred = []
column_names_real = []    


for perm_ixd in range(perm):
    for cog in cognition:
        column_names_pred.append(f'{cog}_perm_{perm_ixd + 1}_pred')
        column_names_real.append(f'{cog}_perm_{perm_ixd + 1}_real')

#print(data_paths[:4])     
# 
for n_feat in n_features_array:   

    for data_path_i, data_path in enumerate(csv_paths): ##loop over atlases
            
        current_path = data_path
        #current_path = csv_paths[4]
        current_atlas = current_path.split('/')[-1].split('_')[-1].split('.')[0] #change this gives v short names for some of the altases
        print(f'current ' + current_atlas)

        fc = regresson.load_data(current_path) ##set the imput variable to the current atlas connectome, gives subjects x features matrix

        #set x data to be the input variable you want to use (we use always fc)
        
        X = fc
        X[X<0] = 0 #filter the negative values from the correlations
        #set the number of features 
        if Feature_selection:
            n_feat = n_feat
        else:
            n_feat = X.shape[1]



        r2_iq_fMRI_preds, r2_iq_edu_preds, r2_iq_avg_preds, r2_iq_resid_preds, r2_preds_edu, corr_iq_fMRI_preds, corr_iq_edu_preds, corr_iq_avg_preds, corr_iq_resid_preds,corr_preds_edu, n_pred, cogtest, featimp,preds, preds2, preds3,var,opt_alpha = regresson.regression(X = X, Y = Y, perm = perm, cv_loops = cv_loops, k = k, train_size = train_size, n_cog = n_cog, regr = regr, alphas = alphas,n_feat = n_feat,
        cognition = cognition,manual_folds = False, n_iter_search=n_iter,Feature_selection = Feature_selection,n_train = n_train, n_test = n_test, z_score = False,fit_intercept = False,bias_reduct = True)

        
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

        if Feature_selection:
            result_df.to_csv(path + f'results/ridge_regression/{CT}/feat_select/ridge_results_FStd_50perm_n_feat_{n_feat}_both_{CT}_{current_atlas}_train_size_{train_size}.csv')
            preds_real_df.to_csv(path + f'results/ridge_regression/{CT}/feat_select/ridge_preds_FStd_50_perm_n_feat_{n_feat}_both_{CT}_{current_atlas}_train_size_{train_size}.csv')
        else:
            result_df.to_csv(path + f'results/ridge_regression/{CT}/feat_select/ridge_results_FStd_50perm_n_feat_{n_feat}_both_{CT}_{current_atlas}_train_size_{train_size}.csv')
            preds_real_df.to_csv(path + f'results/ridge_regression/{CT}/feat_select/ridge_preds_FStd_50_perm_n_feat_{n_feat}_both_{CT}_{current_atlas}_train_size_{train_size}.csv')