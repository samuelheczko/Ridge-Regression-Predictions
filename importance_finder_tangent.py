
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
from sklearn.utils.fixes import loguniform



from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

cluster = True
if cluster:
    path = '/home/sheczko/ptmp/data/' #global path for cluster
else:
    path = 'data/' ##local path for local computations

CT = 'tangent' #set the correlation type


csv_paths  = glob.glob(path + f'/results/connectomes/{CT}_relevant/*.csv')
print(csv_paths)



##let's find the paths to the the csv connectivity data we ahve
#data_paths = glob.glob(path + '/results/connectomes/*tangent*.csv')

CT = 'tangent'

csv_paths  = glob.glob(path + f'/results/connectomes/{CT}_relevant/*.csv')

column_names_importance = []


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

#set the number of permutations you want to perform
perm = 50
#set the proportion of data you want in your training set
train_size = .8
#set the number of variable you want to predict to be the number of variables stored in the cognition variablse
n_cog = np.size(cognition)
#set regression model type
regr = Ridge(fit_intercept = True, max_iter=1000000)
#regr = LinearRegression(fit_intercept = True, use_gpu=False, max_iter=1000000,dual=True,penalty='l2')
#set y to be the cognitive metrics you want to predict. They are the same for every atlas (each subject has behavioural score regradless of parcellation)
Y = cog_metric




for perm_ixd in range(perm):
    for cog in cognition:
        column_names_importance.append(f'{cog}_perm_{perm_ixd + 1}_importance')
        #column_names_real.append(f'{cog}_perm_{perm_ixd + 1}_real')

#print(data_paths[:4])        

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
    
    n_feat = X.shape[1]


    feat_imp = regresson.importance_extractor(X, Y, perm, train_size, n_cog,n_feat,cognition) #gives a matrix (n_perm,n_all_features, n_cog)

    ##save data:

    df_feat = pd.DataFrame(feat_imp.reshape(perm * n_cog,n_feat).T,columns = column_names_importance) ## we flatten the permutation axis 
    #df_real = pd.DataFrame(cogtest.reshape(perm * n_cog,n_pred).T,columns = column_names_real)
    #preds_real_df = pd.concat([df_preds,df_real],axis = 1, sort = True)


    df_feat.to_csv(path + f'results/feature_importance/{CT}/feature_importanace_edu_iq_{current_atlas}.csv')