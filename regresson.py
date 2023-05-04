#to get the fc in correct format

import pandas as pd
import numpy as np
import glob
#import relevant libraries
import sys; sys.path
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import explained_variance_score, r2_score
from sklearn.linear_model import Ridge
#from sklearn.kernel_ridge import KernelRidge
#from sklearn.linear_model import Lasso
#from sklearn.svm import SVR
#from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline



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


def regression(X, Y, perm, cv_loops, k, train_size, n_cog, regr, alphas,n_feat,cognition):
    ##input X, Y, amount of permutations done on the cv, k: amont of inner loops ot find optimal alpha, train_size: the proption of training dataset, n_cog: the amount of behavioural vairables tested, model:regression type, alhaps_n; the range of alphas to be searched, n_feat: the amount of features

    


    #create arrays to store variables
    #r^2 - coefficient of determination
    r2 = np.zeros([perm,n_cog])
    #explained variance
    var = np.zeros([perm,n_cog])
    #correlation between true and predicted (aka prediction accuracy)
    corr = np.zeros([perm,n_cog])
    #optimised alpha (hyperparameter)
    opt_alpha = np.zeros([perm,n_cog])
    #predictions made by the model
    preds = np.zeros([perm,n_cog,int(np.ceil(X.shape[0]*(1-train_size)))])
    #true test values for cognition
    cogtest = np.zeros([perm,n_cog,int(np.ceil(X.shape[0]*(1-train_size)))])
    #feature importance extracted from the model
    featimp = np.zeros([perm,n_feat,n_cog])

    #set the param grid be to the hyperparamters you want to search through
    paramGrid ={'alpha': alphas}

    #iterate through permutations
    for p in range(perm):
        #print permutation # you're on
        print('Permutation %d' %(p+1))
        #split data into train and test sets
        x_train, x_test, cog_train, cog_test = train_test_split(X, Y, test_size=1-train_size, 
                                                                shuffle=True, random_state=p)

        
        #iterate through the cognitive metrics you want to predict
        for cog in range (n_cog):

            #print cognitive metrics being predicted 
            print ("Cognition: %s" % cognition[cog])
            
            #set y values for train and test based on     
            y_train = cog_train[:,cog]
            y_test = cog_test[:,cog]
            
            #store all the y_test values in a separate variable that can be accessed later if needed
            cogtest[p,cog,:] = y_test


            #create variables to store nested CV scores, and best parameters from hyperparameter optimisation
            nested_scores = []
            best_params = []
            

            #optimise regression model using nested CV
            print('Training Models')
            
            #go through the loops of the cross validation
            for i in range(cv_loops):


                #set parameters for inner and outer loops for CV
                inner_cv = KFold(n_splits=k, shuffle=True, random_state=i)
                outer_cv = KFold(n_splits=k, shuffle=True, random_state=i)
                
                #define regressor with grid-search CV for inner loop
                gridSearch = GridSearchCV(estimator=regr, param_grid=paramGrid, n_jobs=-1, 
                                        verbose=0, cv=inner_cv, scoring='r2')

                #fit regressor
                gridSearch.fit(x_train, y_train)

                #save parameters corresponding to the best score
                best_params.append(list(gridSearch.best_params_.values()))

                #call cross_val_score for outer loop
                nested_score = cross_val_score(gridSearch, X=x_train, y=y_train, cv=outer_cv, 
                                            scoring='r2', verbose=1)

                #record nested CV scores
                nested_scores.append(np.median(nested_score))

                #print how many cv loops are complete
                print("%d/%d Complete" % (i+1,cv_loops))
                
            #once all CV loops are complete, fit models based on optimised hyperparameters    
            print('Testing Models')


            #save optimised alpha values
            opt_alpha[p,cog] = np.median(best_params)


            #fit model using optimised hyperparameter
            model = Ridge(fit_intercept = True, alpha = opt_alpha[p,cog], max_iter=1000000)

            model.fit(x_train, y_train)
            
            #compute r^2 (coefficient of determination) 
            r2[p,cog]=model.score(x_test,y_test)

            #generate predictions from model
            preds[p,cog,:] = model.predict(x_test).ravel()
            
            #compute explained variance 
            var[p,cog] = explained_variance_score(y_test, preds[p,cog,:])

            #compute correlation between true and predicted
            corr[p,cog] = np.corrcoef(y_test, preds[p,cog,:])[1,0]

            #extract feature importance
            featimp[p,:,cog] = model.coef_
        
    return r2, preds, var, corr, featimp, cogtest,opt_alpha



