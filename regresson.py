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
#from snapml import LinearRegression
from sklearn.utils.fixes import loguniform

from scipy.stats import uniform
from scipy.stats import randint

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sklearn.cross_decomposition import PLSRegression

from scipy.stats import uniform


from sklearn.feature_selection import r_regression




#from sklearn.kernel_ridge import KernelRidge
#from sklearn.linear_model import Lasso
#from sklearn.svm import SVR
#from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, RandomizedSearchCV
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


def regression(X, Y, perm, cv_loops, k, train_size, n_cog, regr, alphas,n_feat,cognition,n_iter_search,random = True,Feature_selection = True):
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
    #paramGrid ={'regularizer': alphas}
    paramGrid ={'alpha': alphas}
    n_iter_search = n_iter_search

    #iterate through permutations
    for p in range(perm):
        #print permutation # you're on
        print('Permutation %d' %(p+1))
        #split data into train and test sets
        x_train, x_test, cog_train, cog_test = train_test_split(X, Y, test_size=1-train_size, 
                                                                shuffle=True, random_state=p)
        #print(cog_train)

        if Feature_selection:

            #print(cog_train[:,2])
            w_edu = r_regression(x_test,cog_test[:,-1])

            w_cog = r_regression(x_train,cog_train[:,0])
            w_prod = w_cog * w_edu
            w_prod[w_prod < 0] = 0
            w_prod_norm = (w_prod - np.min(w_prod))/(np.max(w_prod)-np.min(w_prod))
            #w_prod_norm[w_prod_norm > 0].shape
            h_idx = np.argpartition(w_prod_norm,-n_feat)[-n_feat:]


            x_train = x_train[:,h_idx] ##Select the highest features
            x_test = x_test[:,h_idx]
            print('feautres selected')

        
        #iterate through the cognitive metrics you want to predict
        for cog in range (n_cog):
            print(f'ncog: {cog}')

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

                
                
               
                
                if random:
                    #define regressor with random-search CV for inner loop
                    gridSearch = RandomizedSearchCV(estimator=regr, param_distributions = paramGrid, n_jobs=-1, n_iter=n_iter_search,
                                            verbose=0, cv=inner_cv, scoring='r2')

                    #fit regressor
                    gridSearch.fit(x_train, y_train)
                
                else:
                    #define regressor with random-search CV for inner loop
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
            opt_alpha[p,cog] = np.mean(best_params)


            #fit model using optimised hyperparameter
            #model = LinearRegression(fit_intercept = True, regularizer = opt_alpha[p,cog],use_gpu=False, max_iter=1000000,dual=True,penalty='l2')
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
            print (var)

            #extract feature importance
            featimp[p,:,cog] = model.coef_
        
    return r2, preds, var, corr, featimp, cogtest,opt_alpha, y_test.shape[0]





def regressionSVR(X, Y, perm, cv_loops, k, train_size, n_cog, regr, params,n_feat,cognition,n_iter_search,random = True):
    ##input X, Y, amount of permutations done on the cv, k: amont of inner loops ot find optimal alpha, train_size: the proption of training dataset, n_cog: the amount of behavioural vairables tested, model:regression type, alhaps_n; the range of alphas to be searched, n_feat: the amount of features


    #create arrays to store variables
    #r^2 - coefficient of determination
    r2 = np.zeros([perm,n_cog])
    #explained variance
    var = np.zeros([perm,n_cog])
    #correlation between true and predicted (aka prediction accuracy)
    corr = np.zeros([perm,n_cog])
    #optimised alpha (hyperparameter)
    opt_params = np.zeros([perm,n_cog,len(params)])
    #predictions made by the model
    preds = np.zeros([perm,n_cog,int(np.ceil(X.shape[0]*(1-train_size)))])
    #true test values for cognition
    cogtest = np.zeros([perm,n_cog,int(np.ceil(X.shape[0]*(1-train_size)))])
    #feature importance extracted from the model
    featimp = np.zeros([perm,n_feat,n_cog])

    #set the param grid be to the hyperparamters you want to search through
    #paramGrid ={'regularizer': alphas}
    param_dist = params
    n_iter_search = n_iter_search

    #iterate through permutations
    for p in range(perm):
        #print permutation # you're on
        print('Permutation %d' %(p+1))
        #split data into train and test sets
        x_train, x_test, cog_train, cog_test = train_test_split(X, Y, test_size=1-train_size, 
                                                                shuffle=True, random_state=p)
        scaler = StandardScaler()
        x_train_scaled = x_train
        x_test_scaled = x_test
        
        #x_train_scaled = scaler.fit_transform(x_train)
        #x_test_scaled = scaler.fit_transform(x_test)

        
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
            best_params = np.zeros([len(params),cv_loops])
            

            #optimise regression model using nested CV
            print('Training Models')
            
            #go through the loops of the cross validation
    
            for i in range(cv_loops):


                #set parameters for inner and outer loops for CV
                inner_cv = KFold(n_splits=k, shuffle=True, random_state=i)
                outer_cv = KFold(n_splits=k, shuffle=True, random_state=i)

                
            
                #define regressor with random-search CV for inner loop
                random_search = RandomizedSearchCV(estimator=regr, param_distributions = param_dist, n_jobs=-1, n_iter=n_iter_search,
                                        verbose=0, cv=inner_cv, scoring='r2')

                #fit regressor
                random_search.fit(x_train, y_train)
    
                
                

                #save parameters corresponding to the best score
                best_params[0,i] = random_search.best_params_['C']
                best_params[1,i] = random_search.best_params_['epsilon']

                #call cross_val_score for outer loop
                nested_score = cross_val_score(random_search, X=x_train, y=y_train, cv=outer_cv, 
                                            scoring='r2', verbose=1)

                #record nested CV scores
                nested_scores.append(np.median(nested_score))

                #print how many cv loops are complete
                print("%d/%d Complete" % (i+1,cv_loops))
                
            #once all CV loops are complete, fit models based on optimised hyperparameters    
            print('Testing Models')


            #save optimised alpha values
            opt_params[p,cog,:] = np.mean(best_params,axis = 1)


            #fit model using optimised hyperparameter
            #model = LinearRegression(fit_intercept = True, regularizer = opt_alpha[p,cog],use_gpu=False, max_iter=1000000,dual=True,penalty='l2')
            model = SVR(max_iter=10000,kernel = 'linear', C = opt_params[p,cog,0], epsilon = opt_params[p,cog,1])

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
            #featimp[p,:,cog] = model.coef_
        
    return r2, preds, var, corr, cogtest, opt_params, y_test.shape[0]


def regressionPLS(X, Y, perm, cv_loops, k, train_size, n_cog, regr, params, n_feat, cognition, n_iter_search):

      ##input X, Y, amount of permutations done on the cv, k: amont of inner loops ot find optimal alpha, train_size: the proption of training dataset, n_cog: the amount of behavioural vairables tested, model:regression type, alhaps_n; the range of alphas to be searched, n_feat: the amount of features


    #create arrays to store variables
    #r^2 - coefficient of determination
    r2 = np.zeros([perm,n_cog])
    #explained variance
    var = np.zeros([perm,n_cog])
    #correlation between true and predicted (aka prediction accuracy)
    corr = np.zeros([perm,n_cog])
    #optimised alpha (hyperparameter)
    opt_params = np.zeros([perm,n_cog,len(params)])
    #predictions made by the model
    preds = np.zeros([perm,n_cog,int(np.ceil(X.shape[0]*(1-train_size)))])
    #true test values for cognition
    cogtest = np.zeros([perm,n_cog,int(np.ceil(X.shape[0]*(1-train_size)))])
    #feature importance extracted from the model
    featimp = np.zeros([perm,n_feat,n_cog])

    #set the param grid be to the hyperparamters you want to search through
    #paramGrid ={'regularizer': alphas}
    param_dist = params
    n_iter_search = n_iter_search

    
    # iterate through permutations
    for p in range(perm):
        #print permutation # you're on
        print('Permutation %d' %(p+1))
        #split data into train and test sets
        x_train, x_test, cog_train, cog_test = train_test_split(X, Y, test_size=1-train_size, 
                                                                shuffle=True, random_state=p)
        scaler = StandardScaler()
        x_train_scaled = x_train
        x_test_scaled = x_test
        
        #x_train_scaled = scaler.fit_transform(x_train)
        #x_test_scaled = scaler.fit_transform(x_test)

        
        
        # iterate through the cognitive metrics you want to predict
        for cog in range(n_cog):
            #print cognitive metrics being predicted 
            print ("Cognition: %s" % cognition[cog])
            
            #set y values for train and test based on     
            y_train = cog_train[:,cog]
            y_test = cog_test[:,cog]
            
            #store all the y_test values in a separate variable that can be accessed later if needed
            cogtest[p,cog,:] = y_test


            
            #go through the loops of the cross validation
            
            # create variables to store nested CV scores and best parameters from hyperparameter optimization
            nested_scores = []
            best_params = np.zeros([len(params), cv_loops])
            
            # optimise regression model using nested CV
            print('Training Models')
            
            # go through the loops of cross-validation
            for i in range(cv_loops):

                #set parameters for inner and outer loops for CV
                inner_cv = KFold(n_splits=k, shuffle=True, random_state=i)
                outer_cv = KFold(n_splits=k, shuffle=True, random_state=i)
                # ... existing code ...
                
                # define regressor with random-search CV for inner loop
                random_search = RandomizedSearchCV(estimator=regr, param_distributions=params, n_jobs=-1, n_iter=n_iter_search,
                                                   verbose=0, cv=inner_cv, scoring='r2')
                
                # fit regressor
                random_search.fit(x_train, y_train)
                
                # save parameters corresponding to the best score
                best_params[:, i] = list(random_search.best_params_.values())
                print(f'best params: {best_params[:, i]}')

                # call cross_val_score for outer loop
                #nested_score = cross_val_score(random_search, X=x_train, y=y_train, cv=outer_cv,
                                               #scoring='r2', verbose=1)

                # record nested CV scores
                #nested_scores.append(np.median(nested_score))

                # print how many CV loops are complete
                print("%d/%d Complete" % (i + 1, cv_loops))
            
            # once all CV loops are complete, fit models based on optimized hyperparameters    
            print('Testing Models')

            # save optimized parameters
            opt_params[p, cog, :] = np.mean(best_params, axis=1)
            print(f'opt params{int(opt_params[p, cog])}')
            # fit PLS model using optimized hyperparameters
            model = PLSRegression(n_components=int(opt_params[p, cog]))
            model.fit(x_train, y_train)
            
            # compute r^2 (coefficient of determination)
            r2[p, cog] = model.score(x_test, y_test)
            
            # generate predictions from model
            preds[p, cog, :] = model.predict(x_test).ravel()
            
            # compute explained variance
            var[p, cog] = explained_variance_score(y_test, preds[p, cog, :])
            
            # compute correlation between true and predicted
            corr[p, cog] = np.corrcoef(y_test, preds[p, cog, :])[1, 0]
            
            # extract feature importance
            # featimp[p, :, cog] = model.coef_
    
    return r2, preds, var, corr, cogtest, opt_params, y_test.shape[0]
