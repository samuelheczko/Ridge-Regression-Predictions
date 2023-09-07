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
from sklearn.linear_model import LinearRegression




#from sklearn.kernel_ridge import KernelRidge
#from sklearn.linear_model import Lasso
#from sklearn.svm import SVR
#from sklearn.pipeline import make_pipeline, Pipeline
#from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, RandomizedSearchCV
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline



def load_data(path):

    """
    Load functional connectivity data from a CSV file.

    Parameters:
        path (str): Path to the CSV file containing functional connectivity data.

    Returns:
        fc (numpy.ndarray): Subjects x features matrix of functional connectivity.
    """

    fc_panda = pd.read_csv(path) #load in the desired datafile
    sorted_df = fc_panda.sort_index(axis=1) ##sort for subjects in order
    #print(sorted_df.columns)
    #print(sorted_df.columns)
    #print(f'currently the shape of the array is {sorted_df.shape}. We need to cut the columns which dont contain subject connectome data)')
    fc = sorted_df.loc[:,sorted_df.columns.str.startswith('sub')].T.values # derive the pure connectome as np array where each subjerct has a row, to fit the sklearn convention
    return fc


def load_regressors(path):

    """
    Load cognitive trait data from CSV files.

    Parameters:
        path (str): Path to the directory containing cognitive trait CSV files.

    Returns:
        regressors_df (pandas.DataFrame): Subjects as rows, cognitive traits as columns.
    """

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

def load_regressors_HCP(path):
    #get the cognitive traits in correct format
    regressors_path = glob.glob(path) ##how to extract the target variables
    #print(regressors_path)
    regressors_df = pd.DataFrame()
    for regressors in regressors_path:
        name_r = regressors.split('/')[-1].split('.')[0]
        #print(name_r)
        regressors_df[f'{name_r}'] = pd.read_csv(regressors,header=None) ##in this format each subject has a row, to test the model we can try to predict the subject id (presumably random)



    return regressors_df  #subjects as row, regressors as colums (one of them being the subject id)


def regression(X, Y, perm, cv_loops, k, train_size, n_cog, regr, alphas,n_feat,cognition,n_iter_search,random = True,Feature_selection = True,manual_folds = False,fold_list = None,n_test = None,n_train = None,prop = False,z_score = False,bias_reduct = False,n_loops = 1,fit_intercept = True):
    ##input X, Y, amount of permutations done on the cv, k: amont of inner loops ot find optimal alpha, train_size: the proption of training dataset, n_cog: the amount of behavioural vairables tested, model:regression type, alhaps_n; the range of alphas to be searched, n_feat: the amount of features


    #Perform regression analysis using various models and evaluate performance.

    #Parameters:
        ##input X, Y, amount of permutations done on the cv, k: amont of inner loops ot find optimal alpha
        # train_size: the proption of training dataset, n_cog: the amount of behavioural vairables tested, model:regression type, alhaps_n; the range of alphas to be searched, n_feat: the amount of features

    #Returns:
        # Describe the returned values here
    


    #create arrays to store variables
    #r^2 - coefficient of determination -set up the arrays to hold all the statistical mesures
    r2_iq_fMRI_preds = np.zeros([perm,n_cog])
    r2_iq_avg_preds = np.zeros([perm,n_cog])
    r2_iq_edu_preds = np.zeros([perm,n_cog])
    r2_iq_resid_preds = np.zeros([perm,n_cog])
    r2_preds_edu = np.zeros([perm,n_cog])
    r2_preds_avg_edu = np.zeros([perm,n_cog])
  
    #explained variance
    var = np.zeros([perm,n_cog])
    #correlation between true and predicted (aka prediction accuracy)
    corr_iq_fMRI_preds = np.zeros([perm,n_cog])
    corr_iq_edu_preds = np.zeros([perm,n_cog])
    corr_iq_avg_preds = np.zeros([perm,n_cog])
    corr_iq_resid_preds = np.zeros([perm,n_cog])
    corr_preds_edu =  np.zeros([perm,n_cog])
    corr_preds_avg_edu =  np.zeros([perm,n_cog])



    #optimised alpha (hyperparameter)
    opt_alpha = np.zeros([perm,n_cog,n_loops])
    #predictions made by the model
    if manual_folds:
        preds = np.zeros([perm,n_cog,n_test,n_loops])
        preds2 = np.zeros([perm,n_cog,n_test,n_loops])
        preds3 = np.zeros([perm,n_cog,n_test,n_loops])
        preds_resid = np.zeros([perm,n_cog,n_test,n_loops])
        cogtest = np.zeros([perm,n_cog,n_test])
    else:    
        preds = np.zeros([perm,n_cog,int(np.ceil(X.shape[0]*(1-train_size))),n_loops])
        preds2 = np.zeros([perm,n_cog,int(np.ceil(X.shape[0]*(1-train_size))),n_loops])
        preds3 = np.zeros([perm,n_cog,int(np.ceil(X.shape[0]*(1-train_size))),n_loops])
        preds_resid = np.zeros([perm,n_cog,int(np.ceil(X.shape[0]*(1-train_size))),n_loops])

        #true test values for cognition
        cogtest = np.zeros([perm,n_cog,int(np.ceil(X.shape[0]*(1-train_size)))])


        ##loop here ---> get the preds and then average over them

    
    #feature importance extracted from the model
    featimp = np.zeros([perm,n_feat,n_cog]) ##we dont use it actually could be good for future

    print(f'n_feat when creating {n_feat}')

    #set the param grid be to the hyperparamters you want to search through
    #paramGrid ={'regularizer': alphas}
    paramGrid ={'alpha': alphas}
    alphas_original = alphas
    n_iter_search_original = n_iter_search

    if z_score:
        scaler = StandardScaler()
        #X = X.T
        scaler.fit(X)
        X = scaler.transform(X)
        #X = X.T
    #iterate through permutations
    for p in range(perm):
        #print permutation # you're on
        print('Permutation %d' %(p+1))
        #split data into train and test sets
        if manual_folds:
            start = p * (n_train + n_test)
            x_train = X[(fold_list.values.flatten()).astype(int)[start : start + n_train]]
            x_test = X[(fold_list.values.flatten()).astype(int)[start + n_train:start + n_test + n_train]]
            
            cog_train = Y[(fold_list.values.flatten()).astype(int)[start : start + n_train]]
            cog_test = Y[(fold_list.values.flatten()).astype(int)[start + n_train:start + n_test + n_train]]
        else:    
            x_train, x_test, cog_train, cog_test = train_test_split(X, Y, test_size=1-train_size, 
                                                                shuffle=True, random_state=p)
        n_sub = n_train + n_test
        #print(cog_train)

        for n in range(n_loops):
            
            if Feature_selection:

                if bias_reduct:
                    
                    rng = np.random.default_rng(seed = n)
                    indecies_add_subs = rng.choice(n_train,size = int(n_sub/2- n_test), replace=False) ## sample rsubjects to add to the 'train' set to perform feature selection
                   

                    #print(x_train.shape)
                    x_test_FS = np.concatenate((x_train[indecies_add_subs,:],x_test)) ##arrays of 'test' and 'train' to do the correlation analysis
                    y_test_FS = np.concatenate((cog_train[indecies_add_subs,-1],cog_test[:,-1])) ## choose cog_train[0] for iq, [-1] for edu

                    print(f'x_test_FS_shape lenght {x_test_FS.shape}')
                    print(f'x_train_FS_shape lenght {np.delete(x_train,indecies_add_subs,axis = 0).shape}')
                    
                                
                    w_edu = r_regression(x_test_FS,y_test_FS)        
                    w_cog = r_regression(np.delete(x_train,indecies_add_subs,axis = 0),np.delete(cog_train[:,0],indecies_add_subs)) #take the array wich doesnt have the specific subjects

                
                else:
                    w_edu = r_regression(x_test,cog_test[:,-1])
                    w_cog = r_regression(x_train,cog_train[:,0]) ## choose cog_train[0] for iq, [-1] for edu
                
                w_prod = w_cog * w_edu
                w_prod[w_prod <= 0] = 0
                w_prod_norm = (w_prod - np.min(w_prod))/(np.max(w_prod)-np.min(w_prod))
               

                #w_prod_norm[w_prod_norm > 0].shape
                if prop:
                    n_feat_new = int(X.shape[1]/2)
                else:
                    n_feat_new = n_feat
                
                print (f'feature amount: {n_feat_new}')
                h_idx = np.argpartition(w_prod_norm,-n_feat_new)[-n_feat_new:]
                #print(f'h_idx = {h_idx}')
                #not_zero = np.nonzero(w_prod)[0]
                #print(f'not_zero {not_zero}')


                x_train1 = x_train[:,h_idx] ##Select the highest features
                x_test1 = x_test[:,h_idx]
                print('features selected')
                print(f'x_train_shape {x_train.shape}')

            
            #iterate through the cognitive metrics you want to predict
            for cog in range (n_cog):
                print(f'ncog: {cog}')

                #print cognitive metrics being predicted 
                print ("Cognition: %s" % cognition[cog])
                
                #set y values for train and test based on     
                y_train = cog_train[:,cog]
                y_mean = np.mean(y_train)
                y_train -= y_mean
                y_test = cog_test[:,cog]
                y_test -= y_mean
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
                        gridSearch.fit(x_train1, y_train)
                    
                    else:
                        #define regressor with random-search CV for inner loop
                        gridSearch = GridSearchCV(estimator=regr, param_grid=paramGrid, n_jobs=-1, 
                                            verbose=0, cv=inner_cv, scoring='r2')

                        #fit regressor
                        gridSearch.fit(x_train1, y_train)
                        
                        

                    #save parameters corresponding to the best score
                    best_params.append(list(gridSearch.best_params_.values()))
                    print(gridSearch.best_score_)

                    #call cross_val_score for outer loop
                    #nested_score = cross_val_score(gridSearch, X=x_train, y=y_train, cv=outer_cv, 
                                                #scoring='r2', verbose=1)

                    #record nested CV scores
                    #nested_scores.append(np.mean(nested_score))

                    #print how many cv loops are complete
                    print("%d/%d Complete" % (i+1,cv_loops))
                    
                #once all CV loops are complete, fit models based on optimised hyperparameters    
                print('Testing Models')


                #save optimised alpha values
                opt_alpha[p,cog,n] = np.mean(np.array(best_params))
                #print(np.array(best_params)[:])


            #1. model - to fit the intelligence using connectome features - usest the hyperparamter discoverd in cross validation
                model = Ridge(fit_intercept = fit_intercept, alpha = opt_alpha[p,cog,n], max_iter=1000000)
            #2. model - using only eduation
                model2 = LinearRegression(fit_intercept = True)
            #3/4. model - to track statistical relationshsips of predicitons and education    
                model3 = LinearRegression(fit_intercept = True)
                model4 = LinearRegression(fit_intercept = True)
            #shape the education vecotrs to suit the models    
                edu_train = np.array(cog_train[:,-1]).reshape(-1, 1)
                edu_test = np.array(cog_test[:,-1]).reshape(-1, 1)
            #fit the models
                model.fit(x_train1, y_train) #usingt the connecotme features
    
                model2.fit(edu_train,y_train) #usingt the education as the only feature
                

                #generate predictions from models
                preds[p,cog,:,n] = model.predict(x_test1).ravel()
                preds2[p,cog,:,n] = model2.predict(edu_test).ravel()#predict using educational info only, as one feture linear regression
                preds3[p,cog,:,n] = 0.5*preds2[p,cog,:,n] + 0.5*preds[p,cog,:,n] #calcuate the average with predicts from both iq and education -- this should make the alhorigm more robust aginst outliner

                #fit the analysis model using predicitons
                model3.fit(edu_test,preds[p,cog,:,n]) #fit the model on the predicitons - predict our fMRI iq prediction from the education level only (gives a binary vector) - answeres the question : how much our predicitons are determied by 
                model4.fit(edu_test,preds3[p,cog,:,n]) #fit the model on the predicitons - predict our fMRI iq prediction from the education level only (gives a binary vector) - answeres the question : how much our predicitons are determied by 

                pred_pred = model3.predict(edu_test) #predit the predictions using the linear model

            
                preds_resid[p,cog,:,n]  = preds[p,cog,:,n] - pred_pred  #find residuals of the predicitons wrt the education


        #print(f'are preds same? {preds[0,0,:,0] - preds[0,0,:,1]}')   
        preds4 = np.average(preds, axis = 3)
        preds5 = np.average(preds2, axis = 3)
        preds6 = np.average(preds3, axis = 3)
        preds_resid2 = np.average(preds_resid, axis = 3)
        print(f'preds4 shape {preds4.shape}')
        print(f'y_test shape {y_test.shape}')

        #compute explained variance 
        var[p,cog] = explained_variance_score(y_test, preds4[p,cog,:])

        
        #compute correlation between true and predicted - correlatuon is symmetric no need to think abt order
        corr_iq_fMRI_preds[p,cog] = np.corrcoef(y_test, preds4[p,cog,:])[1,0] ##main correlation - fMRI with IQ (with educaiton)
        corr_iq_edu_preds[p,cog] = np.corrcoef(y_test, preds5[p,cog,:])[1,0]  ##secondary correlation - only using educaiton level
        corr_iq_avg_preds[p,cog] = np.corrcoef(y_test, preds6[p,cog,:])[1,0] ##third correlation - simple average prediction
        corr_iq_resid_preds[p,cog] = np.corrcoef(y_test, preds_resid2[p,cog,:])[1,0] ##fourth correlation - residuals (information left after regressing out)

        corr_preds_edu[p,cog] = np.corrcoef(cog_test[:,-1], preds4[p,cog,:])[1,0]  ##correlation of our results with edu
        corr_preds_avg_edu[p,cog] = np.corrcoef(cog_test[:,-1], preds6[p,cog,:])[1,0] 

        #plt.scatter(cog_test[:,-1],preds[p,cog,:]) #plot if u want
        #plt.show()
        #print (var)


        #compute r^2s (coefficient of determination) 
        r2_iq_fMRI_preds[p,cog] = r2_score(y_test, preds4[p,cog,:]) ##first x,then real values
        r2_iq_edu_preds[p,cog] =r2_score(y_test, preds5[p,cog,:]) ##first x,then real values - using only the one feature (educaiton)
        r2_iq_avg_preds[p,cog] = r2_score(y_test, preds6[p,cog,:]) ##first y_true,then prediciton - average prediciton
        r2_iq_resid_preds[p,cog] = r2_score(y_test, preds_resid2[p,cog,:] ) ##first y_true, then prediciton - residuals (information left after regressing out)

        r2_preds_edu[p,cog] = model3.score(edu_test,preds4[p,cog,:]) ##first x,then real values - here we see how much the education explains the predictions we have usnig fMRI - to see the quality of the fit
        r2_preds_avg_edu[p,cog] = model4.score(edu_test,preds6[p,cog,:]) 

        #print the values
        print (f'opt alpha {np.average(opt_alpha[p,cog,:])}')
        print(f'r2 pred 1 fmri: {r2_iq_fMRI_preds[p,cog]}')
        print(f'r2 pred 2 edu only: {r2_iq_edu_preds[p,cog] }')
        print(f'r2 pred 3 average: {r2_iq_avg_preds[p,cog] }')
        print(f'r2 pred 3 residuals: {r2_iq_resid_preds[p,cog] }')
        print(f'r2 edu and pred fmri {r2_preds_edu[p,cog] }')

        print(f'corr pred 1 fmri: {corr_iq_fMRI_preds[p,cog]}')
        print(f'corr pred 2 edu only: {corr_iq_edu_preds[p,cog] }')
        print(f'corr pred 3 average: {corr_iq_avg_preds[p,cog] }')
        print(f'corr pred 3 residuals: {corr_iq_resid_preds[p,cog] }')
        print(f'corr edu and pred fmri: {corr_preds_edu[p,cog] }')

        #extract feature importance
        featimp[p,:,cog] = model.coef_
            #print(r2)
         
    return r2_iq_fMRI_preds, r2_iq_edu_preds, r2_iq_avg_preds, r2_iq_resid_preds, r2_preds_edu, corr_iq_fMRI_preds, corr_iq_edu_preds, corr_iq_avg_preds, corr_iq_resid_preds,corr_preds_edu, y_test.shape[0], cogtest, featimp, np.average(preds, axis = 3), np.average(preds2, axis = 3), np.average(preds3, axis = 3),var,np.average(opt_alpha,axis = 2)





def regressionSVR(X, Y, perm, cv_loops, k, train_size, n_cog, regr, params,n_feat,cognition,n_iter_search,random = True,Feature_selection = True):
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
        

        if Feature_selection: ##select the features

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

        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)


        
        
        #x_train = scaler.fit_transform(x_train)
        #x_test = scaler.fit_transform(x_test)

        
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
            print(f'r2: {r2}')
        
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


def importance_extractor(X, Y, perm, train_size, n_cog,n_feat,cognition):
    ##input X, Y, size of the train/test, number of cognitive features

    

    featimp = np.zeros([perm,n_feat,n_cog])

    #iterate through permutations
    for p in range(perm):
        for cog in range(n_cog):

            #print permutation # you're on
            print('Permutation %d' %(p+1))
            #split data into train and test sets
            x_train, x_test, cog_train, cog_test = train_test_split(X, Y, test_size=1-train_size, 
                                                                    shuffle=True, random_state=p)
            #print(cog_train)

        
            #print(cog_train[:,2])
            w_edu = r_regression(x_test,cog_test[:,-1])

            w_cog = r_regression(x_train,cog_train[:,0])
            w_prod = w_cog * w_edu
            w_prod[w_prod < 0] = 0
            w_prod_norm = (w_prod - np.min(w_prod))/(np.max(w_prod)-np.min(w_prod))
            #w_prod_norm[w_prod_norm > 0].shape
            #h_idx = np.argpartition(w_prod_norm,-n_feat)[-n_feat:]
            featimp[p,:,cog] = w_prod_norm
        #print(w_prod_norm)


        print(f'feautres selected, fold {p}')
        
    return(featimp)
