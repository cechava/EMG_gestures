
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat August 19 2021
@author: cechava
"""
from itertools import groupby
import numpy as np
import pandas as pd
import scipy
import scipy.signal

#to visualize 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#style params for figures
sns.set(font_scale = 2)
plt.style.use('seaborn-white')
plt.rc("axes", labelweight="bold")


#to load files
import os
import h5py

#ML packages
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score, f1_score,make_scorer, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold


from tensorflow import keras
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential, Model, load_model, Sequential, save_model
from tensorflow. keras.layers import Dense, Activation, Dropout, Input,  TimeDistributed, GRU, Masking, LSTM
from keras.callbacks import EarlyStopping

from tensorflow.keras.utils import to_categorical

from EMG_gestures.utils import *
from EMG_gestures.models import DANN

__all__ = ['within_subject_nn_performance','get_trained_model','evaluate_trained_log_reg','log_reg_xsubject_test',\
'log_reg_xsubject_joint_data_train_frac_subjects',\
'log_reg_xsubject_transform_module_train_all_subjects',\
'log_reg_xsubject_transform_module_train_frac_subjects','log_reg_xsubject_transform_module_train_all_subjects',\
'DANN_test',\
'within_subject_rnn_performance','evaluate_trained_rnn','get_trained_rnn_model','rnn_xsubject_test']

# ~~~~~~~~ NON-TEMPORAL NEURAL NET FUNCTIONS ~~~~~~~~
def within_subject_nn_performance(X, Y, series_labels, model_dict, exclude = [0,7],\
                                  score_list = ['f1'], epochs = 1000, batch_size = 5, es_patience = 50, verbose = 0,\
                                  prob_output = False, mv = None):

    #default values
    if 'fe_layers' not in model_dict.keys():
        model_dict['fe_layers'] = 1
    if 'fe_activation' not in model_dict.keys():
        model_dict['fe_activation'] = 'tanh'


    n_splits = np.unique(series_labels).size
    kf = KFold(n_splits=n_splits,shuffle = True)
    #initialize empty arrays
    n_scores = len(score_list)
    train_scores = np.empty((n_splits,n_scores))
    test_scores = np.empty((n_splits,n_scores))
    #training deets
    train_info_dict = {'val_loss': np.empty((n_splits,)),\
                       'train_loss': np.empty((n_splits,)),\
                       'epochs_trained':np.empty((n_splits,))}
    #retrieve some values from input
    nclass = (np.unique(Y).size)-np.sum(np.isin(np.unique(Y),exclude))
    nsamples, nfeat = X.shape
    prob_class = np.empty((nsamples,nclass))

    for split_count, (series_train, series_test) in enumerate(kf.split(np.unique(series_labels))):
        print('Split Count: %i'% (split_count+1))
        #get train and test idxs
        train_idxs = np.where(series_labels==series_train)[0]
        test_idxs = np.where(series_labels==series_test)[0]
        #get training and test data format
        X_train, Y_train, scaler = prepare_data_for_TF(X,Y, train_idxs, exclude, train = True)
        X_test, Y_test , scaler = prepare_data_for_TF(X,Y, test_idxs, exclude, train = False, scaler = scaler)

        n_features, n_outputs = X_train.shape[1], Y_train.shape[1]

        # patient early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=es_patience)

        #define model and compile
        model = get_vanilla_nn_model((n_features,),n_outputs, fe_layers = model_dict['fe_layers'],\
            fe_activation = model_dict['fe_activation'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #model.summary

        print('Training Model')
        # fit network
        history = model.fit(X_train, Y_train,validation_data = (X_test, Y_test),\
                            epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks = [es])

        #save training details to dict
        train_info_dict['train_loss'][split_count] = history.history['loss'][-1]
        train_info_dict['val_loss'][split_count] = history.history['val_loss'][-1]
        train_info_dict['epochs_trained'][split_count] = len(history.history['val_loss'])

        # # evaluate trained network
        print('Evaluate Model')

        if prob_output:
            #get prediction probabiliity on test set samples
            X_test, Y_test , scaler = prepare_data_for_TF(X,Y, test_idxs, [], train = False, scaler = scaler) 
            prob_class[test_idxs,:] = model.predict(X_test) 

        if mv:
            #get score for training data
            train_scores[split_count,:] = apply_mv_and_get_scores(X, Y, train_idxs, exclude,\
                                                                  scaler, model, mv, score_list)
            #get score for testing data
            test_scores[split_count,:] = apply_mv_and_get_scores(X, Y, test_idxs, exclude,\
                                                                  scaler, model, mv, score_list)

        else:

            #get score for training data
            train_scores[split_count,:] = get_scores(X_train, Y_train, model, score_list)

            #get score for testing data
            X_test, Y_test , scaler = prepare_data_for_TF(X,Y, test_idxs, exclude, train = False, scaler = scaler)
            test_scores[split_count,:] = get_scores(X_test, Y_test, model, score_list)
            
    return train_scores, test_scores, prob_class, train_info_dict




def get_trained_model(X, Y, train_idxs, exclude = [], model_dict = {},score_list = ['f1'], verbose = 0, epochs = 40, batch_size = 2,\
                      validation_split = 0, mv = False):


    if not model_dict:
        model_dict = {'n_dense_pre':0, 'activation':''}

    #exclude indicated labels
    in_samples = np.where(np.isin(Y,exclude, invert = True))[0]
    train_idxs_orig = train_idxs.copy()
    train_idxs = np.intersect1d(train_idxs,in_samples)

    #get training data cubes
    X_train_cube, Y_train_cube, scaler = prepare_data_for_TF(X,Y, train_idxs, exclude, train = True)

    #testfor equal number of samples
    assert X_train_cube.shape[0] == Y_train_cube.shape[0]
    n_features, n_outputs = X_train_cube.shape[1], Y_train_cube.shape[1]
    #setting timestep dimension to None 
    model = get_vanilla_nn_model((n_features,),n_outputs, n_dense_pre=model_dict['n_dense_pre'], activation=model_dict['activation'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary

    print('Training Model')
    # fit network
    history = model.fit(X_train_cube, Y_train_cube,validation_split = validation_split, \
                        epochs=epochs, batch_size=batch_size, verbose=verbose)
    # # evaluate trained network
    print('Evaluate Model on Trained Data')

    if mv:
        train_scores = apply_mv_and_get_scores(X, Y, train_idxs_orig, exclude,\
                                               scaler, model, mv, score_list)
    else:
        #get score for training data
        train_scores = get_scores(X_train_cube, Y_train_cube, model, score_list)
    return train_scores, model, scaler, history

def evaluate_trained_log_reg(X, Y, test_idxs, exclude, trained_model, score_list = ['f1'],scaler = None, mv = None):
    #exclude indicated labels
    test_idxs_orig = test_idxs.copy()
    in_samples = np.where(np.isin(Y,exclude, invert = True))[0]
    test_idxs = np.intersect1d(test_idxs,in_samples)

    print('Evaluate Model')
    if mv:
         test_scores = apply_mv_and_get_scores(X, Y, test_idxs_orig, exclude,\
                                               scaler, trained_model, mv, score_list)
    else:

        # get testing data cubes
        X_test_cube, Y_test_cube, scaler = prepare_data_for_TF(X,Y, test_idxs, exclude, train = False, scaler = scaler)
        #get score for testing data
        test_scores = get_scores(X_test_cube, Y_test_cube, trained_model, score_list)
    return test_scores

def log_reg_xsubject_test(data_folder, src_subject_id, nsubjects, nreps, lo_freq, hi_freq, win_size, step, exclude, score_list = ['f1'], \
                          verbose = 0, epochs = 40, batch_size = 2, mv = False, permute = False):
    
    
    subject_folder = os.path.join(data_folder,'%02d'%(src_subject_id))
    print('=======================')
    print(subject_folder)

    # Process data and get features 
    #get features across segments and corresponding info
    feature_matrix_src, target_labels_src, window_tstamps_src, \
    block_labels_src, series_labels_src = get_subject_data_for_classification(subject_folder, lo_freq, hi_freq, \
                                                                win_size, step)
    target_labels_src_orig = target_labels_src.copy()#keep originals before permuting
    train_idxs = np.arange(target_labels_src.size)
    np.random.seed(1)#for reproducibility

    results_df = []#initialize empty array for dataframes
    n_scores = len(score_list)
    train_scores_all = np.empty((nreps,n_scores))
    for rep in range(nreps):
        if permute:
            #permute while ignoring excluded blocks
            target_labels_src = permute_class_within_sub(target_labels_src_orig, block_labels_src, np.ones((target_labels_src.size,)), exclude)

        print('Subject %d|Rep %d'%(src_subject_id, rep+1))
        train_scores, trained_model, scaler, train_history = get_trained_model(feature_matrix_src, target_labels_src, train_idxs, exclude,\
                                                                               score_list = score_list,\
                                                                        verbose = verbose, epochs = epochs, batch_size = batch_size,\
                                                                        mv = mv)
        train_scores_all[rep,:] = train_scores
        # test on all other subjects
        # initialize empty lists
        test_scores_all = np.empty((0,0))
        targ_subject_list = []
        for targ_subject_id in range(1,nsubjects+1):
            if targ_subject_id != src_subject_id:

                subject_folder = os.path.join(data_folder,'%02d'%(targ_subject_id))
                print('Target Subject :%s'%(subject_folder))

                # Process data and get features 
                #get features across segments and corresponding info
                feature_matrix_targ, target_labels_targ, window_tstamps_targ, \
                block_labels_targ, series_labels_targ = get_subject_data_for_classification(subject_folder, lo_freq, hi_freq, \
                                                                        win_size, step)
                test_idxs = np.arange(target_labels_targ.size)

                test_scores = evaluate_trained_log_reg(feature_matrix_targ, target_labels_targ, test_idxs, exclude, trained_model,\
                                                       score_list, scaler, mv = mv)
                #append to list
                test_scores_all = np.vstack((test_scores_all, test_scores)) if test_scores_all.size else test_scores
                targ_subject_list.append(targ_subject_id)

        #put testing results in dataframe
        data_dict = {'Type':['Test' for x in range(nsubjects-1)],\
                     'Rep':[rep+1 for x in range(nsubjects-1)],\
                     'Test_Subject':targ_subject_list}
        for sidx in range(n_scores):
            data_dict['%s_score'%(score_list[sidx])] = test_scores_all[:,sidx]
        results_df.append(pd.DataFrame(data_dict))


    # #put training results in dataframe
    data_dict = {'Type':['Train' for x in range(nreps)],\
                 'Rep':[x+1 for x in range(nreps)],\
                 'Test_Subject':[src_subject_id for x in range(nreps)]}
    for sidx in range(n_scores):
        data_dict['%s_score'%(score_list[sidx])] = train_scores_all[:,sidx]
    results_df.append(pd.DataFrame(data_dict))

    
    results_df = pd.concat(results_df, axis = 0).reset_index(drop = True)

    return results_df

    



def log_reg_xsubject_joint_data_train_frac_subjects(feature_matrix, target_labels, sub_labels, block_labels, exclude,\
                                                    model_dict, score_list, n_splits = 4,\
                                                    verbose = 0, epochs = 40, batch_size = 2, validation_split = 0.1, mv = False, permute = False):
    """
    train and validate a logistic regression model using data from multiple subjects 
    train and validate model performance by splitting subjects into a train and test set
    """

    #subjects in list. there are the units over which we will do train/test split
    subs = np.unique(sub_labels)

    if permute:
        #permute while ignoring excluded blocks
        target_labels = permute_class_within_sub(target_labels, block_labels, sub_labels, exclude)


    #initialize object for k-fold cross-validation
    kf = KFold(n_splits=n_splits,shuffle = True)
    #initialize empty arrays

    n_scores = len(score_list)
    train_scores_all = np.empty((n_splits,n_scores))
    test_scores_all = np.empty((n_splits,n_scores))
    train_history = dict()
    train_history['loss'] = np.empty((0,0))
    train_history['val_loss'] = np.empty((0,0))

    for split_count, (subs_train_idxs, subs_test_idxs) in enumerate(kf.split(subs)):
        print('Split Count: %i'% (split_count+1))

        #get train and test indices
        train_subs = subs[subs_train_idxs]
        test_subs = subs[subs_test_idxs]
        train_idxs = np.where(np.isin(sub_labels,train_subs, invert = False))[0]
        test_idxs = np.where(np.isin(sub_labels,test_subs, invert = False))[0]

        #get trained model
        train_scores, trained_model, scaler, history = get_trained_model(feature_matrix, target_labels, train_idxs, exclude,\
                                                                         model_dict, score_list,\
                                                                         verbose = verbose, epochs = epochs, batch_size = batch_size,\
                                                                         validation_split = validation_split,\
                                                                         mv = mv)
        #Evaluating on held-out subjects
        test_scores = evaluate_trained_log_reg(feature_matrix, target_labels, test_idxs, exclude, trained_model, score_list,scaler, mv = mv)
    
        #put scores in array
        train_scores_all[split_count,:] = train_scores
        test_scores_all[split_count,:] = test_scores

        #append history
        train_history['loss'] = np.vstack((train_history['loss'],history.history['loss'])) if train_history['loss'].size else np.array(history.history['loss'])
        if validation_split>0:
            train_history['val_loss'] = np.vstack((train_history['val_loss'],history.history['val_loss'])) if train_history['val_loss'].size else np.array(history.history['val_loss']) 
    
    #put in data frame
    results_df = []
    data_dict = {'Fold':np.arange(n_splits)+1,\
                  'Type':['Train' for x in range(n_splits)]}
    for sidx in range(n_scores):
        data_dict['%s_score'%(score_list[sidx])] = train_scores_all[:,sidx]
    results_df.append(pd.DataFrame(data_dict))

    data_dict = {'Fold':np.arange(n_splits)+1,\
                 'Type':['Test' for x in range(n_splits)]}
    for sidx in range(n_scores):
        data_dict['%s_score'%(score_list[sidx])] = test_scores_all[:,sidx]
    results_df.append(pd.DataFrame(data_dict))

    results_df = pd.concat(results_df,axis = 0)

    return results_df, train_history

def log_reg_xsubject_joint_data_train_all_subjects(feature_matrix, target_labels, sub_labels, block_labels, exclude,\
                                                    model_dict, score_list,
                                                    verbose = 0, epochs = 40, batch_size = 2, validation_split = 0.1, mv = False, permute = False):
    """
    train and validate a logistic regression model using data from multiple subjects 
    train on all subjects
    """

    #subjects in list. there are the units over which we will do train/test split
    subs = np.unique(sub_labels)

    if permute:
        #permute while ignoring excluded blocks
        target_labels = permute_class_within_sub(target_labels, block_labels, sub_labels, exclude)



    n_scores = len(score_list)


    train_subs = subs
    train_idxs = np.where(np.isin(sub_labels,train_subs, invert = False))[0]
    #get trained model
    train_scores, trained_model, scaler, history = get_trained_model(feature_matrix, target_labels, train_idxs, exclude,\
                                                                        model_dict, score_list,\
                                                                        verbose = verbose, epochs = epochs, batch_size = batch_size,\
                                                                        validation_split = validation_split,\
                                                                        mv = mv)


    #put in data frame
    data_dict = {'Type':'Train'}
    for sidx in range(n_scores):
        data_dict['%s_score'%(score_list[sidx])] = train_scores[sidx]
    results_df = pd.DataFrame(data_dict, index = [0])


    return results_df, history, trained_model, scaler


def log_reg_xsubject_transform_module_train_frac_subjects(feature_matrix, target_labels, sub_labels, block_labels, series_labels, exclude,\
                                                         model_dict,score_list = ['f1'],n_train_splits = 4, n_val_splits = 2,\
                                                         verbose = 0, epochs = 40, batch_size = 2, mv = None,permute = False):
    """
    train and validate a logistic regression model with a transform module for domain adaptation. 
    train and validate model performance by splitting subjects into a train and test set
    """

    if permute:
        #permute while ignoring excluded blocks
        target_labels = permute_class_within_sub(target_labels, block_labels, sub_labels, exclude)

    results_df = []
    #subjects in list. there are the units over which we will do train/test split
    subs = np.unique(sub_labels)

    #exclude indicated labels
    in_samples = np.where(np.isin(target_labels,exclude, invert = True))[0]

    #initialize object for k-fold cross-validation
    kf = KFold(n_splits=n_train_splits,shuffle = True)


    for split_count, (subs_train_idxs, subs_test_idxs) in enumerate(kf.split(subs)):
        print('-------Split Count: %i-------'% (split_count+1))
        #get train and test indices
        train_subs = subs[subs_train_idxs]
        test_subs = subs[subs_test_idxs]
        train_idxs = np.where(np.isin(sub_labels,train_subs, invert = False))[0]
        test_idxs = np.where(np.isin(sub_labels,test_subs, invert = False))[0]

        #get train and test indices
        train_subs = subs[subs_train_idxs]
        test_subs = subs[subs_test_idxs]
        train_idxs = np.where(np.isin(sub_labels,train_subs, invert = False))[0]
        test_idxs = np.where(np.isin(sub_labels,test_subs, invert = False))[0]
        
        #get training data cubes
        X_train_cube, Y_train_cube, scaler = prepare_data_for_TF(feature_matrix, target_labels, train_idxs, exclude, train = True)
        sub_labels_train = sub_labels[np.intersect1d(train_idxs,in_samples)]


        #testfor equal number of samples
        assert X_train_cube.shape[0] == Y_train_cube.shape[0]
        n_features, n_outputs = X_train_cube.shape[1], Y_train_cube.shape[1]

        # get testing data cubes
        X_test_cube, Y_test_cube, scaler = prepare_data_for_TF(feature_matrix, target_labels, test_idxs, exclude, train = False, scaler = scaler)
        
        sub_labels_test = sub_labels[np.intersect1d(test_idxs,in_samples)]
        series_labels_test = series_labels[np.intersect1d(test_idxs,in_samples)]

        # permute order in which subjects' data is used for training
        train_subs_perm = np.random.permutation(train_subs)
        #initialize empty list
        n_scores = len(score_list)
        train_scores_all = np.empty((train_subs.size,n_scores))

        # --- Training Stage ---
        # Define model architecture

        #setting timestep dimension to None 
        model = get_vanilla_nn_model((n_features,),n_outputs, n_dense_pre=model_dict['n_dense_pre'], activation=model_dict['activation'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #model.summary
        # # Get transform module template
        transform_module_template = get_transform_module(model, Input(shape = (None,n_features)),1)


        # iterate thorugh subjects' data
        for sub_idx, train_sub in enumerate(train_subs_perm):
            print('Training: Subject %02d out of %02d'%(sub_idx+1, train_subs.size))
            # get subject-specific samples
            train_sub_idxs = np.where(sub_labels_train == train_sub)[0]
            X_cube_sub = X_train_cube[train_sub_idxs,:]
            Y_cube_sub = Y_train_cube[train_sub_idxs,]

            # initialize weights of the transform module
            model = tm_template_weights_to_model(transform_module_template, model)

            print('Training Model')
            # fit network
            history = model.fit(X_cube_sub, Y_cube_sub, epochs=epochs, batch_size=batch_size, verbose=verbose)

            #copy weights to a transfer module template, save if wanted
            trained_transfer_module = model_weights_to_tm_template(transform_module_template, model)
            # # evaluate trained network
            print('Evaluate Model on Trained Data')

            if mv:
                #get f1 score after applying majority voting scheme to model predictions
                train_scores_all[sub_idx,:] = apply_mv_and_get_scores(feature_matrix, target_labels,\
                                                                np.intersect1d(np.where(sub_labels==train_sub)[0],train_idxs), exclude,\
                                                                scaler, model, mv, score_list)
            else:
                #get score for training data
                train_scores_all[sub_idx,:] = get_scores(X_cube_sub, Y_cube_sub, model, score_list)

        # #put results in dataframe
        data_dict = {'Subject':train_subs_perm+1,\
                     'Fold':[split_count+1 for x in range(train_subs_perm.size)],\
                     'Type':['Train' for x in range(train_subs_perm.size)]}

        for s_idx in range(n_scores):
            data_dict['%s_score'%(score_list[s_idx])] = train_scores_all[:,s_idx]
            
        results_df.append(pd.DataFrame(data_dict))

        # --- Validation Stage ---
        #freeze top layers
        for layer in model.layers[-1:]:
                layer.trainable = False

        # iterate through test subjects
        for sub_idx, test_sub in enumerate(test_subs):
            print('Validation: Subject %02d out of %02d'%(sub_idx+1, test_subs.size))

            #get relevant subject samples
            test_sub_idxs = np.where(sub_labels_test == test_sub)[0]
            test_series = series_labels_test[test_sub_idxs]
            X_cube_sub = X_test_cube[test_sub_idxs,:]
            Y_cube_sub = Y_test_cube[test_sub_idxs,]
            test_sub_labels = np.argmax(Y_cube_sub,1)

            #stratify split to retain ratio of class labels
            kf = KFold(n_splits=n_val_splits,shuffle = True)

            val_train_scores = np.empty((n_val_splits,n_scores))
            val_test_scores = np.empty((n_val_splits,n_scores))

            #systematically use one fold of the data as a held-out test set
            for split_count_val, (series_train, series_test) in enumerate(kf.split(np.unique(test_series))):
                
                #split data cubes into train and test subsets
                series_train_idxs = np.where(test_series==series_train)[0]
                X_train_cube_sub = X_cube_sub[series_train_idxs,:]
                Y_train_cube_sub = Y_cube_sub[series_train_idxs,:]
 

                series_test_idxs = np.where(test_series==series_test)[0]
                X_test_cube_sub = X_cube_sub[series_test_idxs,:]
                Y_test_cube_sub = Y_cube_sub[series_test_idxs,:]

                #initialize transform module
                model = tm_template_weights_to_model(transform_module_template, model)
                #train
                model.fit(X_train_cube_sub, Y_train_cube_sub, epochs=epochs, batch_size=batch_size, verbose=verbose)

                #copy weights to a transfer module template, save if wanted
                trained_transfer_module = model_weights_to_tm_template(transform_module_template, model)

                #evaluate on training and testing
                if mv:
                    #get f1 score after applying majority voting scheme to model predictions
                    val_train_scores[split_count_val,:]  = apply_mv_and_get_scores(feature_matrix[test_idxs], target_labels[test_idxs],\
                                                                    np.where(series_labels[test_idxs]==series_train)[0], exclude,\
                                                                    model = model, n_votes = mv, scaler = scaler, score_list = score_list)
                    val_test_scores[split_count_val,:]  = apply_mv_and_get_scores(feature_matrix[test_idxs], target_labels[test_idxs],\
                                                                    np.where(series_labels[test_idxs]==series_test)[0], exclude,\
                                                                    model = model, n_votes = mv, scaler = scaler, score_list = score_list)
                else:
                    val_train_scores[split_count_val,:] = get_scores(X_train_cube_sub, Y_train_cube_sub, model, score_list)
                    val_test_scores[split_count_val,:] = get_scores(X_test_cube_sub, Y_test_cube_sub, model, score_list)


            #put results in dataframe
            data_dict = {'Subject':[test_sub+1 for x in range(n_val_splits)],\
                         'Fold':[split_count+1 for x in range(n_val_splits)],\
                         'Type':['Val_Train' for x in range(n_val_splits)]}
            for sidx in range(n_scores):
                data_dict['%s_score'%(score_list[sidx])] = val_train_scores[:,sidx]
            results_df.append(pd.DataFrame(data_dict))

            data_dict = {'Subject':[test_sub+1 for x in range(n_val_splits)],\
                         'Fold':[split_count+1 for x in range(n_val_splits)],\
                         'Type':['Val_Test' for x in range(n_val_splits)]}
            for sidx in range(n_scores):
                data_dict['%s_score'%(score_list[sidx])] = val_test_scores[:,sidx]
            results_df.append(pd.DataFrame(data_dict))

            
    results_df = pd.concat(results_df,axis = 0)

    return results_df

def log_reg_xsubject_transform_module_train_all_subjects(feature_matrix, target_labels, sub_labels, block_labels,\
                                                         train_idxs, test_idxs, exclude, model_dict, score_list,\
                                                         figure_folder = '', model_folder = '', 
                                                         verbose = 0, epochs = 40, batch_size = 2, mv = None, permute = False):
    """
    train and validate a logistic regression model with a transform module for domain adaptation. 
    validate model performance by holding out indicated samples for each subject

    """

    results_df = []

    subs = np.unique(sub_labels)

    if permute:
        #permute while ignoring excluded blocks
        target_labels = permute_class_within_sub(target_labels, block_labels, sub_labels, exclude)


    #get training data cubes
    X_train_cube, Y_train_cube, scaler = prepare_data_for_TF(feature_matrix, target_labels, train_idxs, exclude, train = True)
    in_samples = np.where(np.isin(target_labels,exclude, invert = True))[0]
    sub_labels_train = sub_labels[np.intersect1d(train_idxs,in_samples)]

    #test for equal number of samples
    assert X_train_cube.shape[0] == Y_train_cube.shape[0]
    n_features, n_outputs = X_train_cube.shape[1], Y_train_cube.shape[1]

    if test_idxs.size>0:
        #get testing data cubes
        X_test_cube, Y_test_cube, scaler = prepare_data_for_TF(feature_matrix, target_labels, test_idxs, exclude, train = False, scaler = scaler)
        in_samples = np.where(np.isin(target_labels,exclude, invert = True))[0]
        sub_labels_test = sub_labels[np.intersect1d(test_idxs,in_samples)]

    # permute order in which subjects' data is used for training
    subs_perm = np.random.permutation(subs)

    #initialize empty list
    n_scores = len(score_list)
    train_scores = np.empty((subs.size,n_scores))
    test_scores = np.empty((subs.size,n_scores))

    # --- Training Stage ---
    # Define model architecture

    #setting timestep dimension to None 
    model = get_vanilla_nn_model((n_features,),n_outputs, n_dense_pre=model_dict['n_dense_pre'], activation=model_dict['activation'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary
    # # Get transform module template
    transform_module_template = get_transform_module(model, Input(shape = (None,n_features)),1)


    # iterate thorugh subjects' data
    for sub_idx, train_sub in enumerate(subs_perm):
        print('Training: Subject %02d out of %02d'%(sub_idx+1, subs_perm.size))
        # get subject-specific samples
        train_sub_idxs = np.where(sub_labels_train == train_sub)[0]

        X_cube_sub = X_train_cube[train_sub_idxs,:]
        Y_cube_sub = Y_train_cube[train_sub_idxs,]
        # initialize weights of the transform module
        model = tm_template_weights_to_model(transform_module_template, model)

        print('Training Model')
        # fit network
        history = model.fit(X_cube_sub, Y_cube_sub, epochs=epochs, batch_size=batch_size, verbose=verbose)
        if figure_folder:
            #plot training loss
            fig_title = 'Subject %02d'%(train_sub)
            fig_fn = os.path.join(figure_folder,'log_reg_model_subject_%02d_all_train_data_permuted_%s_loss.png'%(train_sub,str(permute)))
            plot_train_loss(history, fig_title, fig_fn)

        #copy weights to a transfer module template, save if wanted
        trained_transfer_module = model_weights_to_tm_template(transform_module_template, model)
        if model_folder:
            #save trained transfer module to file
            model_fn = os.path.join(model_folder, 'transform_module_subject_%02d_all_train_data_permuted_%s.h5'%(train_sub, str(permute)))
            keras.models.save_model(trained_transfer_module, model_fn, save_format= 'h5')
        # # evaluate trained network
        print('Evaluate Model on Trained Data')

        if mv:
            # get f1 score after applying majority voting scheme to model predictions
            train_scores[sub_idx,:]  = apply_mv_and_get_scores(feature_matrix, target_labels,\
                                                                np.intersect1d(np.where(sub_labels==train_sub)[0],train_idxs), exclude,\
                                                                scaler, model, mv, score_list)
            if test_idxs.size>0:
                test_scores[sub_idx,:] = apply_mv_and_get_scores(feature_matrix, target_labels, \
                                                                    np.intersect1d(np.where(sub_labels==train_sub)[0],test_idxs), exclude,\
                                                                    scaler, model, mv, score_list)
        else:
            #get score for training data
            train_scores[sub_idx,:]  = get_scores(X_cube_sub, Y_cube_sub, model, score_list)
            if test_idxs.size>0:
                #get score for test data
                test_sub_idxs = np.where(sub_labels_test == train_sub)[0]
                test_scores[sub_idx,:]  = get_scores(X_test_cube[test_sub_idxs,:], Y_test_cube[test_sub_idxs,:], model, score_list)

    #put results in dataframe
    data_dict = {'Subject':subs_perm,\
                 'Type':['Train' for x in range(subs_perm.size)]}
    for sidx in range(n_scores):
        data_dict['%s_score'%(score_list[sidx])] = train_scores[:,sidx]
    results_df.append(pd.DataFrame(data_dict))

    if test_idxs.size>0:
            data_dict = {'Subject':subs_perm,\
                         'Type':['Train_val' for x in range(subs_perm.size)]}
            for sidx in range(n_scores):
                data_dict['%s_score'%(score_list[sidx])] = test_scores[:,sidx]
            results_df.append(pd.DataFrame(data_dict))
        
    results_df = pd.concat(results_df,axis = 0)

    if model_folder:
        #save complete model to file
        model_fn = os.path.join(model_folder, 'trained_model_all_train_data_permuted_%s.h5'%(str(permute)))
        keras.models.save_model(model, model_fn, save_format= 'h5')
    return results_df, scaler


def DANN_test(source_X, source_Y, target_X, target_Y, score_list, n_splits, epochs, batch_size,\
              permute = False):

    if permute:
        #scramble class labels
        source_Y = np.random.permutation(source_Y)
        target_Y = np.random.permutation(target_Y)
        
    #put together labels
    all_X =  np.vstack((source_X,target_X))
    all_Y = np.hstack((source_Y, target_Y))
    
    domain_Y = np.hstack((np.ones(source_X.shape[0],)*0,np.ones(source_X.shape[0],)*1))#source_label


    skf = StratifiedKFold(n_splits=n_splits)

    #initialize empty arrays
    n_scores = len(score_list)

    source_train_scores = np.empty((n_splits,n_scores))
    source_test_scores = np.empty((n_splits,n_scores))

    naive_target_train_scores = np.empty((n_splits,n_scores))
    naive_target_test_scores = np.empty((n_splits,n_scores))

    adapt_source_test_scores  = np.empty((n_splits,n_scores))
    adapt_target_train_scores  = np.empty((n_splits,n_scores))
    adapt_target_test_scores  = np.empty((n_splits,n_scores))

    revealed_source_train_scores = np.empty((n_splits,n_scores))
    revealed_source_test_scores = np.empty((n_splits,n_scores))
    revealed_target_train_scores = np.empty((n_splits,n_scores))
    revealed_target_test_scores = np.empty((n_splits,n_scores))

    #re-code so that stratified split preserves both label and comain fractions
    dummy_Y = all_Y+(domain_Y*(np.max(all_Y)+1))

    for split_count,(train_idxs, test_idxs) in enumerate(skf.split(all_X, dummy_Y)):
        print('Split Count: %d'%(split_count+1))

        #get indices
        source_train_idxs =  np.intersect1d(train_idxs,np.where(dummy_Y<2)[0])
        source_test_idxs = np.intersect1d(test_idxs,np.where(dummy_Y<2)[0])
        target_train_idxs = np.intersect1d(train_idxs,np.where(dummy_Y>=2)[0])
        target_test_idxs = np.intersect1d(test_idxs,np.where(dummy_Y>=2)[0])

        #prepare data for neural net
        source_train_X, source_train_Y, scaler = prepare_data_for_TF(all_X, all_Y, source_train_idxs,\
                                                                          [], train = True)
        source_test_X, source_test_Y, scaler = prepare_data_for_TF(all_X, all_Y, source_test_idxs,\
                                                                        [], scaler = scaler)
        target_train_X, target_train_Y, scaler = prepare_data_for_TF(all_X, all_Y, target_train_idxs,\
                                                                          [], scaler = scaler)
        target_test_X, target_test_Y, scaler = prepare_data_for_TF(all_X, all_Y, target_test_idxs,\
                                                                        [], scaler = scaler)

        n_features, n_outputs = source_train_X.shape[1], source_train_Y.shape[1]

        #define and compile model
        input_shape = (n_features,)
        dann_model = DANN(input_shape, n_outputs, fe_layers = 1, dp_layers = 1, activation = 'tanh')
        dann_model.compile(loss='categorical_crossentropy')

        #train on source labels
        print('Training on Source Labels')
        dann_model.train_label_pred(source_train_X, source_train_Y, epochs=epochs, batch_size=batch_size, verbose=0)

        #score on source data
        source_train_scores[split_count,:] = get_scores(source_train_X, source_train_Y, dann_model.predict_label, score_list)
        source_test_scores[split_count,:] = get_scores(source_test_X, source_test_Y, dann_model.predict_label, score_list)
        #score on target data (naive transfer test)
        naive_target_train_scores[split_count,:] = get_scores(target_train_X, target_train_Y, dann_model.predict_label, score_list)
        naive_target_test_scores[split_count,:] = get_scores(target_test_X, target_test_Y, dann_model.predict_label, score_list)

        #train same model with domain labels of target
        print('Adapting to target Domain')
        dann_model.train_domain_adapt(source_train_X, source_train_Y, target_train_X,\
                                      epochs = epochs*3, batch_size = batch_size, verbose = 0)

        #score on source data
        adapt_source_test_scores[split_count,:] = get_scores(source_test_X, source_test_Y, dann_model.predict_label,\
                                                             score_list)
        #score on target data
        adapt_target_train_scores[split_count,:]  = get_scores(target_train_X, target_train_Y, dann_model.predict_label,\
                                                             score_list)
        adapt_target_test_scores[split_count,:]  = get_scores(target_test_X, target_test_Y, dann_model.predict_label,\
                                                             score_list)

        # for comparison, use all available labels 
        # define and compile model
        input_shape = (n_features,)
        dann_model = DANN(input_shape, n_outputs, fe_layers = 1, dp_layers = 1, activation = 'tanh')
        dann_model.compile(loss='categorical_crossentropy')
        print('Training with all labels revealed')
        dann_model.train_domain_and_labels(source_train_X, source_train_X, target_train_X, target_train_Y,\
                                           epochs=epochs*2, batch_size=batch_size, verbose=0)

        revealed_source_train_scores[split_count,:] = get_scores(source_train_X, source_train_Y, dann_model.predict_label,\
                                                             score_list)
        revealed_source_test_scores[split_count,:] = get_scores(source_test_X, source_test_Y, dann_model.predict_label,\
                                                             score_list)
        revealed_target_train_scores[split_count,:] = get_scores(target_train_X, target_train_Y, dann_model.predict_label,\
                                                             score_list)
        revealed_target_test_scores[split_count,:] = get_scores(target_test_X, target_test_Y, dann_model.predict_label,\
                                                             score_list)

    #score arrays in dataframes
    results_df = []

    results_df = results_to_df(results_df, n_splits, score_list,\
                               source_train_scores, 'Source_Train')
    results_df = results_to_df(results_df, n_splits, score_list,\
                               source_test_scores, 'Source_Test')

    results_df = results_to_df(results_df, n_splits, score_list,\
                               naive_target_train_scores, 'Naive_Target_Train')
    results_df = results_to_df(results_df, n_splits, score_list,\
                               naive_target_test_scores, 'Naive_Target_Test')

    results_df = results_to_df(results_df, n_splits, score_list,\
                               adapt_source_test_scores, 'Adapt_Source_Test')
    results_df = results_to_df(results_df, n_splits, score_list,\
                               adapt_target_train_scores, 'Adapt_Target_Train')
    results_df = results_to_df(results_df, n_splits, score_list,\
                               adapt_target_test_scores, 'Adapt_Target_Test')

    results_df = results_to_df(results_df, n_splits, score_list,\
                               revealed_source_train_scores, 'Revealed_Source_Train')
    results_df = results_to_df(results_df, n_splits, score_list,\
                               revealed_source_test_scores, 'Revealed_Source_Test')
    results_df = results_to_df(results_df, n_splits, score_list,\
                               revealed_target_train_scores, 'Revealed_Target_Train')
    results_df = results_to_df(results_df, n_splits, score_list,\
                               revealed_source_test_scores, 'Revealed_Target_Test')
    results_df = pd.concat(results_df, axis = 0)

    return results_df

def within_subject_rnn_performance(X, Y, block_labels, series_labels, model_dict, exclude = [0,7],\
                                  score_list = ['f1'], n_shuffled_sets = 10,epochs = 1000,\
                                  batch_size = 5, es_patience = 50, verbose = 0,  mv = None):

    #initialize object for k-fold cross-validation
    n_splits = np.unique(series_labels).size
    kf = KFold(n_splits=n_splits,shuffle = True)
    #initialize empty arrays
    n_scores = len(score_list)
    train_scores = np.empty((n_splits,n_scores))
    test_scores = np.empty((n_splits,n_scores))
    #training deets
    train_info_dict = {'val_loss': np.empty((n_splits,)),\
                       'train_loss': np.empty((n_splits,)),\
                       'epochs_trained':np.empty((n_splits,))}

    for split_count, (series_train, series_test) in enumerate(kf.split(np.unique(series_labels))):
        print('Split Count: %i'% (split_count+1))
        #get train and test idxs
        train_idxs = np.where(series_labels==series_train)[0]
        test_idxs = np.where(series_labels==series_test)[0]
        #get training data cube
        X_train_cube, Y_train_cube, scaler = prepare_data_for_RNN(X, Y, train_idxs, exclude, train = True,\
                                                    block_labels = block_labels, nsets = n_shuffled_sets)
        n_features, n_outputs = X_train_cube.shape[2], Y_train_cube.shape[2]
        #get testing data cube
        X_test_cube, Y_test_cube, scaler = prepare_data_for_RNN(X, Y, test_idxs, exclude, train = False, scaler = scaler)


        # patient early stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=es_patience)

        # #setting timestep dimension to None 
        model = get_rnn_model((None,n_features,),n_outputs, fe_layers = model_dict['fe_layers'],\
                              fe_activation = model_dict['fe_activation'],\
                              fc_layers = model_dict['fe_layers'], fc_activation = model_dict['fc_activation'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #model.summary()


        print('Training Model')
        # fit network
        history = model.fit(X_train_cube, Y_train_cube, validation_data = (X_test_cube, Y_test_cube),\
                            epochs=epochs, batch_size=batch_size, verbose=0 , callbacks = [es])


        #save training details to dict
        train_info_dict['train_loss'][split_count] = history.history['loss'][-1]
        train_info_dict['val_loss'][split_count] = history.history['val_loss'][-1]
        train_info_dict['epochs_trained'][split_count] = len(history.history['val_loss'])

        # # evaluate trained network
        print('Evaluate Model')

        if mv:
            #get score for training data
            train_scores[split_count,:] = apply_mv_and_get_scores(X, Y, train_idxs, exclude,\
                                                                    scaler, model, mv, score_list, rnn = True)
            #get score for testing data
            test_scores[split_count,:] = apply_mv_and_get_scores(X, Y, test_idxs, exclude,\
                                                                    scaler, model, mv, score_list, rnn = True)

        else:

            #get score for training data
            train_scores[split_count,:] = get_scores(X_train_cube, Y_train_cube, model, score_list, rnn = True)

            #get score for testing data
            test_scores[split_count,:] = get_scores(X_test_cube, Y_test_cube, model, score_list, rnn = True)
            
    return train_scores, test_scores, prob_class, train_info_dict

def evaluate_trained_rnn(X, Y, test_idxs, exclude, trained_model, score_list = ['f1'],scaler = None, mv = None):
    #exclude indicated labels
    test_idxs_orig = test_idxs.copy()
    in_samples = np.where(np.isin(Y,exclude, invert = True))[0]
    test_idxs = np.intersect1d(test_idxs,in_samples)

    print('Evaluate Model')
    if mv:
         test_scores = apply_mv_and_get_scores(X, Y, test_idxs_orig, exclude,\
                                               scaler, trained_model, mv, score_list, rnn = True)
    else:

        # get testing data cubes
        X_test_cube, Y_test_cube, scaler = prepare_data_for_RNN(X,Y, test_idxs, exclude, train = False, scaler = scaler)
        #get score for testing data
        test_scores = get_scores(X_test_cube, Y_test_cube, trained_model, score_list, rnn = True)
    return test_scores

def get_trained_rnn_model(X, Y, train_idxs, block_labels, nsets = 10, exclude = [], model_dict = {},score_list = ['f1'], verbose = 0, epochs = 40, batch_size = 2,\
                      validation_split = 0, mv = False):


    if not model_dict:
        model_dict = {'n_dense_pre':0,'n_grus':24, 'activation':'','n_dense_post':0}

    #exclude indicated labels
    in_samples = np.where(np.isin(Y,exclude, invert = True))[0]
    train_idxs_orig = train_idxs.copy()
    train_idxs = np.intersect1d(train_idxs,in_samples)

    #get training data cubes
    X_train_cube, Y_train_cube, scaler = prepare_data_for_RNN(X,Y, train_idxs, exclude, train = True,\
                                                              block_labels = block_labels, nsets = nsets)

    #testfor equal number of samples
    assert X_train_cube.shape[0] == Y_train_cube.shape[0]
    #testfor equal number of timepoints
    assert X_train_cube.shape[1] == Y_train_cube.shape[1]

    n_features, n_outputs = X_train_cube.shape[2], Y_train_cube.shape[2]
    #setting timestep dimension to None 
    model = get_rnn_model((None,n_features,),n_outputs,n_dense_pre=model_dict['n_dense_pre'], n_dense_post=model_dict['n_dense_post'],\
                          n_grus = model_dict['n_grus'], activation=model_dict['activation'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary

    print('Training Model')
    # fit network
    history = model.fit(X_train_cube, Y_train_cube,validation_split = validation_split, \
                        epochs=epochs, batch_size=batch_size, verbose=verbose)
    # # evaluate trained network
    print('Evaluate Model on Trained Data')

    if mv:
        train_scores = apply_mv_and_get_scores(X, Y, train_idxs_orig, exclude,\
                                               scaler, model, mv, score_list, rnn = True)
    else:
        #get score for training data
        train_scores = get_scores(X_train_cube, Y_train_cube, model, score_list, rnn = True)
    return train_scores, model, scaler, history

def rnn_xsubject_test(data_folder, src_subject_id, nsubjects, nreps, lo_freq, hi_freq, win_size, step, exclude, score_list = ['f1'], \
                          nsets_training = 10,verbose = 0, epochs = 40, batch_size = 2, mv = False, permute = False):
    """
    Test naive cross-subject generalization of an RNN model
    Train an RNN model on data from one subject; test on all other subjects
    """


    subject_folder = os.path.join(data_folder,'%02d'%(src_subject_id))
    print('=======================')
    print(subject_folder)

    # Process data and get features 
    #get features across segments and corresponding info
    feature_matrix_src, target_labels_src, window_tstamps_src, \
    block_labels_src, series_labels_src = get_subject_data_for_classification(subject_folder, lo_freq, hi_freq, \
                                                                win_size, step)
    target_labels_src_orig = target_labels_src.copy()#keep originals before permuting
    train_idxs = np.arange(target_labels_src.size)

    np.random.seed(1)#for reproducibility

    results_df = []#initialize empty array for dataframes
    n_scores = len(score_list)
    train_scores_all = np.empty((nreps,n_scores))

    for rep in range(nreps):
        if permute:
            #permute while ignoring excluded blocks
            target_labels_src = permute_class_within_sub(target_labels_src_orig, block_labels_src, np.ones((target_labels_src.size,)), exclude)

        print('Subject %d|Rep %d'%(src_subject_id, rep+1))
        train_scores, trained_model, scaler, train_history = get_trained_rnn_model(feature_matrix_src, target_labels_src, train_idxs, block_labels_src,
                                                                                        nsets_training,exclude,\
                                                                                score_list = score_list,\
                                                                        verbose = verbose, epochs = epochs, batch_size = batch_size,\
                                                                        mv = mv)
        train_scores_all[rep,:] = train_scores

        # test on all other subjects
        # initialize empty lists
        test_scores_all = np.empty((0,0))
        targ_subject_list = []
        for targ_subject_id in range(1,nsubjects+1):
            if targ_subject_id != src_subject_id:

                subject_folder = os.path.join(data_folder,'%02d'%(targ_subject_id))
                print('Target Subject :%s'%(subject_folder))

                # Process data and get features 
                #get features across segments and corresponding info
                feature_matrix_targ, target_labels_targ, window_tstamps_targ, \
                block_labels_targ, series_labels_targ = get_subject_data_for_classification(subject_folder, lo_freq, hi_freq, \
                                                                        win_size, step)
                test_idxs = np.arange(target_labels_targ.size)

                test_scores = evaluate_trained_rnn(feature_matrix_targ, target_labels_targ, test_idxs, exclude, trained_model,\
                                                        score_list, scaler, mv = mv)
                #append to list
                test_scores_all = np.vstack((test_scores_all, test_scores)) if test_scores_all.size else test_scores
                targ_subject_list.append(targ_subject_id)

            #put testing results in dataframe
        data_dict = {'Type':['Test' for x in range(nsubjects-1)],\
                        'Rep':[rep+1 for x in range(nsubjects-1)],\
                        'Test_Subject':targ_subject_list}
        for sidx in range(n_scores):
            data_dict['%s_score'%(score_list[sidx])] = test_scores_all[:,sidx]
        results_df.append(pd.DataFrame(data_dict))

    # #put training results in dataframe
    data_dict = {'Type':['Train' for x in range(nreps)],\
                    'Rep':[x+1 for x in range(nreps)],\
                    'Test_Subject':[src_subject_id for x in range(nreps)]}
    for sidx in range(n_scores):
        data_dict['%s_score'%(score_list[sidx])] = train_scores_all[:,sidx]
    results_df.append(pd.DataFrame(data_dict))


    results_df = pd.concat(results_df, axis = 0).reset_index(drop = True)

    return results_df

def rnn_xsubject_joint_data_train_frac_subjects(feature_matrix, target_labels, sub_labels, block_labels, exclude,\
                                                    model_dict, score_list, n_splits = 4, nsets_training = 10,\
                                                    verbose = 0, epochs = 40, batch_size = 2, validation_split = 0.1, mv = False, permute = False):
    """
    train and validate a RNN model using data from multiple subjects 
    train and validate model performance by splitting subjects into a train and test set
    """

    #subjects in list. there are the units over which we will do train/test split
    subs = np.unique(sub_labels)

    if permute:
        #permute while ignoring excluded blocks
        target_labels = permute_class_within_sub(target_labels, block_labels, sub_labels, exclude)


    #initialize object for k-fold cross-validation
    kf = KFold(n_splits=n_splits,shuffle = True)
    #initialize empty arrays

    n_scores = len(score_list)
    train_scores_all = np.empty((n_splits,n_scores))
    test_scores_all = np.empty((n_splits,n_scores))
    train_history = dict()
    train_history['loss'] = np.empty((0,0))
    train_history['val_loss'] = np.empty((0,0))

    for split_count, (subs_train_idxs, subs_test_idxs) in enumerate(kf.split(subs)):
        print('Split Count: %i'% (split_count+1))

        #get train and test indices
        train_subs = subs[subs_train_idxs]
        test_subs = subs[subs_test_idxs]
        train_idxs = np.where(np.isin(sub_labels,train_subs, invert = False))[0]
        test_idxs = np.where(np.isin(sub_labels,test_subs, invert = False))[0]

        #get trained model
        train_scores, trained_model, scaler, history = get_trained_rnn_model(feature_matrix, target_labels, train_idxs, block_labels, nsets_training,\
                                                                            exclude, model_dict, score_list,\
                                                                            verbose = verbose, epochs = epochs, batch_size = batch_size,\
                                                                            validation_split = validation_split,\
                                                                            mv = mv)
        #Evaluating on held-out subjects
        test_scores = evaluate_trained_rnn(feature_matrix, target_labels, test_idxs, exclude, trained_model, score_list,scaler, mv = mv)

        #put scores in array
        train_scores_all[split_count,:] = train_scores
        test_scores_all[split_count,:] = test_scores

        #append history
        train_history['loss'] = np.vstack((train_history['loss'],history.history['loss'])) if train_history['loss'].size else np.array(history.history['loss'])
        if validation_split>0:
            train_history['val_loss'] = np.vstack((train_history['val_loss'],history.history['val_loss'])) if train_history['val_loss'].size else np.array(history.history['val_loss']) 

    #put in data frame
    results_df = []
    data_dict = {'Fold':np.arange(n_splits)+1,\
                    'Type':['Train' for x in range(n_splits)]}
    for sidx in range(n_scores):
        data_dict['%s_score'%(score_list[sidx])] = train_scores_all[:,sidx]
    results_df.append(pd.DataFrame(data_dict))

    data_dict = {'Fold':np.arange(n_splits)+1,\
                    'Type':['Test' for x in range(n_splits)]}
    for sidx in range(n_scores):
        data_dict['%s_score'%(score_list[sidx])] = test_scores_all[:,sidx]
    results_df.append(pd.DataFrame(data_dict))

    results_df = pd.concat(results_df,axis = 0)

    return results_df, train_history

def rnn_xsubject_joint_data_train_all_subjects(feature_matrix, target_labels, sub_labels, block_labels, exclude,\
                                                    model_dict, score_list, nsets_training = 10,\
                                                    verbose = 0, epochs = 40, batch_size = 2, validation_split = 0.1, mv = False, permute = False):
    """
    train and validate a RNN model using data from multiple subjects 
    train on all subjects
    """

    #subjects in list. there are the units over which we will do train/test split
    subs = np.unique(sub_labels)

    if permute:
        #permute while ignoring excluded blocks
        target_labels = permute_class_within_sub(target_labels, block_labels, sub_labels, exclude)


    n_scores = len(score_list)
    train_subs = subs
    train_idxs = np.where(np.isin(sub_labels,train_subs, invert = False))[0]

    #get trained model
    train_scores, trained_model, scaler, history = get_trained_rnn_model(feature_matrix, target_labels, train_idxs, block_labels, nsets_training,\
                                                                        exclude, model_dict, score_list,\
                                                                        verbose = verbose, epochs = epochs, batch_size = batch_size,\
                                                                        validation_split = validation_split,\
                                                                        mv = mv)


    #put in data frame
    data_dict = {'Type':'Train'}
    for sidx in range(n_scores):
        data_dict['%s_score'%(score_list[sidx])] = train_scores[sidx]
    results_df = pd.DataFrame(data_dict, index = [0])


    return results_df, history, trained_model, scaler

def rnn_xsubject_transform_module_train_frac_subjects(feature_matrix, target_labels, sub_labels, block_labels, series_labels, exclude,\
                                                         model_dict,score_list = ['f1'],n_train_splits = 4, n_val_splits = 2,\
                                                          nsets_training = 10,verbose = 0, epochs = 40, batch_size = 2, mv = None,permute = False):
    """
    train and validate an RNN model with a transform module for domain adaptation. 
    train and validate model performance by splitting subjects into a train and test set
    """

    #default values
    if 'n_dense_post' not in model_dict.keys():
        model_dict['n_dense_post'] = 0
    if 'n_grus' not in model_dict.keys():
        model_dict['n_grus'] = 24

    if permute:
        #permute while ignoring excluded blocks
        target_labels = permute_class_within_sub(target_labels, block_labels, sub_labels, exclude)

    results_df = []
    #subjects in list. there are the units over which we will do train/test split
    subs = np.unique(sub_labels)

    #exclude indicated labels
    in_samples = np.where(np.isin(target_labels,exclude, invert = True))[0]

    #initialize object for k-fold cross-validation
    kf = KFold(n_splits=n_train_splits,shuffle = True)


    for split_count, (subs_train_idxs, subs_test_idxs) in enumerate(kf.split(subs)):
        print('-------Split Count: %i-------'% (split_count+1))
        #get train and test indices
        train_subs = subs[subs_train_idxs]
        test_subs = subs[subs_test_idxs]
        train_idxs = np.where(np.isin(sub_labels,train_subs, invert = False))[0]
        test_idxs = np.where(np.isin(sub_labels,test_subs, invert = False))[0]

        #get train and test indices
        train_subs = subs[subs_train_idxs]
        test_subs = subs[subs_test_idxs]
        train_idxs = np.where(np.isin(sub_labels,train_subs, invert = False))[0]
        test_idxs = np.where(np.isin(sub_labels,test_subs, invert = False))[0]
    
        #get training data cubes
        X_train_cube, Y_train_cube, scaler = prepare_data_for_RNN(feature_matrix, target_labels, train_idxs, exclude, train = True,\
                                                                    block_labels = block_labels, nsets = nsets_training)
        sub_labels_train = sub_labels[np.intersect1d(train_idxs,in_samples)]

        #testfor equal number of samples
        assert X_train_cube.shape[0] == Y_train_cube.shape[0]
        #testfor equal number of timepoints
        assert X_train_cube.shape[1] == Y_train_cube.shape[1]
        n_features, n_outputs = X_train_cube.shape[2], Y_train_cube.shape[2]

        # get testing data cubes
        X_test_cube, Y_test_cube, scaler = prepare_data_for_RNN(feature_matrix, target_labels, test_idxs, exclude, train = False, scaler = scaler)

        sub_labels_test = sub_labels[np.intersect1d(test_idxs,in_samples)]
        series_labels_test = series_labels[np.intersect1d(test_idxs,in_samples)]

        # permute order in which subjects' data is used for training
        train_subs_perm = np.random.permutation(train_subs)
        #initialize empty list
        n_scores = len(score_list)
        train_scores_all = np.empty((train_subs.size,n_scores))

        # --- Training Stage ---
        # Define model architecture

        #setting timestep dimension to None 
        model = get_rnn_model((None,n_features,),n_outputs,n_dense_pre=model_dict['n_dense_pre'], n_dense_post=model_dict['n_dense_post'],\
                            n_grus = model_dict['n_grus'], activation=model_dict['activation'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #model.summary()

        # # Get transform module template
        transform_module_template = get_transform_module(model, Input(shape = (None,n_features)),3 + model_dict['n_dense_post'])

        # iterate thorugh subjects' data
        for sub_idx, train_sub in enumerate(train_subs_perm):
            print('Training: Subject %02d out of %02d'%(sub_idx+1, train_subs.size))
            # get subject-specific samples
            train_sub_idxs = np.where(sub_labels_train == train_sub)[0]
            X_cube_sub = X_train_cube[:,train_sub_idxs,:]
            Y_cube_sub = Y_train_cube[:,train_sub_idxs,]

            # initialize weights of the transform module
            model = tm_template_weights_to_model(transform_module_template, model)

            print('Training Model')
            # fit network
            history = model.fit(X_cube_sub, Y_cube_sub, epochs=epochs, batch_size=batch_size, verbose=verbose)

            #copy weights to a transfer module template, save if wanted
            trained_transfer_module = model_weights_to_tm_template(transform_module_template, model)
            # # evaluate trained network
            print('Evaluate Model on Trained Data')

            if mv:
                #get f1 score after applying majority voting scheme to model predictions
                train_scores_all[sub_idx,:] = apply_mv_and_get_scores(feature_matrix, target_labels,\
                                                                np.intersect1d(np.where(sub_labels==train_sub)[0],train_idxs), exclude,\
                                                                scaler, model, mv, score_list, rnn = True)
            else:
                #get score for training data
                train_scores_all[sub_idx,:] = get_scores(X_cube_sub, Y_cube_sub, model, score_list, rnn = True)

        # #put results in dataframe
        data_dict = {'Subject':train_subs_perm+1,\
                        'Fold':[split_count+1 for x in range(train_subs_perm.size)],\
                        'Type':['Train' for x in range(train_subs_perm.size)]}

        for s_idx in range(n_scores):
            data_dict['%s_score'%(score_list[s_idx])] = train_scores_all[:,s_idx]
            
        results_df.append(pd.DataFrame(data_dict))

        # --- Validation Stage ---
        #freeze top layers
        for layer in model.layers[-3-(2*model_dict['n_dense_post']):]:
            layer.trainable = False

        # iterate through test subjects
        for sub_idx, test_sub in enumerate(test_subs):
            print('Validation: Subject %02d out of %02d'%(sub_idx+1, test_subs.size))

            #get relevant subject samples
            test_sub_idxs = np.where(sub_labels_test == test_sub)[0]
            test_series = series_labels_test[test_sub_idxs]
            
            X_cube_sub = X_test_cube[:,test_sub_idxs,:]
            Y_cube_sub = Y_test_cube[:,test_sub_idxs,]
            test_sub_labels = np.argmax(Y_cube_sub,1)

            #stratify split to retain ratio of class labels
            kf = KFold(n_splits=n_val_splits,shuffle = True)

            val_train_scores = np.empty((n_val_splits,n_scores))
            val_test_scores = np.empty((n_val_splits,n_scores))

            #systematically use one fold of the data as a held-out test set
            for split_count_val, (series_train, series_test) in enumerate(kf.split(np.unique(test_series))):
                
                #split data cubes into train and test subsets
                series_train_idxs = np.where(test_series==series_train)[0]
                X_train_cube_sub = X_cube_sub[:,series_train_idxs,:]
                Y_train_cube_sub = Y_cube_sub[:,series_train_idxs,:]


                series_test_idxs = np.where(test_series==series_test)[0]
                X_test_cube_sub = X_cube_sub[:,series_test_idxs,:]
                Y_test_cube_sub = Y_cube_sub[:,series_test_idxs,:]

                #initialize transform module
                model = tm_template_weights_to_model(transform_module_template, model)
                #train
                model.fit(X_train_cube_sub, Y_train_cube_sub, epochs=epochs, batch_size=batch_size, verbose=verbose)

                #copy weights to a transfer module template, save if wanted
                trained_transfer_module = model_weights_to_tm_template(transform_module_template, model)

                #evaluate on training and testing
                if mv:
                    #get f1 score after applying majority voting scheme to model predictions
                    val_train_scores[split_count_val,:]  = apply_mv_and_get_scores(feature_matrix[test_idxs], target_labels[test_idxs],\
                                                                    np.where(series_labels[test_idxs]==series_train)[0], exclude,\
                                                                    model = model, n_votes = mv, scaler = scaler, score_list = score_list, rnn = True)
                    val_test_scores[split_count_val,:]  = apply_mv_and_get_scores(feature_matrix[test_idxs], target_labels[test_idxs],\
                                                                    np.where(series_labels[test_idxs]==series_test)[0], exclude,\
                                                                    model = model, n_votes = mv, scaler = scaler, score_list = score_list, rnn = True)
                else:
                    val_train_scores[split_count_val,:] = get_scores(X_train_cube_sub, Y_train_cube_sub, model, score_list, rnn = True)
                    val_test_scores[split_count_val,:] = get_scores(X_test_cube_sub, Y_test_cube_sub, model, score_list, rnn = True)


            #put results in dataframe
            data_dict = {'Subject':[test_sub+1 for x in range(n_val_splits)],\
                            'Fold':[split_count+1 for x in range(n_val_splits)],\
                            'Type':['Val_Train' for x in range(n_val_splits)]}
            for sidx in range(n_scores):
                data_dict['%s_score'%(score_list[sidx])] = val_train_scores[:,sidx]
            results_df.append(pd.DataFrame(data_dict))

            data_dict = {'Subject':[test_sub+1 for x in range(n_val_splits)],\
                            'Fold':[split_count+1 for x in range(n_val_splits)],\
                            'Type':['Val_Test' for x in range(n_val_splits)]}
            for sidx in range(n_scores):
                data_dict['%s_score'%(score_list[sidx])] = val_test_scores[:,sidx]
            results_df.append(pd.DataFrame(data_dict))

            
    results_df = pd.concat(results_df,axis = 0)

    return results_df


