
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
from sklearn.metrics import f1_score,make_scorer, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold


from tensorflow import keras
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential, Model, load_model, Sequential, save_model
from tensorflow. keras.layers import Dense, Activation, Dropout, Input,  TimeDistributed, GRU, Masking, LSTM

from tensorflow.keras.utils import to_categorical

from EMG_gestures.utils import *

__all__ = ['within_subject_log_reg_performance','get_trained_model','evaluate_trained_log_reg','log_reg_xsubject_test',\
'log_reg_xsubject_transform_module_train_frac_subjects','log_reg_xsubject_transform_module_train_all_subjects']

# ~~~~~~~~ LOGISTIC REGRESSION FUNCTIONS ~~~~~~~~


def within_subject_log_reg_performance(X, Y, series_labels, exclude,  verbose = 0, epochs = 40, batch_size = 2, mv = False, permute = False):
    """
    Train and test performance of a logisitc regression model within the same subject
    """

    #initialize object for k-fold cross-validation
    n_splits = np.unique(series_labels).size
    kf = KFold(n_splits=n_splits,shuffle = True)
    #initialize empty arrays
    train_f1_scores = np.empty((n_splits,))
    test_f1_scores = np.empty((n_splits,))

    for split_count, (series_train, series_test) in enumerate(kf.split(np.unique(series_labels))):
        print('Split Count: %i'% (split_count+1))
        #get train and test idxs
        train_idxs = np.where(series_labels==series_train)[0]
        test_idxs = np.where(series_labels==series_test)[0]
        #get training data cubes
        X_train_cube, Y_train_cube, scaler = prepare_data_for_log_reg(X,Y, train_idxs, exclude, train = True)
        if permute:
            perm_idxs = np.random.permutation(np.arange(Y_train_cube.shape[0]))
            Y_train_cube = Y_train_cube[perm_idxs,:]

        n_features, n_outputs = X_train_cube.shape[1], Y_train_cube.shape[1]

        #setting timestep dimension to None 
        model = get_log_reg_model((n_features,),n_outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        #model.summary

        print('Training Model')
        # fit network
        history = model.fit(X_train_cube, Y_train_cube, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # # evaluate trained network
        print('Evaluate Model')
        

        if mv:
            # get testing data cubes
            X_test_cube, Y_test_cube, scaler = prepare_data_for_log_reg(X,Y, train_idxs, [], train = False, scaler = scaler)
            y_pred = get_mv_preds(X_test_cube, model, n_votes= 5)+1
            y_true = np.squeeze(np.argmax(Y_test_cube,1))
            include_idxs = np.where(np.isin(y_true,exclude, invert = True))[0]
            y_true = y_true[include_idxs]
            y_pred = y_pred[include_idxs]
            train_f1 = f1_score(y_true,y_pred,average = 'weighted')

            # get testing data cubes
            X_test_cube, Y_test_cube, scaler = prepare_data_for_log_reg(X,Y, test_idxs, [], train = False, scaler = scaler)
            y_pred = get_mv_preds(X_test_cube, model, n_votes= 5)+1
            y_true = np.squeeze(np.argmax(Y_test_cube,1))
            include_idxs = np.where(np.isin(y_true,exclude, invert = True))[0]
            y_true = y_true[include_idxs]
            y_pred = y_pred[include_idxs]
            test_f1 = f1_score(y_true,y_pred,average = 'weighted')
        else:
            #get score for training data
            train_f1 = get_log_reg_f1(X_train_cube, Y_train_cube, model)
            # get testing data cubes
            X_test_cube, Y_test_cube, scaler = prepare_data_for_log_reg(X,Y, test_idxs, exclude, train = False, scaler = scaler)
            #get score for testing data
            test_f1 = get_log_reg_f1(X_test_cube, Y_test_cube, model)
        #put scores in array
        train_f1_scores[split_count] = train_f1
        test_f1_scores[split_count] = test_f1

    return train_f1_scores, test_f1_scores


def get_trained_model(X, Y, train_idxs, exclude, verbose = 0, epochs = 40, batch_size = 2, mv = False, permute = False):

    #get training data cubes
    X_train_cube, Y_train_cube, scaler = prepare_data_for_log_reg(X,Y, train_idxs, exclude, train = True)
    if permute:
        perm_idxs = np.random.permutation(np.arange(Y_train_cube.shape[0]))
        Y_train_cube = Y_train_cube[perm_idxs,:]

    n_features, n_outputs = X_train_cube.shape[1], Y_train_cube.shape[1]

    #setting timestep dimension to None 
    model = get_log_reg_model((n_features,),n_outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary

    print('Training Model')
    # fit network
    history = model.fit(X_train_cube, Y_train_cube, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # # evaluate trained network
    print('Evaluate Model on Trained Data')

    if mv:
        # get testing data cubes
        X_test_cube, Y_test_cube, scaler = prepare_data_for_log_reg(X,Y, train_idxs, [], train = False, scaler = scaler)
        y_pred = get_mv_preds(X_test_cube, model, n_votes= 5)+1
        y_true = np.squeeze(np.argmax(Y_test_cube,1))
        include_idxs = np.where(np.isin(y_true,exclude, invert = True))[0]
        y_true = y_true[include_idxs]
        y_pred = y_pred[include_idxs]
        train_f1 = f1_score(y_true,y_pred,average = 'weighted')

    else:
        #get score for training data
        train_f1 = get_log_reg_f1(X_train_cube, Y_train_cube, model)
    return train_f1, model, scaler, history

def evaluate_trained_log_reg(X, Y, test_idxs, exclude, trained_model, scaler, mv):

    print('Evaluate Model')
    if mv:
        # get testing data cubes
        X_test_cube, Y_test_cube, scaler = prepare_data_for_log_reg(X,Y, test_idxs, [], train = False, scaler = scaler)
        y_pred = get_mv_preds(X_test_cube, trained_model, n_votes= 5)+1
        y_true = np.squeeze(np.argmax(Y_test_cube,1))
        include_idxs = np.where(np.isin(y_true,exclude, invert = True))[0]
        y_true = y_true[include_idxs]
        y_pred = y_pred[include_idxs]
        test_f1 = f1_score(y_true,y_pred,average = 'weighted')
    else:

        # get testing data cubes
        X_test_cube, Y_test_cube, scaler = prepare_data_for_log_reg(X,Y, test_idxs, exclude, train = False, scaler = scaler)
        #get score for testing data
        test_f1 = get_log_reg_f1(X_test_cube, Y_test_cube, trained_model)
    return test_f1

def log_reg_xsubject_test(data_folder, src_subject_id, nsubjects, nreps, lo_freq, hi_freq, win_size, step, exclude, \
                          verbose = 0, epochs = 40, batch_size = 2, mv = False, permute = False):
    
    
    subject_folder = os.path.join(data_folder,'%02d'%(src_subject_id))
    print('=======================')
    print(subject_folder)

    # Process data and get features 
    #get features across segments and corresponding info
    feature_matrix_src, target_labels_src, window_tstamps_src, \
    block_labels_src, series_labels_src = get_subject_data_for_classification(subject_folder, lo_freq, hi_freq, \
                                                                win_size, step)
    train_idxs = np.arange(target_labels_src.size)
    np.random.seed(1)#for reproducibility

    results_df = []#initialize empty array for dataframes
    train_f1_scores = np.empty((nreps,))
    for rep in range(nreps):

        print('Subject %d|Rep %d'%(src_subject_id, rep+1))
        train_f1, trained_model, scaler, train_history = get_trained_model(feature_matrix_src, target_labels_src, train_idxs, exclude,\
                                                                        verbose = verbose, epochs = epochs, batch_size = batch_size,\
                                                                        mv = mv, permute = permute)
        train_f1_scores[rep] = train_f1
        # test on all other subjects
        # initialize empty lists
        test_f1_scores = np.empty((0,))
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

                test_f1 = evaluate_trained_log_reg(feature_matrix_targ, target_labels_targ, test_idxs, exclude, trained_model, scaler, mv = mv)
                #append to list
                test_f1_scores = np.hstack((test_f1_scores, test_f1))
                targ_subject_list.append(targ_subject_id)

        #put test results in dataframe
        results_df.append(pd.DataFrame({'F1_score':test_f1_scores,\
                                                    'Type':['Test' for x in range(test_f1_scores.size)],\
                                                    'Rep':[rep+1 for x in range(test_f1_scores.size)],\
                                                'Test_Subject':targ_subject_list}))
    #put training results in dataframe
    results_df.append(pd.DataFrame({'F1_score':train_f1_scores,\
                                    'Type':['Train' for x in range(train_f1_scores.size)],\
                                    'Rep': np.arange(nreps)+1,\
                                    'Test_Subject':[src_subject_id for x in range(train_f1_scores.size)]}))
    
    results_df = pd.concat(results_df, axis = 0).reset_index(drop = True)

    return results_df

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
        X_train_cube, Y_train_cube, scaler = prepare_data_for_log_reg(feature_matrix, target_labels, train_idxs, exclude, train = True)
        sub_labels_train = sub_labels[np.intersect1d(train_idxs,in_samples)]


        #testfor equal number of samples
        assert X_train_cube.shape[0] == Y_train_cube.shape[0]
        n_features, n_outputs = X_train_cube.shape[1], Y_train_cube.shape[1]

        # get testing data cubes
        X_test_cube, Y_test_cube, scaler = prepare_data_for_log_reg(feature_matrix, target_labels, test_idxs, exclude, train = False, scaler = scaler)
        
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
        model = get_log_reg_model((n_features,),n_outputs, n_dense_pre=model_dict['n_dense_pre'], activation=model_dict['activation'])
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
                    val_train_scores[split_count_val,:]  = apply_mv_and_get_f1_scores(feature_matrix[test_idxs], target_labels[test_idxs],\
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
    X_train_cube, Y_train_cube, scaler = prepare_data_for_log_reg(feature_matrix, target_labels, train_idxs, exclude, train = True)
    in_samples = np.where(np.isin(target_labels,exclude, invert = True))[0]
    sub_labels_train = sub_labels[np.intersect1d(train_idxs,in_samples)]

    #test for equal number of samples
    assert X_train_cube.shape[0] == Y_train_cube.shape[0]
    n_features, n_outputs = X_train_cube.shape[1], Y_train_cube.shape[1]

    if test_idxs.size>0:
        #get testing data cubes
        X_test_cube, Y_test_cube, scaler = prepare_data_for_log_reg(feature_matrix, target_labels, test_idxs, exclude, train = False, scaler = scaler)
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
    model = get_log_reg_model((n_features,),n_outputs, n_dense_pre=model_dict['n_dense_pre'], activation=model_dict['activation'])
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