
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 2021
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

from tensorflow.keras.utils import to_categorical

__all__ = ['get_gesture_times','get_steady_samp_rate_data','butter_bandpass_filter','get_window_features',\
'window_and_get_features','parse_data_blocks','get_file_data_for_classification','get_subject_data_for_classification',\
'get_data_cube',\
'plot_sensor_values','plot_signal_pspec','visualize_time_series_prob','dim_reduction_visualization',\
'plot_train_loss','plot_training_history',\
'preds_to_scores','get_scores',\
'shift_array','get_mv_preds','apply_mv_and_get_scores',\
'permute_class_blocks','permute_class_within_sub',\
'get_log_reg_model','get_log_reg_f1','prepare_data_for_log_reg',\
'get_transform_module','tm_template_weights_to_model','model_weights_to_tm_template']


# ~~~~~~~ DATA WRANGLING FUNCTIONS ~~~~~~~
def get_gesture_times(data_df):
    """
    Get start times, end times, and event labels of each block of time with a given label
    
    Args:
        data_df: dataframe with columns for 'class' and 'time' 
            
    Returns:
        start times, end times, and event labels: 1-d numpy arrays 
        
    """

    #get start indices of each condition 
    
    #taking advantage that every gesture is preceded by a period with 'undefined' class label (value 0)
    start_idxs = np.hstack([0,np.where(np.abs(np.diff(data_df['class']))>0)[0]+1])
    
    #end indices correspond to just before the start of the next condition
    end_idxs = np.hstack((start_idxs[1:]-1,data_df.time.size-1))
    

    #create arrays with start times and class labels for each hand gesture
    start_times = data_df.time[start_idxs].values
    end_times = data_df.time[end_idxs].values
    event_labels = data_df['class'][start_idxs].values
    
    return start_times, end_times, event_labels

def get_steady_samp_rate_data(data_df):
    
    """
    Resample time series data for a steady sampling rate by performing linear interpolation
    between missing samples
    
    Args:
        data_df: dataframe with a 'time' column and timeseries of interest on other columns
            
    Returns:
        data_df: dataframe with interpolated 
    """
    
    #retrieve time value
    time = data_df.time.values
    #define time points on which we want to interpolate (i.e., a sample every ms)
    time_new = np.arange(time[0],time[-1]+1,1)
    
    #create separate dataframe with new time points and set these as index
    steady_time_df = pd.DataFrame({'time_constant':time_new})
    steady_time_df = steady_time_df.set_index('time_constant')
    
    #set time coulumn of original dataframe to index
    data_df = data_df.set_index('time')

    #perform a right join with the steady time dataframe then linearly interpolate missing values
    data_df = data_df.join(steady_time_df, how = 'right')
    data_df = data_df.interpolate().reset_index().rename(columns={'time_constant':'time'})
    
    return data_df

def butter_bandpass_filter(input_signal, lowcut, highcut, fs, order=4, axis = -1):
    #lowcut = lower bound of desired freq band
    #hicut = upperbound bound of desired freq band
    #fs = sampling rate
    #order = order of filter
    #axis = axis of data matrix on which to apply filters

    low = float(lowcut)# / nyq
    high = float(highcut)# / nyq
    b, a = scipy.signal.butter(order, [low, high],fs = fs, btype='band')
    
    output_signal = scipy.signal.filtfilt(b, a, input_signal,axis = axis)
    
    return output_signal
    
def get_window_features(signal):
    #signal: EMG signal matrix with dims time x channels
    #return: Mean Absolute Value and Wavelength features for each channel

    MAV = np.mean(np.absolute(signal),axis = 0) #Mean Absolute Value

    WL = np.sum(np.absolute(np.diff(signal,axis = 0)),axis = 0) #Wavelength
    
    return MAV, WL

def window_and_get_features(parsed_df,win_size,step):
    """
    Compute features (Mean Absolute Value and Wavelength) for time series segments with given length 
    and amount of overlap
    
    Args:
        parsed_df: dataframe containing timeseries data for each block. expects a 'Block' column
        and the channel data in the first columns of dataframe
        
            
    Returns:
        feat_matrix: 2D numpy array of size [number of segments, number of channels * number of features]
        target_labels: 1D numpy array with class label for each segment
        window_tstamps: 1D numpy array with timestamp for each timeseries segment used to compute features
        block_labels: 1D numpy array indicating block provenance of each segment (useful for RNN data prep)
    """
    #set empty lists
    feat_matrix = []
    target_labels = []
    window_tstamps = []
    block_labels = []

    #get number of channels
    nchannels = np.sum(['Channel' in col_name for col_name in parsed_df.columns])

    for block in parsed_df.Block.unique():
        #get relevant subset of dataframe
        block_df = parsed_df.loc[parsed_df.Block == block]

        #extract relevant info
        block_data_matrix = block_df.iloc[:,0:8].values
        block_class = block_df['Class'][0]
        tstamps = block_df['Time'].values

        #compute desired features over segments of the data
        for win_start_idx in range(0, block_df.shape[0], step):
            win_end_idx = win_start_idx + win_size
            if win_end_idx< block_df.shape[0]: #exclude window if not enough timepoints before end of block

                #compute features within this segment
                MAV, WL = get_window_features(block_data_matrix[win_start_idx:win_end_idx,:])

                #append info to matrices
                feat_matrix.append(np.hstack((MAV,WL)))
                target_labels.append(block_class)
                window_tstamps.append(np.mean([tstamps[win_start_idx],tstamps[win_end_idx]]))
                block_labels.append(block)
                
    return np.array(feat_matrix), np.array(target_labels), np.array(window_tstamps), np.array(block_labels)



def parse_data_blocks(start_times, end_times, event_labels, data_matrix, tstamps, pre_tpts = 0, exclude_class=[]):
    """
    Function to parse timeseries into blocks corresponding to each event. 
    Returns a dataframe - seemed most convenient given variable length of event blocks
    
    Args:
        start_times: 1D numpy array with start times of event blocks
        end_times: 1D numpy array with start times of event blocks
        event_labels: 1D numpy array with start times of event blocks
        data_matrix: 2D numpy array with of dimension [ntimepoints, nchannels] containing signal values
        pre_tpts: how many timepoints before event onset to include(int)
        exclude_class: list with class labels to exclude in parsing blocks
            
    Returns:
        parsed_df: dataframe with parsed data
        block_length: 1D numpy array with timepoints in each event block
    """
    
    parsed_df = []

    #note characterisitcs of matrix
    ntpts, nchannels = data_matrix.shape

    parsed_df = []
    block_lengths = []
    for block,c in enumerate(event_labels):
        if c not in exclude_class:
            # get relevant indives to get dara from matrix
            start_idx = np.where(tstamps>start_times[block])[0][0]
            end_idx = np.where(tstamps>=end_times[block])[0][0]

            # append block length to list
            block_lengths.append(end_idx-start_idx)

            # get timestamps relative to event onset
            t_ase = np.arange(-pre_tpts,end_idx-start_idx)

            # put data matrix values into a dataframe
            block_df = pd.DataFrame(data_matrix[start_idx-pre_tpts:end_idx,:], columns = ['Channel_%i'%(c+1) for c in range(nchannels)])

            # add relevant value columns
            block_df['Block'] = block
            block_df['Class'] = c 
            block_df['Time_ASE'] = t_ase
            block_df['Time'] = np.arange(tstamps[start_idx]-pre_tpts,tstamps[end_idx])

            #append to collecting datafram
            parsed_df.append(block_df)

    #merge all dataframes together
    parsed_df = pd.concat(parsed_df,axis = 0)
    
    return parsed_df, block_lengths

def get_file_data_for_classification(data_fn, lo_freq, hi_freq, win_size, step, remove_file_mean = True):
    """
    Get the relevant info for classification from indicated file
    
    Args:
        data_fn: filename
        lo_freq: lower bound of bandpass filter
        hi_freq: higher bound of bandpass filter
        win_size: length of segment over which to compute features
        step: amount of overlap between neighboring segments
        remove_file_mean: Boolean indicating removal of mean value for each channel in the series
            
    Returns:
        feature_matrix: 2D numpy array of size [number of segments, number of channels * number of features]
        target_labels: 1D numpy array with class label for each segment
        window_tstamps: 1D numpy array with timestamp for each timeseries segment used to compute features
        block_labels: 1D numpy array indicating block provenance of each segment (useful for RNN data prep)
    """
    
    # load file
    data_df = pd.DataFrame(pd.read_csv(data_fn, sep='\t'))
    
    # retrieve start times, end times, and labels for each condition block (will make it easier to parse traces later)
    start_times, end_times, event_labels  = get_gesture_times(data_df)

    # interpolate data to steady frame rate
    data_df = get_steady_samp_rate_data(data_df)
    samp_period = np.diff(data_df.time)[0]/1000.0
    samp_rate = 1/samp_period
    
    # Unpack values into numpy arrays
    data_matrix = data_df.iloc[:,1:-1].values
    class_labels = data_df.iloc[:,-1].values
    tstamps = data_df.time.values
    
    #remove series offset for each channel, if indicated
    if remove_file_mean:
        data_matrix = data_matrix- np.mean(data_matrix,0)

    # filter data
    filt_data_matrix = butter_bandpass_filter(data_matrix, lo_freq, hi_freq, samp_rate,axis = 0)

    # parse timeseries into block corresponding to different blocks of time
    parsed_df, block_lengths = parse_data_blocks(start_times,end_times,event_labels,filt_data_matrix, tstamps,
                                                 pre_tpts = 0, exclude_class=[])

    # compute desired features over individual time segments
    feature_matrix, target_labels, window_tstamps, block_labels = window_and_get_features(parsed_df,win_size,step)
    
    return feature_matrix, target_labels, window_tstamps, block_labels


def get_subject_data_for_classification(data_folder, lo_freq, hi_freq, win_size, step):
    """
    Get the relevant info for classification from indicated file
    
    Args:
        data_fn: filename
        lo_freq: lower bound of bandpass filter
        hi_freq: higher bound of bandpass filter
        win_size: length of segment over which to compute features
        step: amount of overlap between neighboring segments
            
    Returns:
        feature_matrix: 2D numpy array of size [number of segments, number of channels * number of features]
        target_labels: 1D numpy array with class label for each segment
        window_tstamps: 1D numpy array with timestamp for each timeseries segment used to compute features
        block_labels: 1D numpy array indicating block provenance of each segment (useful for RNN data prep)
        series_labels: 1D numpy array indicating block provenance of each segment (useful for visualization)
    """
    
    #find files in subject folder
    file_list = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]


    #initialize empty matrices
    feature_matrix = np.empty((0,0))
    target_labels = np.empty((0,))
    window_tstamps = np.empty((0,))
    block_labels = np.empty((0,))
    series_labels = np.empty((0,))
    max_block_id = 0

    for series_id,file in enumerate(file_list):

        #get relevant info from each file
        feature_matrix_sub, target_labels_sub,\
        window_tstamps_sub, block_labels_sub = get_file_data_for_classification(os.path.join(data_folder,file),\
                                                                                 lo_freq, hi_freq, win_size, step)
        nsamples,nfeats = feature_matrix_sub.shape

        #offset block labels
        block_labels_sub = block_labels_sub+max_block_id
        max_block_id = np.max(block_labels_sub)#update

        series_labels_sub = np.ones((nsamples,))*series_id#make array with series ID of samples

        #append file samples
        feature_matrix = np.vstack((feature_matrix,feature_matrix_sub)) if feature_matrix.size else feature_matrix_sub
        target_labels = np.hstack((target_labels,target_labels_sub))
        window_tstamps = np.hstack((window_tstamps,window_tstamps_sub))
        block_labels = np.hstack((block_labels,block_labels_sub))
        series_labels = np.hstack((series_labels,series_labels_sub))
        
    return feature_matrix, target_labels, window_tstamps, block_labels, series_labels


def get_data_cube(X, window_blocks, train = True, scaler = None, magic_value = -100):
    """
    Create data cube for use with Keras RNN. Standardize data then pad and reshape data to have
    [samples, timesteps, features] dimensions with an equal number of timesteps for each slice
    I use a Masking layer in the RNN architecture to allow for sequences of different length
    
    Args:
        X: 2D nuumpy array with data, dimensions [features, samples]
        window_blocks: 1D numpy array indicating block of provenance for input segment values
        train: Boolean indicating whether the input data is training data
        scaler: StandardScaler to transform data
        magic_value: integer indicating value with which to pad samples
            
    Returns:
        X_cube: 3D numpy array of size [samples, timesteps, features]
        scaler: 1D numpy array with class label for each segment
    """
    #standardize across each feature dimension
    if train:
        scaler = StandardScaler()
        scaler = scaler.fit(X.T)
        X = scaler.transform(X.T).T
    else:
        #for testing data, we want to use same transform as was fit to training data
        X = scaler.transform(X.T).T
        
    # common number of time steps
    common_timesteps = np.max(np.bincount(window_blocks.astype('int')))
    
    # get each block, pad, and stack to form a data cube
    X_cube = []
    for b_count, b_idx in enumerate(np.unique(window_blocks)):
       #slice
        X_slice = X[:,np.where(window_blocks==b_idx)[0]]
        #pad
        pad_size = common_timesteps-X_slice.shape[1]
        X_slice_pad = np.pad(X_slice,pad_width=((0,0),(0,pad_size)), mode='constant', constant_values= magic_value)
        #stack
        if b_count == 0:
            X_cube  = X_slice_pad
        else:
            X_cube = np.dstack((X_cube,X_slice_pad))

    # swap dimension to get [samples, timesteps, features]
    X_cube = np.swapaxes(X_cube,0,2)
    
    return X_cube, scaler

#~~~~~ VISUALIZATION FUNCTIONS ~~~~~~

def plot_sensor_values(data_fn, x_limits = []):
    """
    Plot signal timecourse for all channels using data in indicated file
    
    Args:
        data_fn: filename
        x_limits: minimum and maximum limits for x-axis (useful to looking at specific sections)
 
    Returns:
        fig: figure handle
    """

    # load file
    data_df = pd.DataFrame(pd.read_csv(data_fn, sep='\t'))

    # retrieve start times, end times, and labels for each condition block (will make it easier to parse traces later)
    start_times, end_times, event_labels  = get_gesture_times(data_df)

    # interpolate data to steady frame rate
    data_df = get_steady_samp_rate_data(data_df)

    # Unpack values into numpy arrays
    data_matrix = data_df.iloc[:,1:-1].values
    class_labels = data_df.iloc[:,-1].values
    tstamps = data_df.time.values

    # define color palette
    palette = sns.color_palette('deep',8)

    # define class legend
    labels = []
    custom_lines = []

    classes = np.unique(event_labels)[1:]#exclude 'unmarked' label
    for c in classes.astype('int'):
        labels.append('Class %i'%(c))
        custom_lines.append(Line2D([0], [0], color=palette[c-1], lw=4))

    #make figure
    nrows = 8
    ncols = 1

    fig,ax = plt.subplots(nrows,ncols,figsize=(15,30),sharex = True)

    #plot each channel
    for ch in range(8):
        ax[ch//ncols].plot(tstamps,data_matrix[:,ch]);
        ax[ch//ncols].axhline(y = 0, xmin = 0, xmax = 1, color = 'k', linestyle = '--', alpha = 0.5)
        #label subplot
        ax[ch//ncols].set_title('Channel %i'%(ch+1))

        #mark events
        ymin,ymax = ax[ch//ncols].get_ylim()
        for idx,c in enumerate(event_labels):
            if c>0:#eclude 'unmarked label'
                ax[ch//ncols].hlines(y = ymax + .001, xmin = start_times[idx], xmax = end_times[idx], color = palette[int(c-1)],linewidth = 10)

        if len(x_limits):
            ax[ch//ncols].set_xlim(x_limits)


    #set legend with events
    ax[0].legend(custom_lines, labels,bbox_to_anchor=(1, 1), loc='upper left', ncol=1)


    #label axes
    ax[0].set_ylabel('Sensor Voltage')
    ax[ch].set_xlabel('Time (ms)')

    #despine
    sns.despine(fig= plt.gcf(), left = False, right = True, top = True, bottom = True)


    fig.tight_layout() 

    return fig

def plot_signal_pspec(data_fn):
    """
    Plot power spectrum of signal for each channel using data in datafile indicated
    
    Args:
        data_fn: filename
 
    Returns:
        fig: figure handle
    """

    # load file
    data_df = pd.DataFrame(pd.read_csv(data_fn, sep='\t'))

    # retrieve start times, end times, and labels for each condition block (will make it easier to parse traces later)
    start_times, end_times, event_labels  = get_gesture_times(data_df)

    # interpolate data to steady sampling rate
    data_df = get_steady_samp_rate_data(data_df)
    samp_period = np.diff(data_df.time)[0]/1000.0
    samp_rate = 1/samp_period

    # Unpack values into numpy arrays
    data_matrix = data_df.iloc[:,1:-1].values
    class_labels = data_df.iloc[:,-1].values
    tstamps = data_df.time.values

    #Calculate the Welch's PSD of the data - yields a smoother, more informative, spectrum 
    f,pspec = scipy.signal.welch(data_matrix, fs=samp_rate, window='hanning', nperseg=2*samp_rate, noverlap=samp_rate/2,
                              nfft=None, detrend='linear', return_onesided=True, scaling='density',axis = 0)

    #make figure
    nrows = 4
    ncols = 2
    fig,ax = plt.subplots(nrows,ncols,figsize=(16,10),sharey = True, sharex = True)

    for ch in range(data_matrix.shape[1]):
        ax[ch//ncols][ch%ncols].loglog(f[1:200*2],pspec[1:200*2,ch])#line noise will be obvious under 100 Hz
        #label subplot

        ax[ch//ncols][ch%ncols].set_title('Channel %i'%(ch+1))
    #label axes
    ax[0][0].set_ylabel('Power')
    ax[ch//ncols][0].set_xlabel('Frequency')

    #despine
    sns.despine(fig= plt.gcf(), left = False, right = True, top = True, bottom = False)

    fig.tight_layout() 

    return fig

def visualize_time_series_prob(data_folder, prob_class, times, series_labels):
    """
    Visualize probability of each class across time for individual files
    
    Args:
        data_folder: folder with subject data
        prob_class: array with probability of each class for each sample
        times: array with timestamps for each signal segment
        series: array with file of provenance for each singla segment
            
    Returns:
        figure
    """

    #find files in subject folder
    file_list = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f))]

    for file_idx in range(len(file_list)):

        # load file
        data_df = pd.DataFrame(pd.read_csv(os.path.join(data_folder,file_list[file_idx]), sep='\t'))

        # retrieve start times, end times, and labels for each condition block (will make it easier to parse traces later)
        start_times, end_times, event_labels  = get_gesture_times(data_df)

        series_idxs = np.where(series_labels==file_idx)[0]

        prob_series = prob_class[series_idxs,:]
        time_series = times[series_idxs]

        classes = np.unique(event_labels)[1:]#exclude 'unmarked' label
        labels = []
        for c in classes.astype('int'):
            labels.append('Class %i'%(c))

        # define color palette
        palette = sns.color_palette('deep',8)[1:]

        plt.figure(figsize=(15,6))
        plt.gca().set_prop_cycle(plt.cycler('color',palette))
        plt.plot(time_series,prob_series, linewidth = 2);

        #mark events
        for idx,c in enumerate(event_labels):
            if c>0:
                plt.hlines(y = 1.1, xmin = start_times[idx], xmax = end_times[idx], color = palette[c-1],linewidth = 10)

        #label axes
        plt.ylabel('Class Probability')
        plt.xlabel('Time (ms)')

        plt.legend(labels, bbox_to_anchor=(1, 1), loc='upper left', ncol=1)

        sns.despine(fig= plt.gcf(), left = False, right = True, top = True, bottom = False)

        plt.gcf().tight_layout() 

        plt.gcf().suptitle('%s'%(file_list[file_idx]), y= 1.05)

    return plt.gcf()

def dim_reduction_visualization(X, target_labels):
    """
    Perform dimensionality reduction with tSNE and visualize results.
    
    Args:
        X: 2D numpy array with data [samples, features]
        target_labels: array. used to color points in embedded space
 
    Returns:
        fig: figure handle
    """
    
    #make pipeline
    tsne_pipe = make_pipeline(StandardScaler(),#standardize
                              TSNE(n_components=2, perplexity = 50))
    #perform embedding
    X_embedded = tsne_pipe.fit_transform(X)
    
    # visualize
    palette = sns.color_palette('deep',8)

    # define class legend
    labels = []
    custom_lines = []

    classes = np.unique(target_labels)#exclude 'unmarked' label
    for c in classes.astype('int'):
        labels.append('Class %i'%(c))
        custom_lines.append(Line2D([0], [0], color=palette[c], lw=4))

    sns.set_context('paper',font_scale = 2)
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    
    #plot
    for s in range(X_embedded.shape[0]):
        ax.scatter(X_embedded[s,0],X_embedded[s,1],\
                        color = palette[int(target_labels[s])],s = 100,linewidth = 2,alpha = 0.8)
    ax.axhline(y=0 ,xmin = 0,xmax = 1,color = 'k',linestyle = '--')    
    ax.axvline(x=0 ,ymin = 0,ymax = 1,color = 'k',linestyle = '--') 
    #add legend
    ax.legend(custom_lines, labels,bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    #label
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    sns.despine(trim=False, offset=0, bottom=False,top = True, left=False, ax=ax)
    
    return fig
    
def plot_train_loss(train_history, fig_title,fig_fn):
    #make df

    fig,ax = plt.subplots(1,1,figsize=(11,5))
    ax.plot(train_history.history['loss'])
    if 'val_loss' in train_history.history.keys():
        ax.plot(train_history.history['val_loss'])
    #label axes
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['train', 'validation'], bbox_to_anchor=(1, 1), loc='upper left')
    #despine
    sns.despine(fig= plt.gcf(), left = False, right = True, top = True, bottom = True)
    fig.tight_layout()
    fig.suptitle(fig_title, y = 1.05)
    fig.savefig(fig_fn,dpi = 300)
    plt.close()

def plot_training_history(train_history, fig_title,fig_fn):
    
    #make dataframe
    history_df = []
    n_folds, n_epochs = train_history['loss'].shape
    for f in range(n_folds):
        history_df.append(pd.DataFrame({'Loss': train_history['loss'][f,:],\
                        'Epoch':np.arange(n_epochs)+1,\
                        'Fold':[f+1 for x in range(n_epochs)],\
                        'Type':['Train' for x in range(n_epochs)]}))
        history_df.append(pd.DataFrame({'Loss': train_history['val_loss'][f,:],\
                        'Epoch':np.arange(n_epochs)+1,\
                        'Fold':[f+1 for x in range(n_epochs)],\
                        'Type':['Val' for x in range(n_epochs)]}))

    history_df = pd.concat(history_df,axis = 0)


    #make figure
    sns.relplot(data=history_df, x = 'Epoch', y = 'Loss', col = 'Fold',col_wrap = 2, style = 'Type',\
                style_order = ['Train','Val'],kind = 'line')
    fig = plt.gcf()
    fig.suptitle(fig_title, y = 1.05)
    fig.savefig(fig_fn,dpi = 300)
    plt.close()

# ~~~~~~~~ SCORING FUNCTIONS ~~~~~~~~

def preds_to_scores(y_pred, y_true, score_list, average = 'weighted'):
    """
    use predictions and ground truth to compute indicated performance scores
    """
    # Get scores
    out_scores = np.empty((0,0))
    for score in score_list:
        if score == 'f1':
            score_val = f1_score(y_true,y_pred,average = average)
        elif score == 'accuracy':
            score_val = accuracy_score(y_true,y_pred)
        else:
            raise NotImplementedError
        #stack
        out_scores = np.hstack((out_scores,score_val)) if out_scores.size else score_val
    return out_scores

def get_scores(X, Y, model, score_list ,average = 'weighted', mask_value = -100):
    """
    Get indicated performance scores for a trained model. can mask out samples

    Args:
        X: 2D numpy array with shape [samples, features]
        Y: 2D numpy array with shape [samples,classes]. one-hot coding of classes
        model: trained model object
        average: string argument for f1_score function. Usually 'macro' or 'weighted'
        mask_value: value indicating which samples to mask out

    Returns:
        out_scores
    """
    # Mask out indices based on mask value
    nonmasked_idxs = np.where(Y[:,0].flatten()!=mask_value)[0]
    # Get target labels for non-masked timepoints
    y_true = np.argmax(Y,1).flatten()[nonmasked_idxs]
    # Get model predictions for non-masked timepoints
    preds = model.predict(X)
    y_pred = np.argmax(preds,1).flatten()[nonmasked_idxs]

    return preds_to_scores(y_pred, y_true, score_list, average = average)


# ~~~~~~~~ MAJORITY VOTING FUNCTIONS ~~~~~~~~

def shift_array(arr, num, fill_value=np.nan):
    """
    shift and stack arrays
    """

    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def get_mv_preds(X, model, n_votes):
    """
    get predictions and refine by applying majority voting scheme
    """

    y_prob = np.squeeze(model.predict(X))
    y_pred = np.argmax(y_prob,1)

    y_stack = y_pred.astype('float').copy()
    y_last = y_pred.astype('float').copy()

    for n in range(n_votes):
        y_shifted = shift_array(y_last,1)
        y_stack = np.vstack((y_stack,y_shifted))
        y_last = y_shifted.copy()

    y_pred_mv, vote_counts = scipy.stats.mode(y_stack,0,nan_policy='omit')
    y_pred_mv = np.squeeze(y_pred_mv.data)

    return y_pred_mv




def apply_mv_and_get_scores(X, Y, subset_idxs, exclude,\
                              scaler = None, model = None, n_votes = 5,\
                             score_list = ['f1'],average = 'weighted'):
    """
        apply majority voting scheme after predicting output with model
        for this to make sense, adjacent samples should have been acquired at adjacent times
    """
    if subset_idxs.size >0:
        # standardize and format original data
        X, Y, scaler = prepare_data_for_log_reg(X,Y, subset_idxs, [], train = False, scaler = scaler)
    print(X.shape)
    #get majority voting prediction
    y_pred = get_mv_preds(X, model, n_votes= n_votes)+1
    # get grount-truth labels
    y_true = np.squeeze(np.argmax(Y,1))

    # exclude indicated conditions
    include_idxs = np.where(np.isin(y_true,exclude, invert = True))[0]
    print(include_idxs.shape)
    y_true = y_true[include_idxs]
    y_pred = y_pred[include_idxs]

    return preds_to_scores(y_pred, y_true, score_list, average = average)

# ~~~~~~~~ DATA PERMUTATION FUNCTIONS ~~~~~~~~

def permute_class_blocks(block_labels,class_labels):
    """
    Permute class labels while retaining the same class label within each block

    Args:
      block_labels: 1D numpy array indicating block of provenance for each sample
      class_labels: 1D numpy array indicating class label for each sample
          
    Returns:
      class_labels_shuffled: permuted class labels for each sample
      classes_perm: permuted class labels for each blocks
    """

    # #get block_ids and corresponding classes in block
    blocks = np.array([k for k,g in groupby(block_labels)])
    classes = np.array([k for k,g in groupby(class_labels) if k!=0])


    #using indexing tricks to have this work out
    classes_perm = np.random.permutation(classes)
    class_labels_shuffled = np.empty((0,))
    for i,b in enumerate(blocks):
        idxs = np.where(block_labels==b)[0]
        class_labels_shuffled = np.hstack((class_labels_shuffled,classes_perm[i]*np.ones((idxs.size,))))

    return class_labels_shuffled, classes_perm

def permute_class_within_sub(target_labels, block_labels, sub_labels, exclude):
    """
    Permute class blocks within each subject
    """

    subs = np.unique(sub_labels)
    #permute labels within each subject
    class_perm = np.empty((0,))
    for s in subs:
        sub_idxs = np.where(sub_labels==s)[0]
        blocks_sub = block_labels[sub_idxs]
        class_sub = target_labels[sub_idxs]

        #exclude indicated samples
        in_samples = np.where(np.isin(class_sub,exclude, invert = True))[0]
        class_sub_perm = class_sub.copy()
        # shuffle non-excluded blocks
        class_sub_perm_tmp, dummy = permute_class_blocks(blocks_sub[in_samples], class_sub[in_samples])
        class_sub_perm[in_samples] = class_sub_perm_tmp
        class_perm = np.hstack((class_perm, class_sub_perm))
    target_labels = class_perm.copy()
    return target_labels




# ~~~~~~~~ LOGISTIC REGRESSION FUNCTIONS ~~~~~~~~

def get_log_reg_model(input_shape, n_outputs, n_dense_pre = 0, drop_prob = 0.5, activation = 'tanh'):
    """
    Create 
    
    Args:
        input_shape
        n_outputs: number of output classes
        mask_value: value indicating which timepoints to mask out
            
    Returns:
        model
    """
    
    #define model architecture
    X_input = Input(shape = input_shape)
    X = X_input
    for n in range(n_dense_pre):
        X = Dense(input_shape[0],activation = activation)(X)
        X = Dropout(drop_prob)(X)
    X = Dense(n_outputs,activation = 'softmax')(X)
    model = Model(inputs = X_input, outputs = X)
    return model

def get_log_reg_f1(X, Y, model, average = 'weighted', mask_value = -100):
    """
    Get f1 score for an RNN model using masked timepoint data

    Args:
        X: 3D numpy array with shape [samples, timepoints, features]
        Y: 3D numpy array with shape [samples, timepoints, classes]. one-hot coding of classes
        model: RNN model object
        average: string argument for f1_score function. Usually 'macro' or 'weighted'
        mask_value: value indicating which timepoints to mask out

    Returns:
        f1: f1 score
    """
    # Mask out indices based on mask value
    nonmasked_idxs = np.where(Y[:,0].flatten()!=mask_value)[0]
    # Get target labels for non-masked timepoints
    y_true = np.argmax(Y,1).flatten()[nonmasked_idxs]
    # Get model predictions for non-masked timepoints
    preds = model.predict(X)
    y_pred = np.argmax(preds,1).flatten()[nonmasked_idxs]
    # Get F1 score
    f1 = f1_score(y_true,y_pred,average = average)

    return f1

def prepare_data_for_log_reg(X,Y, select_idxs, exclude_labels, train = False,scaler = None):

    """
    function to prepare data in format expected by tensorflow functions
    """

    X_cube =  X[select_idxs,:]
    Y_cube = Y[select_idxs]

    if train:
        scaler = StandardScaler()
        scaler = scaler.fit(X_cube)
        X_cube = scaler.transform(X_cube)
    else:
        X_cube = scaler.transform(X_cube)

    include_idxs = np.where(np.isin(Y_cube,exclude_labels, invert = True))[0]

    X_cube = X_cube[include_idxs,:]
    Y_cube = Y_cube[include_idxs]
    Y_cube = to_categorical(Y_cube-np.min(Y_cube))

    return X_cube, Y_cube, scaler




# ~~~~~~~~ TRANSFORM MODULE FUNCTIONS ~~~~~~~~

def get_transform_module(model,input_layer, n_main_layers):
    model1 = keras.models.clone_model(model)
    X = input_layer
    for layer in model1.layers[1:-n_main_layers]:
        X = layer(X)
    transform_module = Model(inputs = input_layer, outputs = X)

    return transform_module

def tm_template_weights_to_model(transform_module, model):
    for lidx, layer in enumerate(transform_module.layers):
        tm_weights = transform_module.get_layer(index = lidx).get_weights()
        model.get_layer(index = lidx).set_weights(tm_weights)
    return model

def model_weights_to_tm_template(transform_module, model):  
    transform_module1 = keras.models.clone_model(transform_module)
    for lidx, layer in enumerate(transform_module1.layers):
        m_weights = model.get_layer(index = lidx).get_weights()
        transform_module1.get_layer(index = lidx).set_weights(m_weights)
    return transform_module1



# # ~~~~~~~~ RNN CLASSIFIER FUNCTIONS ~~~~~~~~
# def get_data_cube(X, Y, window_blocks, train = True, scaler = None, magic_value = -100):
#     """
#     Create data cube for use with Keras RNN. Standardize data then pad and reshape data to have
#     [samples, timesteps, features] dimensions with an equal number of timesteps for each slice
#     I use a Masking layer in the RNN architecture to allow for sequences of different length
    
#     Args:
#         X: 2D nuumpy array with data, dimensions [features, samples]
#         window_blocks: 1D numpy array indicating block of provenance for input segment values
#         train: Boolean indicating whether the input data is training data
#         scaler: StandardScaler to transform data
#         magic_value: integer indicating value with which to pad samples
            
#     Returns:
#         X_cube: 3D numpy array of size [samples, timesteps, features]
#         scaler: 1D numpy array with class label for each segment
#     """
#     #standardize across each feature dimension
#     if train:
#         scaler = StandardScaler()
#         scaler = scaler.fit(X.T)
#         X = scaler.transform(X.T).T
#     else:
#         #for testing data, we want to use same transform as was fit to training data
#         X = scaler.transform(X.T).T

#     # common number of time steps
#     common_timesteps = np.max(np.bincount(window_blocks.astype('int')))
    
#     # get each block, pad, and stack to form a data cube
#     X_cube = []
#     Y_cube = []
#     for b_count, b_idx in enumerate(np.unique(window_blocks)):
#        #slice
#         X_slice = X[:,np.where(window_blocks==b_idx)[0]]
#         Y_slice = Y[np.where(window_blocks==b_idx)[0],:].T
#         #pad - can just use keras padding function
#         pad_size = common_timesteps-X_slice.shape[1]
#         X_slice_pad = np.pad(X_slice,pad_width=((0,0),(0,pad_size)), mode='constant', constant_values= magic_value)
#         Y_slice_pad = np.pad(Y_slice,pad_width=((0,0),(0,pad_size)), mode='constant', constant_values= 0)
#         #stack
#         if b_count == 0:
#             X_cube  = X_slice_pad
#             Y_cube = Y_slice_pad
#         else:
#             X_cube = np.dstack((X_cube,X_slice_pad))
#             Y_cube = np.dstack((Y_cube,Y_slice_pad))
#     # swap dimension to get [samples, timesteps, features]
#     X_cube = np.swapaxes(X_cube,0,2)
#     Y_cube = np.swapaxes(Y_cube,0,2)
    
#     return X_cube, Y_cube, scaler

# def many_to_many_model(input_shape, n_outputs, mask_value = -100):
#     """
#     Create simple RNN model
    
#     Args:
#         input_shape
#         n_outputs: number of output classes
#         mask_value: value indicating which timepoints to mask out
            
#     Returns:
#         model
#     """
    
#     #define model architecture
#     X_input = Input(shape = input_shape)
#     X = Masking(mask_value=mask_value)(X_input)
#     X = GRU(24, return_sequences= True, stateful = False)(X)
#     X = Dropout(0.5)(X)
#     X = TimeDistributed(Dense(n_outputs,activation = 'softmax'))(X)
#     model = Model(inputs = X_input, outputs = X)
#     return model

# def get_RNN_f1(X, Y, model, average = 'weighted', mask_value = -100):
#     """
#     Get f1 score for an RNN model using masked timepoint data

#     Args:
#         X: 3D numpy array with shape [samples, timepoints, features]
#         Y: 3D numpy array with shape [samples, timepoints, classes]. one-hot coding of classes
#         model: RNN model object
#         average: string argument for f1_score function. Usually 'macro' or 'weighted'
#         mask_value: value indicating which timepoints to mask out

#     Returns:
#         f1: f1 score
#     """
#     # Mask out indices based on mask value
#     nonmasked_idxs = np.where(X[:,:,0].flatten()!=mask_value)[0]
#     # Get target labels for non-masked timepoints
#     y_true = np.argmax(Y,2).flatten()[nonmasked_idxs]
#     # Get model predictions for non-masked timepoints
#     preds = model.predict(X)
#     y_pred = np.argmax(preds,2).flatten()[nonmasked_idxs]
#     # Get F1 score
#     f1 = f1_score(y_true,y_pred,average = average)

#     return f1

# def RNN_on_labeled_data(feature_matrix, target_labels, window_tstamps, block_labels, n_splits = 4,\
#                        verbose = 0, epochs = 40, batch_size = 2, permute = False):
#     """
#     Train and evaluate RNN model on labeled data
    
#     Args:
#         feature_matrix: 2D nuumpy array with data, dimensions [features, samples]
#         window_blocks: 1D numpy array indicating block of provenance for input segment values
#         train: Boolean indicating whether the input data is training data
#         scaler: StandardScaler to transform data
#         magic_value: integer indicating value with which to pad samples
            
#     Returns:
#         train_f1_scores: training scores for each split
#         test_f1_scores: test scores for each split


#     """
    
#     # transpose data
#     #feature_matrix = feature_matrix.T
    
#     #initialize empty array
#     train_f1_scores = np.empty((n_splits,))
#     test_f1_scores = np.empty((n_splits,))


#     #get block_ids and corresponding classes in block. there are the units over which we will do train/test split
#     blocks = np.array([k for k,g in groupby(block_labels)])
#     classes = np.array([k for k,g in groupby(target_labels) if k!=0])
    
#     #permute class labels, if indicated
#     if permute:
#         #using indexing tricks to have this work out
#         classes_perm = np.random.permutation(classes)
#         target_labels_shuffled = np.empty((0,))
#         for i,b in enumerate(blocks):
#             idxs = np.where(block_labels==b)[0]
#             target_labels_shuffled = np.hstack((target_labels_shuffled,classes_perm[i]*np.ones((idxs.size,))))
#         target_labels = target_labels_shuffled
#         classes = classes_perm
     
    
#     #stratify split to retain ratio of class labels
#     skf = StratifiedKFold(n_splits=n_splits,shuffle = True)
#     print(block_labels.shape)
#     print(target_labels.shape)
#     print(blocks.shape,classes.shape)

#     #systematically use one fold of the data as a held-out test set
#     for split_count, (blocks_train_idxs, blocks_test_idxs) in enumerate(skf.split(blocks, classes)):
#         print('Split Count: %i'% (split_count+1))

#         #get train and test indices
#         blocks_train = blocks[blocks_train_idxs]
#         blocks_test = blocks[blocks_test_idxs]
#         train_idxs =np.where(np.isin(block_labels,blocks_train))[0]
#         test_idxs =np.where(np.isin(block_labels,blocks_test))[0]

#         # select training data and pad to get an array where each sample has same number of timesteps
#         X_train = feature_matrix[:,train_idxs]
#         y_train = target_labels[train_idxs]
#         #one-hot encoding of class labels
#         y_train = to_categorical(y_train-np.min(y_train))
#         #get block labels of given samples
#         win_blocks_train = block_labels[train_idxs]

#         #get cube
#         X_train_cube, Y_train_cube, scaler = get_data_cube(X_train, y_train,win_blocks_train, train = True, magic_value = -100)
#         print(X_train_cube.shape, Y_train_cube.shape)

#         # select test data and pad to get an array where each sample has same number of timesteps
#         X_test = feature_matrix[:,test_idxs]
#         y_test = target_labels[test_idxs]
#         #one-hot encoding of class labels
#         y_test = to_categorical(y_test-np.min(y_test))
#         #get block labels of given samples
#         win_blocks_test = block_labels[test_idxs]
#         #get data cube
#         X_test_cube, Y_test_cube, scaler = get_data_cube(X_test, y_test, win_blocks_test, train = False, scaler = scaler, magic_value = -100)
#         print(X_test_cube.shape, Y_test_cube.shape)

#         n_timesteps, n_features, n_outputs = X_train_cube.shape[1], X_train_cube.shape[2], Y_test_cube.shape[2]

#         #setting timestep dimension to None 
#         model = many_to_many_model((None,n_features),n_outputs,mask_value = -100)
#         model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#         #model.summary

#         print('Training Model')
#         # fit network
#         model.fit(X_train_cube, Y_train_cube, epochs=epochs, batch_size=batch_size, verbose=verbose)

#         print('Evaluating Model')
#         #evaluate model on train and test data
#         train_f1_scores[split_count] = get_RNN_f1(X_train_cube, Y_train_cube, model)
#         test_f1_scores[split_count] = get_RNN_f1(X_test_cube, Y_test_cube, model)

#     return train_f1_scores, test_f1_scores



# def get_trained_RNN(X , Y, blocks, verbose = 0, epochs = 40, batch_size = 2, permute = False):
#     """
#     Train model

#     Args:
#       X: 2D numpy array with shape [features, samples]
#       Y: array with class label for each sample
#       blocks: 1D numpy array indicating block of provenance for input segment values
#       verbose: argument indicating verbosity for tensorflow traning and evaluating function
#       epochs: number of epochs to use for training
#       batch_size: number of samples to include in each batch
#       permute: Boolean to shuffle class labels (useful to test performance under null hypothesis) 
          
#     Returns:
#       model: trained model
#       train_f1: train F1 score
#       scaler: StandardScaler object fit to training data
#     """

#     #permute class labels, if indicated
#     if permute:
#         Y, dummy = permute_class_blocks(blocks, Y)

#     # select training data and pad to get an array where each sample has same number of timesteps
#     #one-hot encoding of class labels
#     Y = to_categorical(Y-np.min(Y))

#     #get cube
#     X_train_cube, Y_train_cube, scaler = get_data_cube(X, Y, blocks, train = True, magic_value = -100)
#     print(X_train_cube.shape, Y_train_cube.shape)

#     n_timesteps, n_features, n_outputs = X_train_cube.shape[1], X_train_cube.shape[2], Y_train_cube.shape[2]
#     #setting timestep dimension to None 
#     model = many_to_many_model((None,n_features),n_outputs,mask_value = -100)
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     #model.summary

#     print('Training Model')
#     # fit network
#     model.fit(X_train_cube, Y_train_cube, epochs=epochs, batch_size=batch_size, verbose=verbose)

#     #evaluate on training data
#     train_f1 = get_RNN_f1(X_train_cube, Y_train_cube, model)

#     return model, train_f1, scaler

# def evaluate_trained_RNN(X, Y, blocks, model, scaler, permute = False):
#     """
#     Evaluate model

#     Args:
#       X: 2D numpy array with shape [features, samples]
#       Y: array with class label for each sample
#       blocks: 1D numpy array indicating block of provenance for input segment values
#       model: trained RNN model object
#       scaler: StandardScaler object fit to training data
#       permute: Boolean to shuffle class labels (useful to test performance under null hypothesis) 
          
#     Returns:
#       test_f1: test F1 score
#     """
#     #permute class labels, if indicated
#     if permute:
#         Y, dummy = permute_class_blocks(blocks, Y)


#     #one-hot encoding of class labels
#     Y = to_categorical(Y-np.min(Y))

#     #get data cube
#     X_test_cube, Y_test_cube, scaler = get_data_cube(X, Y, blocks, train = False, scaler = scaler, magic_value = -100)
#     print(X_test_cube.shape, Y_test_cube.shape)
#     print('Evaluating Model')
#     test_f1 = get_RNN_f1(X_test_cube, Y_test_cube, model)

#     return test_f1

# def RNN_xsubject(data_folder, src_subject_id, nsubjects, nreps, lo_freq, hi_freq, win_size, step, exclude, \
#                  model_folder, verbose = 0, epochs = 40, batch_size = 2, permute = False):
#     """
#     Train RNN model on one subject's data and test on another subject's data

#     Args:
#         -General Info- 
#         data_folder: path to folder with all subjects' data
#         src_subject_id: ID for subject to use to train RNN model
#         nsubjects: number of subjects in data set
#         nreps: number of times to repeat training and evaluation of model
#         -Preprocesing Parameters-
#         lo_freq: lower bound of bandpass filter
#         hi_freq: higher bound of bandpass filter
#         win_size: length of segment over which to compute features
#         step: amount of overlap between neighboring segments
#         exclude: labels to exclude (default: 0 = unmarked; 7 = open hand - not collected for all subjects)
#         -RNN training Parameters-
#         verbose: argument indicating verbosity for tensorflow traning and evaluating function
#         epochs: number of epochs to use for training
#         batch_size: number of samples to include in each batch
#         permute: Boolean to shuffle class labels (useful to test performance under null hypothesis) 
            
#     Returns:
#         results_df: pandas dataframe with training and testing F1 scores for all subjects and repetitions 
#     """

#     # Set seed for replicability
#     np.random.seed(1)

#     results_df = []

#     subject_folder = os.path.join(data_folder,'%02d'%(src_subject_id))
#     print('Source Subject :%s'%(subject_folder))

#     # Process data and get features 
#     #get features across segments and corresponding info
#     X0, Y0, window_tstamps0, \
#     blocks0, series_labels0 = get_subject_data_for_classification(subject_folder, lo_freq, hi_freq, \
#                                                                     win_size, step)

#     #exclude indicated timepoints
#     in_samples = np.where(np.isin(Y0,exclude, invert = True))[0]
#     X0_in = X0[in_samples,:].T
#     Y0_in = Y0[in_samples]
#     blocks0_in = blocks0[in_samples]

#     #initialize empty list
#     train_f1_scores = np.empty((nreps,))

#     for rep in range(nreps):
#         print('Repetition: %d'%(rep+1))

#         # Set seed for replicability
#         np.random.seed(rep)

#         model, train_f1, scaler = get_trained_RNN(X0_in, Y0_in, blocks0_in,\
#                                           verbose = verbose, epochs = epochs, batch_size = batch_size, permute = permute)

#         #save training score to list
#         train_f1_scores[rep] = train_f1

#         #save trained model
#         model_fn = os.path.join(model_folder, 'trained_model_subject_%02d_rep_%d.h5'%(src_subject_id,rep))
#         keras.models.save_model(model, model_fn, save_format= 'h5')

#         # test on all other subjects

#         # initialize empty lists
#         test_f1_scores = np.empty((0,))
#         src_subject_list = []
#         targ_subject_list = []
#         for targ_subject_id in range(1,nsubjects+1):
#             if targ_subject_id != src_subject_id:
#                 subject_folder = os.path.join(data_folder,'%02d'%(targ_subject_id))
#                 print('Target Subject :%s'%(subject_folder))

#                 # Process data and get features 
#                 #get features across segments and corresponding info
#                 X1, Y1, window_tstamps1, \
#                 blocks1, series_labels1 = get_subject_data_for_classification(subject_folder, lo_freq, hi_freq, \
#                                                                                 win_size, step)
#                 in_samples = np.where(np.isin(Y1,exclude, invert = True))[0]
#                 X1_in = X1[in_samples,:].T
#                 Y1_in = Y1[in_samples]
#                 blocks1_in = blocks1[in_samples]

#                 test_f1 = evaluate_trained_RNN(X1_in, Y1_in, blocks1_in, model, scaler, permute=permute)
#                 #append to list
#                 test_f1_scores = np.hstack((test_f1_scores, test_f1))
#                 targ_subject_list.append(targ_subject_id)

#         #put test results in dataframe
#         results_df.append(pd.DataFrame({'F1_score':test_f1_scores,\
#                                                   'Type':['Test' for x in range(test_f1_scores.size)],\
#                                                   'Rep':[rep+1 for x in range(test_f1_scores.size)],\
#                                                 'Test_Subject':targ_subject_list}))
#     #put training results in dataframe
#     results_df.append(pd.DataFrame({'F1_score':train_f1_scores,\
#                                   'Type':['Train' for x in range(train_f1_scores.size)],\
#                                   'Rep': np.arange(nreps)+1,\
#                                   'Test_Subject':[src_subject_id for x in range(train_f1_scores.size)]}))

#     results_df = pd.concat(results_df, axis = 0).reset_index(drop = True)

#     return results_df