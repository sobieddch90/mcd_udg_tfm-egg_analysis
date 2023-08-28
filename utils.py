'''
Author: Elmo Chavez
Date:   27-Jul-2023
---------------------
All the functions required to extract features and classify AD and FTD from the EEG Data based on different approaches.

'''

import pandas as pd
import numpy as np
import mne
from mne.time_frequency import tfr_multitaper, tfr_morlet, psd_array_welch
from scipy.stats import skew, kurtosis

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectKBest, mutual_info_classif, SelectPercentile, f_classif

import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb

#--------------------------#--------------------------#
# Feature Extraction from Stats, TFR and PSD
#--------------------------#--------------------------#

# Common variables
metrics = ['mean','standard_deviation','variance','peak_to_peak','skewness','kurtosis']

#--------------------------
#       Stats Approach
def Stats_Features(epochs):
    # Get data from Epochs
    channels = epochs.ch_names
    epochs_data = epochs.get_data()
    
    # Compute features for each Channel from Epochs and Points
    ch_mean = np.mean(epochs_data, axis=(0,2))
    ch_std = np.std(epochs_data, axis=(0,2))
    ch_var = np.var(epochs_data, axis=(0,2))
    ch_ptp = np.ptp(epochs_data, axis=(0,2))
    ch_skew = skew(epochs_data, axis=(0,2))
    ch_kurtosis = kurtosis(epochs_data, axis=(0,2))
    
    # Concatenate and intercalate to have each metric's channel together
    stat_features = np.column_stack((ch_mean.reshape(1, -1), 
                                     ch_std.reshape(1,-1),
                                     ch_var.reshape(1, -1), 
                                     ch_ptp.reshape(1, -1), 
                                     ch_skew.reshape(1, -1), 
                                     ch_kurtosis.reshape(1, -1)))
    
    column_names = [f"{ch}_{metric}" for metric in metrics for ch in channels]
    dtypes = ['float'] * len(column_names)
    
    # Sort each feature by Channel
    stat_features = stat_features.astype(list(zip(column_names, dtypes)))
    
    return stat_features

#--------------------------
#       TFR Approach
def TFR_Features(epochs, freqs, n_cycles, method='multitaper', time_bandwidth=4.0):
    
    channels = epochs.ch_names

    if method=='multitaper':
        # Calculate TFR instance with MNE and Multitaper method
        tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, time_bandwidth=time_bandwidth, return_itc=False, verbose='critical')
    elif method=='morlet':
        tfr = tfr_morlet(epochs, freqs, n_cycles, return_itc=False, verbose='critical')

    # Calculate features for each channel
    features_channels = []
    features = [np.mean(tfr.data, axis=(1,2)), 
                np.std(tfr.data, axis=(1,2)), 
                np.var(tfr.data, axis=(1,2)), 
                np.ptp(tfr.data, axis=(1,2)), 
                skew(tfr.data, axis=(1,2)), 
                kurtosis(tfr.data, axis=(1,2))]
    features_channels.append(features)
    flatten_list = [item for sublist in features_channels for item in sublist]
    # Calculate features for each frequency
    tfr_features = np.array(flatten_list)
    column_names = [f"{ch}_{metric}" for metric in metrics for ch in channels]
    dtypes = ['float'] * len(column_names)
    
    tfr_features = tfr_features.astype(list(zip(column_names, dtypes)))
        
    # Get average from all the frequencies and channels
    return tfr_features

#--------------------------
#       PSD Approach
def PSD_Features(epochs, fmin=0.5, fmax=45.0, method='multitaper'):
    # Get all channels
    channels = epochs.ch_names
    # Calculate the PSD 
    spectrum  = epochs.compute_psd(method=method, picks=channels, fmin=fmin, fmax=fmax, verbose='critical')
    mean_spectrum = spectrum.average() # Calculate the average spectra across epochs
    psds = mean_spectrum.get_data(return_freqs=False)
    
    # Calculate features from the average spectra from all epochs
    features = []
    features.append(np.mean(psds, axis=1))
    features.append(np.std(psds, axis=1))
    features.append(np.var(psds, axis=1))
    features.append(np.ptp(psds, axis=1))
    features.append(skew(psds, axis=1))
    features.append(kurtosis(psds, axis=1))

    psd_features = np.concatenate(features, axis=0)
    column_names = [f"{ch}_{metric}" for metric in metrics for ch in channels]
    dtypes = ['float'] * len(column_names)
    
    # Sort each feature by Channel
    psd_features = psd_features.astype(list(zip(column_names, dtypes)))
    
    return psd_features

#--------------------------#--------------------------#
# Feature Extraction from Frequency Bands and PSD
#--------------------------#--------------------------#

# Common Variables
frequency_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}


def get_Features_PSD(raw, type='epoch', channels=None, start_time = 0, end_time = 600, duration = 60.0, overlapping = 30.0):
    # variables
    sfreq = raw.info['sfreq']
    
    # Select channels
    if channels==None: ch = raw.ch_names
    else: ch = channels        
    
    if type=='epoch':
        # Get Epochs Data
        data = mne.make_fixed_length_epochs(raw.copy().pick(ch).crop(tmin=start_time, tmax=end_time),
                                        duration=duration, overlap=overlapping, preload=True, verbose='CRITICAL').get_data()
    elif type=='raw':
        data = raw.copy().pick(ch).crop(tmin=start_time, tmax=end_time).get_data()
    
    # Compute features
    results = {}
    for band, (fmin,fmax) in frequency_bands.items():
        # Apply bandpass filter to select only the frequency band range
        data_filtered = mne.filter.filter_data(data, sfreq=sfreq, l_freq=fmin, h_freq=fmax, verbose='CRITICAL')
        # Compute the PSD using Welch's method
        psd_all, freqs_all = psd_array_welch(data, sfreq=sfreq, verbose='CRITICAL')
        psd, freqs = psd_array_welch(data_filtered, sfreq=sfreq, verbose='CRITICAL')
        
        # Compute metrics
        total_power = psd.sum()
        relative_power = total_power / psd_all.sum()
        average_power = psd.mean()
        #entropy = compute_spect_entropy(data_filtered, sfreq=sfreq)
        entropy = -np.sum(np.log(psd) * psd)
        std_dev = np.std(psd)
        peak_to_peak = np.ptp(psd)
        kurt = kurtosis(psd)
        skewness = skew(psd)
        # Store results
        results[f'{band}_total_power'] = total_power
        results[f'{band}_relative_power'] = relative_power
        results[f'{band}_average_power'] = average_power
        results[f'{band}_spectral_entropy'] = entropy
        results[f'{band}_peak_to_peak'] = peak_to_peak
        results[f'{band}_std_dev'] = std_dev
        results[f'{band}_kurtosis'] = np.mean(kurt)
        results[f'{band}_skewness'] = np.mean(skewness)
        
    return results
    
def get_Features_TFR(raw, channels=None, start_time = 0, end_time = 600, duration = 60.0, overlapping = 30.0, n_cycles=7, time_bandwidth=4.0):
    # variables
    sfreq = raw.info['sfreq']
    
    # Select channels
    if channels==None: ch = raw.ch_names
    else: ch = channels        
    
    # Get Epochs Data
    epochs = mne.make_fixed_length_epochs(raw.copy().pick(ch).crop(tmin=start_time, tmax=end_time),
                                    duration=duration, overlap=overlapping, preload=True, verbose='CRITICAL')
    
    # Compute features
    results = {}
    for band, (fmin,fmax) in frequency_bands.items():
        # Get the TFR object
        tfr = tfr_multitaper(epochs.copy().pick(ch), freqs=np.arange(fmin,fmax,1), n_cycles=n_cycles, time_bandwidth=time_bandwidth, return_itc=False, verbose='critical')
        # Compute features
        total_power = tfr.data.sum()
        average_power = tfr.data.mean()
        peak_power = tfr.data.max()
        std_dev = tfr.data.std()
        skewness = skew(tfr.data, axis=(1,2))
        kurt = kurtosis(tfr.data, axis=(1,2))
        # Store results
        results[f'{band}_total_power'] = total_power
        results[f'{band}_average_power'] = average_power
        results[f'{band}_peak_power'] = peak_power
        results[f'{band}_std'] = std_dev
        results[f'{band}_kurtosis'] = np.mean(kurt)
        results[f'{band}_skewness'] = np.mean(skewness)
    
    return results

#--------------------------#--------------------------#
# Predictions
#--------------------------#--------------------------#
def predictor(X, y, approach, frequencies, channels, features_names, selector_k, selector_perc):
    # Define the classifiers
    svm_classifier = SVC()
    random_forest_classifier = RandomForestClassifier()
    adaboost_classifier = AdaBoostClassifier()
    xgboost_classifier = xgb.XGBClassifier()
    lightgbm_classifier = lgb.LGBMClassifier()

    classifiers = [svm_classifier, random_forest_classifier, adaboost_classifier, #xgboost_classifier, 
                   lightgbm_classifier]
    classifier_names = ['SVM', 'Random Forest', 'AdaBoost', #'XGBoost', 
                        'LightGBM']

    # Create a cross-validator with 5 folds
    n_splits = 5
    test_size = 0.2

    cv_method = [KFold(n_splits=n_splits, shuffle=True, random_state=42),
                StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42),
                StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)]
    cv_method_names = ['KFold','StratifiedKFold','StratifiedShuffleSplit']

    # Create the Feature Selector method
    feat_sel_method =   [SelectKBest(score_func=f_classif, k=selector_k), SelectKBest(score_func=mutual_info_classif, k=selector_k),
                        SelectPercentile(score_func=f_classif, percentile=selector_perc), SelectPercentile(score_func=mutual_info_classif, percentile=selector_perc)]
    feat_sel_method_names = ['SelectKBest - f_classif','SelectKBest - mutual_info_classif',
                            'SelectPercentile - f_classif', 'SelectPercentile - mutual_info_classif']
    
    # Run cross-validation for each classifier and each feature selection
    results = []
    for clf, clf_name in zip(classifiers, classifier_names):
        print('Running',clf_name)
        for cv, cv_name in zip(cv_method, cv_method_names):
            for selector, sel_name in zip(feat_sel_method, feat_sel_method_names):
                scores = []
                for train_index, test_index in cv.split(X, y):
                    # Split the data into training and testing sets based on the current fold indices
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    # Feature Selection
                    X_train_selected = selector.fit_transform(X_train, y_train)
                    selected_index = selector.get_support()
                    X_test_selected = selector.transform(X_test)
                    
                    # Train the classifier on the training data
                    clf.fit(X_train_selected, y_train)
                    
                    # Make predictions on the test data
                    y_pred = clf.predict(X_test_selected)
                    
                    # Calculate accuracy for this fold and append to the scores list
                    accuracy = np.mean(y_pred == y_test)
                    scores.append(accuracy)
        
                # Store results
                result = {'Feature Extraction': approach, 
                          'Frequencies':frequencies, 
                          'Channels': channels,
                          'Metrics': features_names,
                          'Cross-Validate Method': cv_name,
                          'Feature Selection': sel_name,
                          'Metrics Selected':[features_names[i] for i, mask in enumerate(selected_index) if mask],
                          'ML Model': clf_name,
                          'Accuracy': np.mean(scores)}
                
                results.append(result)

    return results
