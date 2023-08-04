'''
Author: Elmo Chavez
Date:   27-Jul-2023
---------------------
All the functions required to extract features from the EEG Data based on different approaches.

'''

import pandas as pd
import numpy as np
from mne.time_frequency import tfr_multitaper, tfr_morlet
from scipy.stats import skew, kurtosis

# Common variables
metrics = ['mean','standard_deviation','variance','peak_to_peak','skewness','kurtosis']

#--------------------------
#       Stats Approach
#--------------------------

def Stats_Features(epochs):
    # Get data from Epochs
    epochs_data = epochs.get_data()
    
    # Compute features for each Channel from Epochs and Points
    ch_mean = np.mean(epochs_data,axis=(0,2))
    ch_var = np.var(epochs_data, axis=(0,2))
    ch_ptp = np.ptp(epochs_data, axis=(0,2))
    ch_skew = skew(epochs_data, axis=(0,2))
    ch_kurtosis = kurtosis(epochs_data, axis=(0,2))
    
    # Concatenate and intercalate to have each metric's channel together
    stat_features = np.column_stack((ch_mean.reshape(1, -1), 
                                     ch_var.reshape(1, -1), 
                                     ch_ptp.reshape(1, -1), 
                                     ch_skew.reshape(1, -1), 
                                     ch_kurtosis.reshape(1, -1)))
    
    return stat_features

#--------------------------
#       TFR Approach
#--------------------------

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
#--------------------------

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
    
    
