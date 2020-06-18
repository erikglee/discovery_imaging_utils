import sklearn
import os
import numpy as np
from discovery_imaging_utils import dictionary_utils
import matplotlib.pyplot as plt

def calc_triple_network_model(parcellated_timeseries, parcel_ids):
    
    salience_ids = []
    control_ids = []
    dmn_ids = []

    salience_identifier = 'SalVentAttnA'
    executive_control_identifier = 'ContA'
    dmn_identifier = 'DefaultA'

    for i, temp_label in enumerate(parcel_ids):

        if salience_identifier in temp_label:

            salience_ids.append(i)

        if executive_control_identifier in temp_label:

            control_ids.append(i)

        if dmn_identifier in temp_label:

            dmn_ids.append(i)


    salience_signal = np.mean(parcellated_timeseries[salience_ids,:],axis=0)
    control_signal = np.mean(parcellated_timeseries[control_ids,:], axis=0)
    dmn_signal = np.mean(parcellated_timeseries[dmn_ids,:], axis=0)
    
    nii_mean, nii_std, nii_corr, z_corr_1, z_corr_2 = calc_network_interaction_index(control_signal, dmn_signal, salience_signal, 0.8)
    
    return nii_mean, nii_std, nii_corr
        
    

def calc_network_interaction_index(timeseries_1, timeseries_2, timeseries_a, TR, window_length_seconds=40, slide_step_seconds=2, decay_constant=0.333):
    
    import numpy as np
    
    #This function calculates the network interaction index as defined
    #by the paper Dysregulated Brain Dynamics in a Triple-Network Saliency 
    #Model of Schizophrenia and Its Relation to Psychosis published in Biological
    #Psychiatry by Supekar et. al., which follows from Time-Resolved Resting-State
    #Brain Networks published in PNAS by Zalesky et. al. 
    
    #For the triple salience model, timeseries_1 should be a cleaned central executive
    #network time signal, timeseries_2 should be a default mode time signal, and
    #timeseries_a should be the salience signal. TR must be defined. The window length
    #defaults to 40s, and slide step defaults to 2s. The decay constant for the window
    #weights defaults to 0.333, and higher values make the weighting approach linearity.
    
    #The window length and slide step will be rounded to the nearest TR. 
    
    #The function will output a nii_mean, nii_std, and nii_corr, where (without including
    #notation for the decaying weights) the values are described as:
    
    # mean_over_sliding_windows(zcorr_i(timeseries_1, timeseries_a) - z_corr_i(timeseries_2, timeseries_a))
    # std_over_sliding_windows(zcorr_i(timeseries_1, timeseries_a) - z_corr_i(timeseries_2, timeseries_a))
    # corr_over_sliding_windows(z_corr_i(timeseries_1, timeseries_a), z_corr_i(timeseries_2, timeseries_a))
    
    
    #The function will calculate nii_mean, nii_std, and nii_corr which is the 
    
    #Calculate window length and slide step length in number of TRs
    window_length = int(window_length_seconds/TR)
    slide_step_length = int(slide_step_seconds/TR)
    
    #Calculate the number of windows and make arrays
    #to store the correlations within each window
    num_steps = int(np.floor((len(timeseries_1) - window_length)/slide_step_length))
    corr_1 = np.zeros(num_steps)
    corr_2 = np.zeros(num_steps)
    
    #Calculate the tapered weights for the sliding windows
    weights = calc_tapered_weights(window_length, decay_constant)
    
    #Calculate the pearson product moment correlation
    #for each window
    for i in range(len(corr_1)):
        
        beginning = int(i*slide_step_length)
        end = int(i*slide_step_length+window_length)
        
        corr_1[i] = calc_pearson_product_moment_correlation(timeseries_1[beginning:end], timeseries_a[beginning:end], weights)
        corr_2[i] = calc_pearson_product_moment_correlation(timeseries_2[beginning:end], timeseries_a[beginning:end], weights)
    
    #Calculate fisher transformation
    z_corr_1 = np.arctanh(corr_1)
    z_corr_2 = np.arctanh(corr_2)
    
    #Calculate mean difference, std difference, and corr across all windows
    nii_mean = np.mean(np.subtract(z_corr_1, z_corr_2))
    nii_std = np.std(np.subtract(z_corr_1, z_corr_2))
    nii_corr = np.corrcoef(z_corr_1, z_corr_2)[1,0]
    
    return nii_mean, nii_std, nii_corr, np.mean(z_corr_1), np.mean(z_corr_2)
    
        

#This function calculates tapered weights for sliding-window
#analyses as described in "Time-Resolved Resting-State Brain
#Networks" by Zalesky et. al in PNAS 2014
def calc_tapered_weights(window_length, decay_constant = 0.333):
    
    #Calculates tapered weights for sliding window connectivity
    #analyses. Uses exponentially tapered window as defined in:
    #"Time-Resolved Resting-State Brain Networks" by Zalesky et. al
    #in PNAS 2014. Window length should be in number of TRs, and decay
    #constant should be relative to window_length. Default decay_constant
    #is 0.333, which is equivelant to one third the window_length.
    #Returns an array with the weights for each timepoint.
    
    #Decay constants << 1 will be more non-linear, and decay constants
    # >> 1 will approach linearity
    
    if decay_constant < 0:
        
        raise NameError('Error: Decay Constant must be positive')
    
    decay_constant = window_length*decay_constant
    w0 = (1 - np.exp(-1/decay_constant))/(1 - np.exp(-1*window_length/decay_constant))
    window_indices = np.linspace(1, window_length, window_length, dtype=int)
    weights = np.zeros(len(window_indices))
    for i in window_indices:
        weights[i - 1] = w0*np.exp((i - len(weights))/decay_constant)
        
    return weights




#These functions implement the pearson product-moment correlation
#as described in "Time-Resolved Resting-State Brain Networks" by 
#Zalesky et. al in PNAS 2014
def calc_weighted_mean(time_signal, weights):
    
    #print('Weighted_Mean: ' + str(np.sum(np.multiply(time_signal, weights))))
    return np.sum(np.multiply(time_signal, weights))/np.sum(weights)

def calc_weighted_cov(time_signal, weights):
    
    #Added a square term that is not listed in the supplement
    #because it makes more sense and I get an imaginary number otherwise
    
    x_dif = np.subtract(time_signal, calc_weighted_mean(time_signal, weights))
    weighted_x_dif = np.multiply(weights, x_dif)
    weighted_x_dif_sqr = np.power(weighted_x_dif, 2)
    weighted_cov = weighted_x_dif_sqr/np.sum(weights)
    #print('Weighted_std: ' + str(np.sqrt(np.sum(weighted_x_dif_sqr))))
    return weighted_cov

def calc_weighted_cov(time_signal_1, time_signal_2, weights):
    
    partial_1 = time_signal_1 - calc_weighted_mean(time_signal_1, weights)
    partial_2 = time_signal_2 - calc_weighted_mean(time_signal_2, weights)
    product = np.multiply(partial_1, partial_2)
    weighted_product = np.multiply(weights, product)
    #print('Weighted_cov: ' + str(np.sum(weighted_product)))
    return np.sum(weighted_product)

def calc_pearson_product_moment_correlation(time_signal_1, time_signal_2, weights):
    
    numerator = calc_weighted_cov(time_signal_1, time_signal_2, weights)
    denominator_sqrd = calc_weighted_cov(time_signal_1, time_signal_1, weights)*calc_weighted_cov(time_signal_2, time_signal_2, weights)
    denominator = np.sqrt(denominator_sqrd)
    return numerator/denominator
