import os
import glob
import json
import _pickle as pickle
from eriks_packages import imaging_utils
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import scipy.interpolate as interp
from sklearn.decomposition import PCA
import scipy.interpolate as interp

def interpolate(timepoint_defined, signal, interp_type, TR):
    #defined_timepoints should be an array the length of the t with True at timepoints
    #that are defined and False at timepoints that are not defined. signal should also
    #be an array of length t. Timepoints at defined as False will be overwritten. This
    #script supports extrapolation at beginning/end of the time signal. As a quality control
    #for the spline interpolation, the most positive/negative values observed in the defined
    #portion of the signal are set as bounds for the interpolated signal
    
    #interpolation types supported:
    
        #(1) linear - takes closest point before/after undefined timepoint and interpolates.
        #    in end cases, uses the two points before/after
        #(2) cubic_spline - takes 5 closest time points before/after undefined timepoints
        #and applies cubic spline to undefined points. Uses defined signal to determine maximum/minimum
        #bounds for new interpolated points.
        #(3) spectral - yet to be implemented, will be based off of code from the 2014 Power
        #    paper

    timepoint_defined = np.array(timepoint_defined)
    
    true_inds = np.where(timepoint_defined == True)[0]
    false_inds = np.where(timepoint_defined == False)[0]
    

    signal_copy = np.array(signal)
    
    if interp_type == 'linear':
        
        #Still need to handle beginning/end cases
        
        for temp_timepoint in false_inds:
            
            
            #past_timepoint = true_inds[np.sort(np.where(true_inds < temp_timepoint)[0])[-1]]
            #future_timepoint = true_inds[np.sort(np.where(true_inds > temp_timepoint)[0])[0]]
            
            
            #Be sure there is at least one future timepoint and one past timepoint.
            #If there isn't, then grab either two past or two future timepoints and use those
            #for interpolation. If there aren't even two total past + future timepoints, then
            #just set the output to 0. Could also set the output to be unadjusted, but this
            #is a way to make the issue more obvious.
            temp_past_timepoint = np.sort(np.where(true_inds < temp_timepoint)[0])
            temp_future_timepoint = np.sort(np.where(true_inds > temp_timepoint)[0])
            
            #If we don't have enough data to interpolate/extrapolate
            if len(temp_past_timepoint) + len(temp_future_timepoint) < 2:
                
                signal_copy[temp_timepoint] = 0
               
            #If we do have enough data to interpolate/extrapolate
            else:
                
                if len(temp_past_timepoint) == 0:
                    past_timepoint = true_inds[temp_future_timepoint[1]]
                else:
                    past_timepoint = true_inds[temp_past_timepoint[-1]]

                if len(temp_future_timepoint) == 0:
                    future_timepoint = true_inds[temp_past_timepoint[-2]]
                else:
                    future_timepoint = true_inds[temp_future_timepoint[0]]

                #Find the appopriate past/future values
                past_value = signal_copy[int(past_timepoint)]
                future_value = signal_copy[int(future_timepoint)]

                #Use the interp1d function for interpolation
                interp_object = interp.interp1d([past_timepoint, future_timepoint], [past_value, future_value], bounds_error=False, fill_value='extrapolate')
                signal_copy[temp_timepoint] = interp_object(temp_timepoint).item(0)

        return signal_copy
            
    
    #For cubic spline interpolation, instead of taking the past/future timepoint
    #we will just take the closest 5 timepoints. If there aren't 5 timepoints, we will
    #set the output to 0
    if interp_type == 'cubic_spline':
        
        sorted_good = np.sort(signal_copy[true_inds])
        min_bound = sorted_good[0]
        max_bound = sorted_good[-1]
        
        #Continue if there are at least 5 good inds
        true_inds_needed = 5
        if len(true_inds) >= true_inds_needed:
        
            for temp_timepoint in false_inds:

                closest_inds = true_inds[np.argsort(np.absolute(true_inds - temp_timepoint))]
                closest_vals = signal_copy[closest_inds.astype(int)]
                interp_object = interp.interp1d(closest_inds, closest_vals, kind = 'cubic', bounds_error=False, fill_value='extrapolate')
                signal_copy[temp_timepoint.astype(int)] = interp_object(temp_timepoint).item(0)
                
            min_bound_exceded = np.where(signal_copy < min_bound)[0]
            if len(min_bound_exceded) > 0:
                
                signal_copy[min_bound_exceded] = min_bound
                
            max_bound_exceded = np.where(signal_copy > max_bound)[0]
            if len(max_bound_exceded) > 0:
                
                signal_copy[max_bound_exceded] = max_bound
             
        #If there aren't enough good timepoints, then set the bad timepoints = 0
        else:
        
            signal_copy[false_inds.astype(int)] = 0
                                               
                                               
        return signal_copy
    
    
    if interp_type == 'spectral':
        
        signal_copy = spectral_interpolation(timepoint_defined, signal_copy, TR)
        
        return signal_copy
    
                                               


def load_comps_dict(parc_obj, comps_dict):
    
    #Internal function to load a specific dictionary
    #file used in denoising. Supports multiple levels
    #of properties such as 'confounds.framewise_displacement'
    #and in cases where PCA reduction is used, does not include
    #n_skip_vols in PCA reduction, but pads the beginning of the
    #PCA reduction output with zeros to cover n_skip_vols.
    #
    # example_comps_dict = {'confounds.framewise_displacement' : False,
    #                       'confounds.twelve_motion_regs' : 3,
    #                       'aroma_noise_ics' : 3}
    #
    #This dictionary would form an output array <7,n_timepoints> including
    #framewise displacement, 3 PCs from twelve motion regressors, and 
    #3 PCs from the aroma noise ICs
    #
    #In cases where dim0 > 3.5*dim1 for an extracted element, swaps element dimensions
    
    if comps_dict == False:
        return False
    comps_matrix = []
        
    #Iterate through all key value pairs
    for key, value in comps_dict.items():
        
        #Load the current attribute of interest
        #if key has '.' representing multiple levels,
        #then recursively go through them to get the object
        if len(key.split('.')) == 1:
            
            temp_arr = getattr(parc_obj, key)
            
        else:
            
            levels = key.split('.')
            new_obj = getattr(parc_obj, levels[0])
            for temp_obj in levels[1:]:
                new_obj = getattr(new_obj, temp_obj)
            temp_arr = new_obj
        
        
        #If temp_arr is only 1d, at a second dimension for comparison
        if len(temp_arr.shape) == 1:
            
            temp_arr = np.reshape(temp_arr, (temp_arr.shape[0],1))
        
        #Current fix to reshape the aroma noise ICs... should
        #be addressing this at the parcel_timeseries object level though
        if temp_arr.shape[0] > 3.5*temp_arr.shape[1]:
            
            temp_arr = np.transpose(temp_arr)
        
        #If necessary, use PCA on the temp_arr
        if value != False:
            
            temp_arr = reduce_ics(temp_arr, value, parc_obj.n_skip_vols)
        
        #Either start a new array or stack to existing
        if comps_matrix == []:
        
            comps_matrix = temp_arr
                        
        else:
            
            comps_matrix = np.vstack((comps_matrix, temp_arr))
                    
    return comps_matrix
                                          

def reduce_ics(input_matrix, num_dimensions, n_skip_vols):

    #Takes input_matrix <num_original_dimensions, num_timepoints>. Returns
    #the num_dimensions top PCs from the input_matrix which are derived excluding
    #n_skip_vols, but zeros are padded to the beginning of the time series
    #in place of the n_skip_vols.
    
    
    if input_matrix.shape[0] > input_matrix.shape[1]:

        raise NameError('Error: input_matrix should have longer dim1 than dim0')
        
    if input_matrix.shape[0] <= 1:
        
        raise NameError('Error: input matrix must have multiple matrices')
        
    input_matrix_transposed = input_matrix.transpose()
    partial_input_matrix = input_matrix_transposed[n_skip_vols:,:]

    pca_temp = PCA()
    pca_temp.fit(partial_input_matrix)
    transformed_pcs = pca_temp.transform(partial_input_matrix)
    pca_time_signal = np.zeros((num_dimensions,input_matrix.shape[1]))
    pca_time_signal[:,n_skip_vols:] = transformed_pcs.transpose()[0:num_dimensions,:]

    #This section is from old iteration WITH ERROR!!!
    #good_components_inds = np.linspace(0,num_dimensions - 1, num = num_dimensions).astype(int)
    #pca_time_signal = np.zeros((num_dimensions, input_matrix.shape[1]))
    #pca_time_signal[:,n_skip_vols:] = pca_temp.components_[good_components_inds,:]
    
    return pca_time_signal


def demean_normalize(one_d_array):
    
    #Takes a 1d array and subtracts mean, and
    #divides by standard deviation
    
    temp_arr = one_d_array - np.nanmean(one_d_array)
        
    return temp_arr/np.nanstd(temp_arr)

def find_timepoints_to_scrub(parc_object, scrubbing_dictionary):
    
    #This function is an internal function for the main denoising script.
    #The purpose of this function is to return a array valued true for 
    #volumes to be included in subsequent analyses and a false for volumes
    #that need to be scrubbed.
    
    #This script will also get rid of the n_skip_vols at the beginning of the
    #scan. And these volumes don't get accounted for in Uniform.
    
    #If you don't want to scrub, just set scrubbing_dictionary equal to False, and
    #this script will only get rid of the initial volumes
    
    
    if type(scrubbing_dictionary) == type(False):
        
        if scrubbing_dictionary == False:
            
            temp_val = getattr(parc_object.confounds, 'framewise_displacement')
            good_arr = np.ones(temp_val.shape)
            good_arr[0:parc_object.n_skip_vols] = 0
            return good_arr
            
        else:
            
            raise NameError ('Error, if scrubbing dictionary is a boolean it must be False')
    
    
    if 'Uniform' in scrubbing_dictionary:
        
        amount_to_keep = scrubbing_dictionary.get('Uniform')[0]
        evaluation_metrics = scrubbing_dictionary.get('Uniform')[1]
        
        
        evaluation_array = []
        
        for temp_metric in evaluation_metrics:
                        
            if evaluation_array == []:
                
                evaluation_array = demean_normalize(getattr(parc_object.confounds, temp_metric))
                
            else:
                
                temp_val = np.absolute(demean_normalize(getattr(parc_object.confounds, temp_metric)))
                evaluation_array = np.add(evaluation_array, temp_val)
                
        num_timepoints_to_keep = int(evaluation_array.shape[0]*amount_to_keep)
        sorted_inds = np.argsort(evaluation_array)
        good_inds = sorted_inds[0:num_timepoints_to_keep]
        good_arr = np.zeros(evaluation_array.shape)
        good_arr[good_inds] = 1
        good_arr[0:parc_object.n_skip_vols] = 0
        
        return good_arr
    
    

    #If neither of the first two options were used, we will assume
    #they dictionary has appropriate key/value pairs describing scrubbing
    #criteria
    
    temp_val = getattr(parc_object.confounds, 'framewise_displacement')
    good_arr = np.ones(temp_val.shape)
    good_arr[0:parc_object.n_skip_vols] = 0
    
    #Iterate through all key/value pairs and set the good_arr
    #value for indices which the nuisance threshold is exceeded
    #equal to 0
    for temp_key, temp_thresh in scrubbing_dictionary.items():
        
        temp_values = getattr(parc_object.confounds, temp_key)
        bad_inds = np.where(temp_values > temp_thresh)[0]
        good_arr[bad_inds] = 0
        
    return good_arr
            
    
def spectral_interpolation(timepoint_defined, signal, TR):
    
    
    
    good_timepoint_inds = np.where(timepoint_defined == True)[0]
    bad_timepoint_inds = np.where(timepoint_defined == False)[0]
    num_timepoints = timepoint_defined.shape[0]
    signal_copy = signal.copy()
    
    t = float(TR)*good_timepoint_inds
    h = signal[good_timepoint_inds]
    TH = np.linspace(0,(num_timepoints - 1)*TR,num=num_timepoints)
    ofac = float(32)
    hifac = float(1)

    N = h.shape[0] #Number of timepoints
    T = np.max(t) - np.min(t) #Total observed timespan

    #Calculate sampling frequencies
    f = np.linspace(1/(T*ofac), hifac*N/(2*T), num = int(((hifac*N/(2*T))/((1/(T*ofac))) + 1)))

    #angular frequencies and constant offsets
    w = 2*np.pi*f


    t1 = np.reshape(t,((1,t.shape[0])))
    w1 = np.reshape(w,((w.shape[0],1)))

    tan_a = np.sum(np.sin(np.matmul(w1,t1*2)), axis=1)
    tan_b = np.sum(np.cos(np.matmul(w1,t1*2)), axis=1)
    tau = np.divide(np.arctan2(tan_a,tan_b),2*w)

    #Calculate the spectral power sine and cosine terms
    cterm = np.cos(np.matmul(w1,t1) - np.asarray([np.multiply(w,tau)]*t.shape[0]).transpose())
    sterm = np.sin(np.matmul(w1,t1) - np.asarray([np.multiply(w,tau)]*t.shape[0]).transpose())

    D = np.reshape(h,(1,h.shape[0]) )#This already has the correct shape

    ##C_final = (sum(Cmult,2).^2)./sum(Cterm.^2,2)
    #This calculation is done speerately for the numerator, denominator, and the division
    Cmult = np.multiply(cterm, D)
    numerator = np.sum(Cmult,axis=1)

    denominator = np.sum(np.power(cterm,2),axis=1)
    c = np.divide(numerator, denominator)

    #Repeat the above for sine term
    Smult = np.multiply(sterm,D)
    numerator = np.sum(Smult, axis=1)
    denominator = np.sum(np.power(sterm,2),axis=1)
    s = np.divide(numerator,denominator)

    #The inverse function to re-construct the original time series
    Time = TH
    T_rep = np.asarray([Time]*w.shape[0])
    #already have w defined
    prod = np.multiply(T_rep, w1)
    sin_t = np.sin(prod)
    cos_t = np.cos(prod)
    sw_p = np.multiply(sin_t,np.reshape(s,(s.shape[0],1)))
    cw_p = np.multiply(cos_t,np.reshape(c,(c.shape[0],1)))
    S = np.sum(sw_p,axis=0)
    C = np.sum(cw_p,axis=0)
    H = C + S

    #Normalize the reconstructed spectrum, needed when ofac > 1
    Std_H = np.std(H)
    Std_h = np.std(h)
    norm_fac = np.divide(Std_H,Std_h)
    H = np.divide(H,norm_fac)

    signal_copy[bad_timepoint_inds] = H[bad_timepoint_inds]
    
    return signal_copy
    
    

def flexible_denoise_parc(parc_obj, hpf_before_regression, scrub_criteria_dictionary, interpolation_method, noise_comps_dict, clean_comps_dict, high_pass, low_pass):
    

    #Function inputs:
    
    #parc_object = a parcellated timeseries object generated from 
    #file "imaging_utility_classes.py" which will contain both an
    #uncleaned parcellated time series, and other nuisance variables
    # etc. of interest
    
    #hpf_before_regression = the cutoff frequency for an optional high
    #pass filter that can be applied to the nuisance regressors (noise/clean) and the
    #uncleaned time signal before any regression or scrubbing occurs. Recommended
    #value would be 0.01 or False (False for if you want to skip this step)
    
    #scrub_criteria_dictionary = a dictionary that describes how scrubbing should be
    #implemented. Three main options are (1) instead of inputting a dictionary, setting this
    #variable to False, which will skip scrubbing, (2) {'Uniform' : [AMOUNT_TO_KEEP, ['std_dvars', 'framewise_displacement']]},
    #which will automatically only keep the best timepoints (for if you want all subjects to be scrubbed an equivelant amount).
    #This option will keep every timepoint if AMOUNT_TO_KEEP was 1, and no timepoints if it was 0. The list of confounds following
    #AMOUNT_TO_KEEP must at least contain one metric (but can be as many as you want) from parc_object.confounds. If more than one
    #metric is given, they will be z-transformed and their sum will be used to determine which timepoints should be
    #kept, with larger values being interpreted as noiser (WHICH MEANS THIS OPTION SHOULD ONLY BE USED WITH METRICS WHERE
    #ZERO OR NEGATIVE BASED VALUES ARE FINE AND LARGE POSITIVE VALUES ARE BAD) - this option could potentially produce
    #slightly different numbers of timepoints accross subjects still if the bad timepoints overlap to varying degrees with
    #the number of timepoints that are dropped at the beginning of the scan. (3) {'std_dvars' : 1.2, 'framewise_displacement' : 0.5} - 
    #similar to the "Uniform" option, the input metrics should be found in parc_object.confounds. Here only timepoints
    #with values below all specified thresholds will be kept for further analyses
    
    
    #interpolation_method: options are 'linear', 'cubic_spline' and (IN FUTURE) 'spectral'.
    #While scrubbed values are not included to determine any of the weights in the denoising
    #model, they will still be interpolated over and then "denoised" (have nuisance variance
    #removed) so that we have values to put into the optional filter at the end of processing.
    #The interpolated values only have any influence on the filtering proceedure, and will be again
    #removed from the time signal after filtering and thus not included in the final output. Interpolation
    #methods will do weird things if there aren't many timepoints after scrubbing. All interpolation
    #schemes besides spectral are essentially wrappers over scipy's 1d interpolation methods. 'spectral'
    #interpolation is implemented based on code from Anish Mitra/Jonathan Power
    #as shown in Power's 2014 NeuroImage paper
    
    
    #noise_comps_dict and clean_comps_dict both have the same syntax. The values
    #specified by both of these matrices will be used (along with constant and linear trend)
    #to construct the denoising regression model for the input timeseries, but only the
    #noise explained by the noise_comps_dict will be removed from the input timeseries (
    #plus also the constant and linear trend). Unlike the scrub_criteria_dictionary, the 
    #data specifed here do not need to come from the confounds section of the parc_object,
    #and because of this, if you want to include something found under parc_object.confounds,
    #you will need to specify "confounds" in the name. An example of the dictionary can be seen below:
    #
    #    clean_comps_dict = {'aroma_clean_ics' : False}
    #
    #
    #    noise_comps_dict = {'aroma_noise_ics' : 5,
    #                       'confounds.wmcsfgsr' : False
    #                       'confounds.twelve_motion_regs' : False
    #                        }
    #
    #
    #The dictionary key should specify an element to be included in the denoising process
    #and the dictionary value should be False if you don't want to do a PCA reduction on
    #the set of nuisance variables (this will be the case more often than not), alternatively
    #if the key represents a grouping of confounds, then you can use the value to specify the
    #number of principal components to kept from a reduction of the grouping. If hpf_before_regression
    #is used, the filtering will happen after the PCA.
    #
    #
    #
    
    
    #high_pass, low_pass: Filters to be applied as the last step in processing.
    #set as False if you don't want to use them, otherwise set equal to the 
    #cutoff frequency
    
    #
    #If any of the input parameters are set to True, they will be treated as if they were
    #set to False, because True values wouldn't mean anything....
    #
    #
    #
    #################################################################################################
    #################################################################################################
    #################################################################################################
    #################################################################################################
    #################################################################################################
    #################################################################################################
    
    #Create an array with 1s for timepoints to use, and 0s for scrubbed timepointsx
    good_timepoints = find_timepoints_to_scrub(parc_obj, scrub_criteria_dictionary)
    
    #Load the arrays with the data for both the clean and noise components to be used in regression
    clean_comps_pre_filter = load_comps_dict(parc_obj, clean_comps_dict)
    noise_comps_pre_filter = load_comps_dict(parc_obj, noise_comps_dict)
    
    #Apply an initial HPF to everything if necessary - this does not remove scrubbed timepoints,
    #but does skips the first n_skip_vols (which will be set to 0 and not used in subsequent steps)
    if hpf_before_regression != False:
        
        b, a = imaging_utils.construct_filter('highpass', [hpf_before_regression], parc_obj.TR, 6)
        
        #start with the clean comps matrix
        if type(clean_comps_pre_filter) != type(False):
                        
            clean_comps_post_filter = np.zeros(clean_comps_pre_filter.shape)
            for clean_dim in range(clean_comps_pre_filter.shape[0]):
                
                clean_comps_post_filter[clean_dim, parc_obj.n_skip_vols:] = imaging_utils.apply_filter(b, a, clean_comps_pre_filter[clean_dim, parc_obj.n_skip_vols:])
        
        #this option for both clean/noise indicates there is no input matrix to filter
        else:
            
            clean_comps_post_filter = False
         
        #Move to the noise comps matrix
        if type(noise_comps_pre_filter) != type(False):
            
            noise_comps_post_filter = np.zeros(noise_comps_pre_filter.shape)
            for noise_dim in range(noise_comps_pre_filter.shape[0]):
                                
                noise_comps_post_filter[noise_dim, parc_obj.n_skip_vols:] = imaging_utils.apply_filter(b, a, noise_comps_pre_filter[noise_dim, parc_obj.n_skip_vols:])
        
        else:
            
            noise_comps_post_filter = False
         
        #then filter the original time signal
        filtered_time_series = np.zeros(parc_obj.time_series.shape)
        for original_ts_dim in range(parc_obj.time_series.shape[0]):
            
            filtered_time_series[original_ts_dim, parc_obj.n_skip_vols:] = imaging_utils.apply_filter(b, a, parc_obj.time_series[original_ts_dim, parc_obj.n_skip_vols:])
    
    #If you don't want to apply the initial HPF, then
    #just make a copy of the matrices of interest
    else:
        
        clean_comps_post_filter = clean_comps_pre_filter
        noise_comps_post_filter = noise_comps_pre_filter
        filtered_time_series = parc_obj.time_series
        
    
    
    
    #Now create the nuisance regression model. Only do this step if
    #the noise_comps_post_filter isn't false.
    good_timepoint_inds = np.where(good_timepoints == True)[0]
    bad_timepoint_inds = np.where(good_timepoints == False)[0]
    if type(noise_comps_post_filter) == type(False):
        
        regressed_time_signal = filtered_time_series
        
    else:
        
        
        #Weird thing where I need to swap dimensions here...(implemented correctly)      
        
        #First add constant/linear trend to the denoising model
        constant = np.ones((1,filtered_time_series.shape[1]))
        linear_trend = np.linspace(0,filtered_time_series.shape[1],num=filtered_time_series.shape[1])
        linear_trend = np.reshape(linear_trend, (1,filtered_time_series.shape[1]))[0]
        noise_comps_post_filter = np.vstack((constant, linear_trend, noise_comps_post_filter))
        
        regressed_time_signal = np.zeros(filtered_time_series.shape).transpose()
        filtered_time_series_T = filtered_time_series.transpose()
        
        #If there aren't any clean components,
        #do a "hard" or "agressive" denosing
        if type(clean_comps_post_filter) == type(False):
                   
            noise_comps_post_filter_T_to_be_used = noise_comps_post_filter[:,good_timepoint_inds].transpose()
            XT_X_Neg1_XT = imaging_utils.calculate_XT_X_Neg1_XT(noise_comps_post_filter_T_to_be_used)
            for temp_time_signal_dim in range(filtered_time_series.shape[0]):
                regressed_time_signal[good_timepoint_inds,temp_time_signal_dim] = imaging_utils.partial_clean_fast(filtered_time_series_T[good_timepoint_inds,temp_time_signal_dim], XT_X_Neg1_XT, noise_comps_post_filter_T_to_be_used)

                        
        
        #If there are clean components, then
        #do a "soft" denoising
        else:
    
            full_matrix_to_be_used = np.vstack((noise_comps_post_filter, clean_comps_post_filter))[:,good_timepoint_inds].transpose()
            noise_comps_post_filter_T_to_be_used = noise_comps_post_filter[:,good_timepoint_inds].transpose()
            XT_X_Neg1_XT = imaging_utils.calculate_XT_X_Neg1_XT(full_matrix_to_be_used)
                        
            for temp_time_signal_dim in range(filtered_time_series.shape[0]):
                regressed_time_signal[good_timepoint_inds,temp_time_signal_dim] = imaging_utils.partial_clean_fast(filtered_time_series_T[good_timepoint_inds,temp_time_signal_dim], XT_X_Neg1_XT, noise_comps_post_filter_T_to_be_used)
                                
        
        #Put back into original dimensions
        regressed_time_signal = regressed_time_signal.transpose()
        
        
    #Now apply interpolation
    interpolated_time_signal = np.zeros(regressed_time_signal.shape)
    
    if interpolation_method == 'spectral':
        
        interpolated_time_signal = spectral_interpolation_fast(good_timepoints, regressed_time_signal, parc_obj.TR)
        
    else:
        for dim in range(regressed_time_signal.shape[0]):
            interpolated_time_signal[dim,:] = interpolate(good_timepoints, regressed_time_signal[dim,:], interpolation_method, parc_obj.TR)

    #Now if necessary, apply additional filterign:
    if high_pass == False and low_pass == False:

        filtered_time_signal = interpolated_time_signal

    else:

        if high_pass != False and low_pass == False:

            b, a = imaging_utils.construct_filter('highpass', [high_pass], parc_obj.TR, 6)

        elif high_pass == False and low_pass != False:

            b, a = imaging_utils.construct_filter('lowpass', [low_pass], parc_obj.TR, 6)

        elif high_pass != False and low_pass != False:

            b, a = imaging_utils.construct_filter('bandpass', [high_pass, low_pass], parc_obj.TR, 6)

        filtered_time_signal = np.zeros(regressed_time_signal.shape)
        for dim in range(regressed_time_signal.shape[0]):

            filtered_time_signal[dim,:] = imaging_utils.apply_filter(b,a,regressed_time_signal[dim,:])
                
        
    #Now set all the undefined timepoints to Nan
    cleaned_time_signal = filtered_time_signal
    cleaned_time_signal[:,bad_timepoint_inds] = np.nan
    
    return cleaned_time_signal, good_timepoint_inds
    
    
    
def flexible_orth_denoise_parc(parc_obj, hpf_before_regression, scrub_criteria_dictionary, interpolation_method, noise_comps_dict, clean_comps_dict, high_pass, low_pass):
    
    #THIS FUNCTION IS THE SAME AS FLEXIBLE DENOISE PARC,
    #EXCEPT FOR HERE, THE REGRESSORS IDENTIFIED BY CLEAN
    #COMPS DICT ARE REGRESSED FROM THE REGRESSORS IDENTIFIED
    #BY NOISE COMPS DICT PRIOR TO THE REGRESSORS FROM NOISE COMPS
    #DICT BEING USED TO CLEAN THE TIMESERIES. THIS MEANS THE MODEL
    #TO CLEAN THE TIMESERIES WILL ONLY CONTAIN THE ORTHOGONALIZED
    #NUISANCE VARIABLES (filtering and other options will be applied
    #as per usual)

    #Function inputs:
    
    #parc_object = a parcellated timeseries object generated from 
    #file "imaging_utility_classes.py" which will contain both an
    #uncleaned parcellated time series, and other nuisance variables
    # etc. of interest
    
    #hpf_before_regression = the cutoff frequency for an optional high
    #pass filter that can be applied to the nuisance regressors (noise/clean) and the
    #uncleaned time signal before any regression or scrubbing occurs. Recommended
    #value would be 0.01 or False (False for if you want to skip this step)
    
    #scrub_criteria_dictionary = a dictionary that describes how scrubbing should be
    #implemented. Three main options are (1) instead of inputting a dictionary, setting this
    #variable to False, which will skip scrubbing, (2) {'Uniform' : [AMOUNT_TO_KEEP, ['std_dvars', 'framewise_displacement']]},
    #which will automatically only keep the best timepoints (for if you want all subjects to be scrubbed an equivelant amount).
    #This option will keep every timepoint if AMOUNT_TO_KEEP was 1, and no timepoints if it was 0. The list of confounds following
    #AMOUNT_TO_KEEP must at least contain one metric (but can be as many as you want) from parc_object.confounds. If more than one
    #metric is given, they will be z-transformed and their sum will be used to determine which timepoints should be
    #kept, with larger values being interpreted as noiser (WHICH MEANS THIS OPTION SHOULD ONLY BE USED WITH METRICS WHERE
    #ZERO OR NEGATIVE BASED VALUES ARE FINE AND LARGE POSITIVE VALUES ARE BAD) - this option could potentially produce
    #slightly different numbers of timepoints accross subjects still if the bad timepoints overlap to varying degrees with
    #the number of timepoints that are dropped at the beginning of the scan. (3) {'std_dvars' : 1.2, 'framewise_displacement' : 0.5} - 
    #similar to the "Uniform" option, the input metrics should be found in parc_object.confounds. Here only timepoints
    #with values below all specified thresholds will be kept for further analyses
    
    
    #interpolation_method: options are 'linear', 'cubic_spline' and (IN FUTURE) 'spectral'.
    #While scrubbed values are not included to determine any of the weights in the denoising
    #model, they will still be interpolated over and then "denoised" (have nuisance variance
    #removed) so that we have values to put into the optional filter at the end of processing.
    #The interpolated values only have any influence on the filtering proceedure, and will be again
    #removed from the time signal after filtering and thus not included in the final output. Interpolation
    #methods will do weird things if there aren't many timepoints after scrubbing. All interpolation
    #schemes besides spectral are essentially wrappers over scipy's 1d interpolation methods. 'spectral'
    #interpolation is implemented based on code from Anish Mitra/Jonathan Power
    #as shown in Power's 2014 NeuroImage paper
    
    
    #noise_comps_dict and clean_comps_dict both have the same syntax. The values
    #specified by both of these matrices will be used (along with constant and linear trend)
    #to construct the denoising regression model for the input timeseries, but only the
    #noise explained by the noise_comps_dict will be removed from the input timeseries (
    #plus also the constant and linear trend). Unlike the scrub_criteria_dictionary, the 
    #data specifed here do not need to come from the confounds section of the parc_object,
    #and because of this, if you want to include something found under parc_object.confounds,
    #you will need to specify "confounds" in the name. An example of the dictionary can be seen below:
    #
    #    clean_comps_dict = {'aroma_clean_ics' : False}
    #
    #
    #    noise_comps_dict = {'aroma_noise_ics' : 5,
    #                       'confounds.wmcsfgsr' : False
    #                       'confounds.twelve_motion_regs' : False
    #                        }
    #
    #
    #The dictionary key should specify an element to be included in the denoising process
    #and the dictionary value should be False if you don't want to do a PCA reduction on
    #the set of nuisance variables (this will be the case more often than not), alternatively
    #if the key represents a grouping of confounds, then you can use the value to specify the
    #number of principal components to kept from a reduction of the grouping. If hpf_before_regression
    #is used, the filtering will happen after the PCA.
    #
    #
    #
    
    
    #high_pass, low_pass: Filters to be applied as the last step in processing.
    #set as False if you don't want to use them, otherwise set equal to the 
    #cutoff frequency
    
    #
    #If any of the input parameters are set to True, they will be treated as if they were
    #set to False, because True values wouldn't mean anything....
    #
    #
    #
    #################################################################################################
    #################################################################################################
    #################################################################################################
    #################################################################################################
    #################################################################################################
    #################################################################################################
    
    #Create an array with 1s for timepoints to use, and 0s for scrubbed timepointsx
    good_timepoints = find_timepoints_to_scrub(parc_obj, scrub_criteria_dictionary)
    
    #Load the arrays with the data for both the clean and noise components to be used in regression
    clean_comps_pre_filter = load_comps_dict(parc_obj, clean_comps_dict)
    noise_comps_pre_filter = load_comps_dict(parc_obj, noise_comps_dict)
    
    #Apply an initial HPF to everything if necessary - this does not remove scrubbed timepoints,
    #but does skips the first n_skip_vols (which will be set to 0 and not used in subsequent steps)
    if hpf_before_regression != False:
        
        b, a = imaging_utils.construct_filter('highpass', [hpf_before_regression], parc_obj.TR, 6)
        
        #start with the clean comps matrix
        if type(clean_comps_pre_filter) != type(False):
                        
            clean_comps_post_filter = np.zeros(clean_comps_pre_filter.shape)
            for clean_dim in range(clean_comps_pre_filter.shape[0]):
                
                clean_comps_post_filter[clean_dim, parc_obj.n_skip_vols:] = imaging_utils.apply_filter(b, a, clean_comps_pre_filter[clean_dim, parc_obj.n_skip_vols:])
        
        #this option for both clean/noise indicates there is no input matrix to filter
        else:
            
            clean_comps_post_filter = False
         
        #Move to the noise comps matrix
        if type(noise_comps_pre_filter) != type(False):
            
            noise_comps_post_filter = np.zeros(noise_comps_pre_filter.shape)
            for noise_dim in range(noise_comps_pre_filter.shape[0]):
                                
                noise_comps_post_filter[noise_dim, parc_obj.n_skip_vols:] = imaging_utils.apply_filter(b, a, noise_comps_pre_filter[noise_dim, parc_obj.n_skip_vols:])
        
        else:
            
            noise_comps_post_filter = False
         
        #then filter the original time signal
        filtered_time_series = np.zeros(parc_obj.time_series.shape)
        for original_ts_dim in range(parc_obj.time_series.shape[0]):
            
            filtered_time_series[original_ts_dim, parc_obj.n_skip_vols:] = imaging_utils.apply_filter(b, a, parc_obj.time_series[original_ts_dim, parc_obj.n_skip_vols:])
    
    #If you don't want to apply the initial HPF, then
    #just make a copy of the matrices of interest
    else:
        
        clean_comps_post_filter = clean_comps_pre_filter
        noise_comps_post_filter = noise_comps_pre_filter
        filtered_time_series = parc_obj.time_series
        
    
    
    
    #Now create the nuisance regression model. Only do this step if
    #the noise_comps_post_filter isn't false.
    good_timepoint_inds = np.where(good_timepoints == True)[0]
    bad_timepoint_inds = np.where(good_timepoints == False)[0]
    if type(noise_comps_post_filter) == type(False):
        
        regressed_time_signal = filtered_time_series
        
    else:
        
        
        #Weird thing where I need to swap dimensions here...(implemented correctly)      
        
        #First add constant/linear trend to the denoising model
        constant = np.ones((1,filtered_time_series.shape[1]))
        linear_trend = np.linspace(0,filtered_time_series.shape[1],num=filtered_time_series.shape[1])
        linear_trend = np.reshape(linear_trend, (1,filtered_time_series.shape[1]))[0]
        noise_comps_post_filter = np.vstack((constant, linear_trend, noise_comps_post_filter))
        
        regressed_time_signal = np.zeros(filtered_time_series.shape).transpose()
        filtered_time_series_T = filtered_time_series.transpose()
        
        #If there aren't any clean components,
        #do a "hard" or "agressive" denosing
        if type(clean_comps_post_filter) == type(False):
                   
            noise_comps_post_filter_T_to_be_used = noise_comps_post_filter[:,good_timepoint_inds].transpose()
            XT_X_Neg1_XT = imaging_utils.calculate_XT_X_Neg1_XT(noise_comps_post_filter_T_to_be_used)
            for temp_time_signal_dim in range(filtered_time_series.shape[0]):
                regressed_time_signal[good_timepoint_inds,temp_time_signal_dim] = imaging_utils.partial_clean_fast(filtered_time_series_T[good_timepoint_inds,temp_time_signal_dim], XT_X_Neg1_XT, noise_comps_post_filter_T_to_be_used)

                        
        
        #If there are clean components, then
        #do a "soft" denoising...
        
        
###########################################################################
###########################################################################
####THIS CHUNK OF CODE IS THE ONLY THING TO BE CHANGED BETWEEN#############
####THE ORIGINAL FLEXIBLE DENOISE FUNC FUNCTION AND THIS ONE###############
###########################################################################
###########################################################################
        
        else:
    
            noise_comps_post_filter_T_to_be_used = noise_comps_post_filter[:,good_timepoint_inds].transpose()
            clean_comps_post_filter_T_to_be_used = clean_comps_post_filter[:,good_timepoint_inds].transpose()

            orth_noise_comps_post_filter = np.zeros(noise_comps_post_filter.shape).transpose()

            initial_XT_X_Neg1_XT = imaging_utils.calculate_XT_X_Neg1_XT(clean_comps_post_filter_T_to_be_used)
            for temp_time_signal_dim in range(orth_noise_comps_post_filter.shape[1]):
                orth_noise_comps_post_filter[good_timepoint_inds,temp_time_signal_dim] = imaging_utils.partial_clean_fast(noise_comps_post_filter_T_to_be_used[:,temp_time_signal_dim], initial_XT_X_Neg1_XT, clean_comps_post_filter_T_to_be_used)

            noise_comps_post_filter_T_to_be_used = orth_noise_comps_post_filter[good_timepoint_inds,:]
            XT_X_Neg1_XT = imaging_utils.calculate_XT_X_Neg1_XT(noise_comps_post_filter_T_to_be_used)
            for temp_time_signal_dim in range(filtered_time_series.shape[0]):
                regressed_time_signal[good_timepoint_inds,temp_time_signal_dim] = imaging_utils.partial_clean_fast(filtered_time_series_T[good_timepoint_inds,temp_time_signal_dim], XT_X_Neg1_XT, noise_comps_post_filter_T_to_be_used)
                
            #full_matrix_to_be_used = np.vstack((noise_comps_post_filter, clean_comps_post_filter))[:,good_timepoint_inds].transpose()
            #noise_comps_post_filter_T_to_be_used = noise_comps_post_filter[:,good_timepoint_inds].transpose()
            #XT_X_Neg1_XT = imaging_utils.calculate_XT_X_Neg1_XT(full_matrix_to_be_used)
                        
            #for temp_time_signal_dim in range(filtered_time_series.shape[0]):
            #    regressed_time_signal[good_timepoint_inds,temp_time_signal_dim] = imaging_utils.partial_clean_fast(filtered_time_series_T[good_timepoint_inds,temp_time_signal_dim], XT_X_Neg1_XT, noise_comps_post_filter_T_to_be_used)
                                
        
        #Put back into original dimensions
        regressed_time_signal = regressed_time_signal.transpose()
        
        
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
        
        
    #Now apply interpolation
    interpolated_time_signal = np.zeros(regressed_time_signal.shape)
    
    if interpolation_method == 'spectral':
        
        interpolated_time_signal = spectral_interpolation_fast(good_timepoints, regressed_time_signal, parc_obj.TR)
        
    else:
        for dim in range(regressed_time_signal.shape[0]):
            interpolated_time_signal[dim,:] = interpolate(good_timepoints, regressed_time_signal[dim,:], interpolation_method, parc_obj.TR)

    #Now if necessary, apply additional filterign:
    if high_pass == False and low_pass == False:

        filtered_time_signal = interpolated_time_signal

    else:

        if high_pass != False and low_pass == False:

            b, a = imaging_utils.construct_filter('highpass', [high_pass], parc_obj.TR, 6)

        elif high_pass == False and low_pass != False:

            b, a = imaging_utils.construct_filter('lowpass', [low_pass], parc_obj.TR, 6)

        elif high_pass != False and low_pass != False:

            b, a = imaging_utils.construct_filter('bandpass', [high_pass, low_pass], parc_obj.TR, 6)

        filtered_time_signal = np.zeros(regressed_time_signal.shape)
        for dim in range(regressed_time_signal.shape[0]):

            filtered_time_signal[dim,:] = imaging_utils.apply_filter(b,a,regressed_time_signal[dim,:])
                
        
    #Now set all the undefined timepoints to Nan
    cleaned_time_signal = filtered_time_signal
    cleaned_time_signal[:,bad_timepoint_inds] = np.nan
    
    return cleaned_time_signal, good_timepoint_inds
    
    
    
def spectral_interpolation_fast(timepoint_defined, signal, TR):
    
    
    good_timepoint_inds = np.where(timepoint_defined == True)[0]
    bad_timepoint_inds = np.where(timepoint_defined == False)[0]
    num_timepoints = timepoint_defined.shape[0]
    signal_copy = signal.copy()
    
    t = float(TR)*good_timepoint_inds
    h = signal[:,good_timepoint_inds]
    TH = np.linspace(0,(num_timepoints - 1)*TR,num=num_timepoints)
    ofac = float(8) #Higher than this is slow without good quality improvements
    hifac = float(1)

    N = timepoint_defined.shape[0] #Number of timepoints
    T = np.max(t) - np.min(t) #Total observed timespan

    #Calculate sampling frequencies
    f = np.linspace(1/(T*ofac), hifac*N/(2*T), num = int(((hifac*N/(2*T))/((1/(T*ofac))) + 1)))

    #angular frequencies and constant offsets
    w = 2*np.pi*f

    t1 = np.reshape(t,((1,t.shape[0])))
    w1 = np.reshape(w,((w.shape[0],1)))

    tan_a = np.sum(np.sin(np.matmul(w1,t1*2)), axis=1)
    tan_b = np.sum(np.cos(np.matmul(w1,t1*2)), axis=1)
    tau = np.divide(np.arctan2(tan_a,tan_b),2*w)

    a1 = np.matmul(w1,t1)
    b1 = np.asarray([np.multiply(w,tau)]*t.shape[0]).transpose()
    cs_input = a1 - b1

    #Calculate the spectral power sine and cosine terms
    cterm = np.cos(cs_input)
    sterm = np.sin(cs_input)
    
    cos_denominator = np.sum(np.power(cterm,2),axis=1)
    sin_denominator = np.sum(np.power(sterm,2),axis=1)
    
    #The inverse function to re-construct the original time series pt. 1
    Time = TH
    T_rep = np.asarray([Time]*w.shape[0])
    #already have w defined
    prod = np.multiply(T_rep, w1)
    sin_t = np.sin(prod)
    cos_t = np.cos(prod)
    
    for i in range(h.shape[0]):

        ##C_final = (sum(Cmult,2).^2)./sum(Cterm.^2,2)
        #This calculation is done speerately for the numerator, denominator, and the division
        Cmult = np.multiply(cterm, h[i,:])
        numerator = np.sum(Cmult,axis=1)

        c = np.divide(numerator, cos_denominator)

        #Repeat the above for sine term
        Smult = np.multiply(sterm,h[i,:])
        numerator = np.sum(Smult, axis=1)
        s = np.divide(numerator,sin_denominator)

        #The inverse function to re-construct the original time series pt. 2
        sw_p = np.multiply(sin_t,np.reshape(s,(s.shape[0],1)))
        cw_p = np.multiply(cos_t,np.reshape(c,(c.shape[0],1)))

        S = np.sum(sw_p,axis=0)
        C = np.sum(cw_p,axis=0)
        H = C + S

        #Normalize the reconstructed spectrum, needed when ofac > 1
        Std_H = np.std(H)
        Std_h = np.std(h)
        norm_fac = np.divide(Std_H,Std_h)
        H = np.divide(H,norm_fac)

        signal_copy[i,bad_timepoint_inds] = H[bad_timepoint_inds]

    
    return signal_copy


