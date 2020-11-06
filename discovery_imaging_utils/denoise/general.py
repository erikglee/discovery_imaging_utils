import numpy as np
from discovery_imaging_utils import imaging_utils
import scipy.interpolate as interp
from sklearn.decomposition import PCA



def run_denoising(time_series, hpf_before_regression, inds_to_include, interpolation_method,
                    noise_comps, clean_comps, high_pass, low_pass, n_skip_vols, TR, inv_method = 'calculate_XT_X_Neg1_XT'):

    """Function to denoise fMRI data.

    Function to denoise fMRI data.


    Parameters
    ----------



    Returns
    -------





    """


    initial_dvars = dvars(time_series, np.linspace(0,n_skip_vols - 1,n_skip_vols,dtype=int))

    #Load the arrays with the data for both the clean and noise components to be used in regression
    clean_comps_pre_filter = clean_comps
    noise_comps_pre_filter = noise_comps

    #Apply an initial HPF to everything if necessary - this does not remove scrubbed timepoints,
    #but does skips the first n_skip_vols (which will be set to 0 and not used in subsequent steps)
    if hpf_before_regression != False:

        b, a = imaging_utils.construct_filter('highpass', [hpf_before_regression], TR, 6)

        #start with the clean comps matrix
        if type(clean_comps_pre_filter) != type(False):

            clean_comps_post_filter = np.zeros(clean_comps_pre_filter.shape)
            for clean_dim in range(clean_comps_pre_filter.shape[0]):

                clean_comps_post_filter[clean_dim, n_skip_vols:] = imaging_utils.apply_filter(b, a, clean_comps_pre_filter[clean_dim, n_skip_vols:])

        #this option for both clean/noise indicates there is no input matrix to filter
        else:

            clean_comps_post_filter = False

        #Move to the noise comps matrix
        if type(noise_comps_pre_filter) != type(False):

            noise_comps_post_filter = np.zeros(noise_comps_pre_filter.shape)
            for noise_dim in range(noise_comps_pre_filter.shape[0]):

                noise_comps_post_filter[noise_dim, n_skip_vols:] = imaging_utils.apply_filter(b, a, noise_comps_pre_filter[noise_dim, n_skip_vols:])

        else:

            noise_comps_post_filter = False

        #then filter the original time signal
        filtered_time_series = np.zeros(time_series.shape)
        for original_ts_dim in range(time_series.shape[0]):

            filtered_time_series[original_ts_dim, n_skip_vols:] = imaging_utils.apply_filter(b, a, time_series[original_ts_dim, n_skip_vols:])

    #If you don't want to apply the initial HPF, then
    #just make a copy of the matrices of interest
    else:

        clean_comps_post_filter = clean_comps_pre_filter
        noise_comps_post_filter = noise_comps_pre_filter
        filtered_time_series = time_series




    #Now create the nuisance regression model. Only do this step if
    #the noise_comps_post_filter isn't false.
    good_timepoint_inds = np.where(inds_to_include == True)[0]
    bad_timepoint_inds = np.where(inds_to_include == False)[0]

    if type(noise_comps_post_filter) == type(False):

        regressed_time_signal = filtered_time_series
        original_std = None

    else:


        #Calculate the standard deviation of the signal before nuisance regression
        original_std = np.std(filtered_time_series[:,good_timepoint_inds], axis=1)

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

            #THIS IS UNDER TESTING TO SEE WHICH PERFORMS BETTER
            if inv_method = 'calculate_XT_X_Neg1_XT':
                XT_X_Neg1_XT = imaging_utils.calculate_XT_X_Neg1_XT(noise_comps_post_filter_T_to_be_used)
            else:
                XT_X_Neg1_XT = imaging_utils.calculate_pinv(noise_comps_post_filter_T_to_be_used)


            for temp_time_signal_dim in range(filtered_time_series.shape[0]):
                regressed_time_signal[good_timepoint_inds,temp_time_signal_dim] = imaging_utils.partial_clean_fast(filtered_time_series_T[good_timepoint_inds,temp_time_signal_dim], XT_X_Neg1_XT, noise_comps_post_filter_T_to_be_used)



        #If there are clean components, then
        #do a "soft" denoising
        else:

            full_matrix_to_be_used = np.vstack((noise_comps_post_filter, clean_comps_post_filter))[:,good_timepoint_inds].transpose()
            noise_comps_post_filter_T_to_be_used = noise_comps_post_filter[:,good_timepoint_inds].transpose()

            #THIS IS UNDER TESTING TO SEE WHICH PERFORMS BETTER
            if inv_method = 'calculate_XT_X_Neg1_XT':
                XT_X_Neg1_XT = imaging_utils.calculate_XT_X_Neg1_XT(full_matrix_to_be_used)
            else:
                XT_X_Neg1_XT = imaging_utils.calculate_pinv(full_matrix_to_be_used)

            for temp_time_signal_dim in range(filtered_time_series.shape[0]):
                regressed_time_signal[good_timepoint_inds,temp_time_signal_dim] = imaging_utils.partial_clean_fast(filtered_time_series_T[good_timepoint_inds,temp_time_signal_dim], XT_X_Neg1_XT, noise_comps_post_filter_T_to_be_used)


        #Put back into original dimensions
        regressed_time_signal = regressed_time_signal.transpose()

        #Calculate the standard deviation of the signal after the nuisance regression
        post_regression_std = np.std(regressed_time_signal[:,good_timepoint_inds], axis=1)


    #Now apply interpolation
    interpolated_time_signal = np.zeros(regressed_time_signal.shape)

    if interpolation_method == 'spectral':

        interpolated_time_signal = spectral_interpolation_fast(inds_to_include, regressed_time_signal, TR)

    else:
        for dim in range(regressed_time_signal.shape[0]):
            interpolated_time_signal[dim,:] = interpolate(inds_to_include, regressed_time_signal[dim,:], interpolation_method, TR)

    #Now if necessary, apply additional filterign:
    if high_pass == False and low_pass == False:

        filtered_time_signal = interpolated_time_signal

    else:

        if high_pass != False and low_pass == False:

            b, a = imaging_utils.construct_filter('highpass', [high_pass], TR, 6)

        elif high_pass == False and low_pass != False:

            b, a = imaging_utils.construct_filter('lowpass', [low_pass], TR, 6)

        elif high_pass != False and low_pass != False:

            b, a = imaging_utils.construct_filter('bandpass', [high_pass, low_pass], TR, 6)

        filtered_time_signal = np.zeros(regressed_time_signal.shape)
        for dim in range(regressed_time_signal.shape[0]):

            filtered_time_signal[dim,:] = imaging_utils.apply_filter(b,a,regressed_time_signal[dim,:])

    final_dvars = dvars(filtered_time_signal, bad_timepoint_inds)

    #Now set all the undefined timepoints to Nan
    cleaned_time_signal = filtered_time_signal
    cleaned_time_signal[:,bad_timepoint_inds] = np.nan

    output_dict = {}
    denoising_stats = {}

    output_dict['cleaned_timeseries'] = cleaned_time_signal

    denoising_stats['dvars_pre_cleaning'] = initial_dvars
    denoising_stats['dvars_post_cleaning'] = final_dvars

    dvars_stats = {}
    dvars_stats['mean_dvars_pre_cleaning'] = np.mean(initial_dvars[(initial_dvars > 0)])
    dvars_stats['mean_dvars_post_cleaning'] = np.mean(final_dvars[(final_dvars > 0)])
    dvars_stats['max_dvars_pre_cleaning'] = np.max(initial_dvars)
    dvars_stats['max_dvars_post_cleaning'] = np.max(final_dvars)
    dvars_stats['dvars_remaining_ratio'] = np.mean(final_dvars[(final_dvars > 0)])/np.mean(initial_dvars[(initial_dvars > 0)])
    dvars_stats['def'] = 'DVARS calculated before any denoising steps (or filtering), and also after.\nBad timepoints not included in any stats.'
    denoising_stats['dvars_stats.json'] = dvars_stats


    if type(original_std) != type(None):

        output_dict['std_before_regression'] = original_std
        output_dict['std_after_regression'] = post_regression_std

    output_dict['denoising_stats'] = denoising_stats



    return output_dict



def interpolate(timepoint_defined, signal, interp_type, TR):
    """
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
        #(3) spectral based off of code from the 2014 Power
        #    paper

    """

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



def reduce_ics(input_matrix, num_dimensions, n_skip_vols):
    """
    #Takes input_matrix <num_original_dimensions, num_timepoints>. Returns
    #the num_dimensions top PCs from the input_matrix which are derived excluding
    #n_skip_vols, but zeros are padded to the beginning of the time series
    #in place of the n_skip_vols.
    """


    if input_matrix.shape[0] > input_matrix.shape[1]:

        raise NameError('Error: input_matrix should have longer dim1 than dim0')

    if input_matrix.shape[0] <= 1:

        raise NameError('Error: input matrix must have multiple matrices')

    input_matrix_transposed = input_matrix.transpose()
    partial_input_matrix = input_matrix_transposed[n_skip_vols:,:]

    pca_temp = PCA(n_components=num_dimensions)
    pca_temp.fit(partial_input_matrix)
    transformed_pcs = pca_temp.transform(partial_input_matrix)
    pca_time_signal = np.zeros((num_dimensions,input_matrix.shape[1]))
    pca_time_signal[:,n_skip_vols:] = transformed_pcs.transpose()[0:num_dimensions,:]

    return pca_time_signal

def demean_normalize(one_d_array):
    """
    #Takes a 1d array and subtracts mean, and
    #divides by standard deviation
    """

    temp_arr = one_d_array - np.nanmean(one_d_array)

    return temp_arr/np.nanstd(temp_arr)




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

def dvars(timeseries, bad_inds=None):
    ''' Function to calculate DVARS based on definition
    listed in Power's 2012 neuroimage paper. timeseries
    should have shape <regions, timepoints> and bad_inds
    is an optional list of indices that have been scrubbed.
    If bad_inds is included, then both the specified indices
    plus the points prior to the bad inds have DVARS set to
    -0.001. The output is an array with the same length as the
    input timesignal and the first element will always be
    -0.001.
    '''

    ts_deriv = np.zeros(timeseries.shape)
    for i in range(1,timeseries.shape[1]):

        ts_deriv[:,i] = timeseries[:,i] - timeseries[:,i-1]

    ts_deriv_sqr = np.power(ts_deriv, 2)
    ts_deriv_sqr_mean = np.mean(ts_deriv_sqr, axis=0)
    dvars_out = np.power(ts_deriv_sqr_mean, 0.5)

    dvars_out[0] = -0.001

    if type(bad_inds) != type(None):

        dvars_out[bad_inds] = -0.001
        bad_inds_deriv = bad_inds - 1
        bad_inds_deriv = bad_inds_deriv[(bad_inds_deriv >=0)]
        dvars_out[bad_inds_deriv] = -0.001

    return dvars_out
