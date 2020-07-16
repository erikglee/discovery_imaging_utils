import numpy as np
from discovery_imaging_utils.denoise.general import run_denoising


def denoise(fmriprep_out_dict, hpf_before_regression, scrub_criteria_dictionary, interpolation_method, noise_comps_dict, clean_comps_dict, high_pass, low_pass):

    """Wrapper function for imaging_utils.denoise.general.run_denoising

    Function that makes the execution of imaging_utils.denoise.general.run_denoising
    more convenient if there is an fmriprep_out_dict, containing a confounds
    dictionary, image_data_dictionary with data to be cleaned, general_info.json
    dictionary with fields including n_skip_vols and TR. The remainder of the
    arguments passed to this function will allow you to configure your denoising
    based on the fmriprep_out_dict.

    Parameters
    ----------

    fmriprep_out_dict : ...

    hpf_before_regression : ...

    scrub_criteria_dictionary : ...

    interpolation_method : ...

    noise_comps_dict : ...

    clean_comps_dict : ...

    high_pass : ...

    low_pass : ...






    Returns
    -------
    denoise_out_dict : dict
        Dictionary containing the output of denoising. This dictionary will have
        confounds, general_info, file_paths (if present), and the non-data elements
        copied from the input fmriprep_out_dict. The data field in this dictionary
        will be overwritten with the cleaned timeseries data. Beyond these fields,
        there will be fields for the settings used in denoising, mean signal
        intensities for different masks of interest, and statistics calculated
        during the denoising process.

        The main element of interest will be -
         denoise_out_dict['image_data_dictionary']['data']


    """


    time_series = fmriprep_out_dict['image_data_dictionary']['data']

    if scrub_criteria_dictionary != False:
        inds_to_include = _find_timepoints_to_scrub(fmriprep_out_dict, scrubbing_dictionary)
    else:
        inds_to_include = np.ones(time_series.shape[1], dtype=int)
        inds_to_include[0:fmriprep_out_dict['general_info.json']['n_skip_vols']] = 0

    if noise_comps_dict != False:
        noise_comps = _load_comps_dict(fmriprep_out_dict, noise_comps_dict)
    else:
        noise_comps = False

    if clean_comps_dict != False:
        clean_comps = _load_comps_dict(fmriprep_out_dict, clean_comps_dict)
    else:
        clean_comps = False


    temp_out_dict = run_denoising(time_series,
                                    hpf_before_regression,
                                    inds_to_include,
                                    interpolation_method,
                                    noise_comps,
                                    clean_comps,
                                    high_pass,
                                    low_pass,
                                    fmriprep_out_dict['general_info.json']['n_skip_vols'],
                                    fmriprep_out_dict['general_info.json']['TR'])










    mean_roi_signal_intensities = {'global_signal' : np.nanmean(fmriprep_out_dict['confounds']['global_signal']),
                                   'white_matter' : np.nanmean(fmriprep_out_dict['confounds']['white_matter']),
                                   'csf' : np.nanmean(fmriprep_out_dict['confounds']['csf'])}




    denoising_settings = {'hpf_before_regression' : hpf_before_regression,
                          'scrub_criteria_dictionary' : scrub_criteria_dictionary,
                          'interpolation_method' : interpolation_method,
                          'noise_comps_dict' : noise_comps_dict,
                          'clean_comps_dict' : clean_comps_dict,
                          'high_pass' : high_pass,
                          'low_pass' : low_pass}



    #Create the dictionary that has all the outputs
    denoise_out_dict = {}
    denoise_out_dict['confounds'] = fmriprep_out_dict['confounds']
    denoise_out_dict['image_data_dictionary'] = fmriprep_out_dict['image_data_dictionary']
    denoise_out_dict['image_data_dictionary']['data'] = temp_out_dict['cleaned_timeseries']
    denoise_out_dict['denoising_stats'] = temp_out_dict['denoising_stats'] #NEED TO IMPLEMENT STILL
    denoise_out_dict['general_info.json'] = fmriprep_out_dict['general_info.json']
    denoise_out_dict['denoising_settings.json'] = denoising_settings
    denoise_out_dict['mean_roi_signal_intensities.json'] = mean_roi_signal_intensities
    denoise_out_dict['inclusion_inds'] = inclusion_inds


    if 'file_paths.json' in fmriprep_out_dict.keys():
        denoise_out_dict['file_paths.json'] = fmriprep_out_dict['file_paths.json']










    return denoise_out_dict



def _load_comps_dict(parc_dict, comps_dict):
    """
    #Internal function, which is given a "parc_dict",
    #with different useful resting-state properties
    #(made by module parc_ts_dictionary), and accesses
    #different components specified by comp_dict, and
    #outputs them as a 2d array.

    #All variables specified must be a key in the dictionary
    #accessed by parc_dict['confounds']

    #For pre-computed groupings of variables, this function
    #supports PCA reduction of the variable grouping.

    #An example comps_dict is shown below:
    #
    # example_comps_dict = {'framewise_displacement' : False,
    #                       'twelve_motion_regs' : 3,
    #                       'aroma_noise_ics' : 3}
    #
    #This dictionary would form an output array <7,n_timepoints> including
    #framewise displacement, 3 PCs from twelve motion regressors, and
    #3 PCs from the aroma noise ICs. False specifies that no PC reduction
    #should be done on the variable, and otherwise the value in the dictionary
    #specifies the number of PCs to be reduced to.
    #
    #PCA is taken while ignoring the n_skip_vols
    #
    """

    if comps_dict == False:
        return False
    comps_matrix = []

    #Iterate through all key value pairs
    for key, value in comps_dict.items():

        #Load the current attribute of interest
        temp_arr = parc_dict['confounds'][key]

        #If temp_arr is only 1d, at a second dimension for comparison
        if len(temp_arr.shape) == 1:

            temp_arr = np.reshape(temp_arr, (temp_arr.shape[0],1))

        #If necessary, use PCA on the temp_arr
        if value != False:

            temp_arr = reduce_ics(temp_arr, value, parc_dict['general_info.json']['n_skip_vols'])

        #Either start a new array or stack to existing
        if comps_matrix == []:

            comps_matrix = temp_arr

        else:

            comps_matrix = np.vstack((comps_matrix, temp_arr))

    return comps_matrix



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


def _find_timepoints_to_scrub(parc_object, scrubbing_dictionary):
    """" Internal function used to find timepoints to scrub.

    Function that takes a parcellated dictionary object and
    another dictionary to specify the scrubbing settings, and
    uses this to find which timepoints to scrub.

    If scrubbing dictionary is set to False, then the initial timepoints
    to remove at the beginning of the scan (specified under parc_object)
    will be the only ones specified for removal. If scrubbing dictioanary
    is defined, either hard thresholds or Uniform scrubbing based on
    criteria specified under the scrubbing dictionary will be used for
    determining bad timepoints. All timepoints identified as bad (outside
    of the initial timepoints) will be padded, and because of this the output
    to the uniform scrubbing may differ by a couple of timepoints depending on
    how the overlap among bad timepoints happens to fall.

    Parameters
    ----------
    parc_object : dict
        parcellated object dictionary containing confounds class and n_skip_vols

    scrubbing_dictionary : bool or dict
        dictionary to specify scrubbing criteria (see documentation for main denoising
        script)

    Returns
    -------
    ndarray
        array with the same length as the input data, having 1s at defined timepoints and
        0s at undefined timepoints

    """


    if type(scrubbing_dictionary) == type(False):

        if scrubbing_dictionary == False:

            temp_val = parc_object['confounds']['framewise_displacement']
            good_arr = np.ones(temp_val.shape)
            good_arr[0:parc_object['general_info.json']['n_skip_vols']] = 0
            return good_arr

        else:

            raise NameError ('Error, if scrubbing dictionary is a boolean it must be False')


    if 'Uniform' in scrubbing_dictionary:

        amount_to_keep = scrubbing_dictionary.get('Uniform')[0]
        evaluation_metrics = scrubbing_dictionary.get('Uniform')[1]


        evaluation_array = []

        for temp_metric in evaluation_metrics:

            if evaluation_array == []:

                evaluation_array = demean_normalize(parc_object['confounds'][temp_metric])

            else:

                temp_val = np.absolute(demean_normalize(parc_object['confounds'][temp_metric]))
                evaluation_array = np.add(evaluation_array, temp_val)

        num_timepoints_to_keep = int(evaluation_array.shape[0]*amount_to_keep)
        sorted_inds = np.argsort(evaluation_array)
        good_inds = np.linspace(0, evaluation_array.shape[0] - 1, evaluation_array.shape[0])

        #Add padding
        for temp_ind in sorted_inds:

            if good_inds.shape[0] > num_timepoints_to_keep:

                temp_ind_pre = temp_ind - 1
                temp_ind_post = temp_ind + 1

                good_inds = good_inds[good_inds != temp_ind_pre]
                good_inds = good_inds[good_inds != temp_ind]
                good_inds = good_inds[good_inds != temp_ind_post]


        good_inds = sorted_inds[0:num_timepoints_to_keep]
        good_arr = np.zeros(evaluation_array.shape)
        good_arr[good_inds.astype(int)] = 1
        good_arr[0:parc_object['general_info.json']['n_skip_vols']] = 0

        return good_arr



    #If neither of the first two options were used, we will assume
    #they dictionary has appropriate key/value pairs describing scrubbing
    #criteria
    else:
        temp_val = parc_object['confounds']['framewise_displacement']
        good_inds = np.linspace(0, temp_val.shape[0] - 1, temp_val.shape[0])

        #Iterate through all key/value pairs and set the good_arr
        #value for indices which the nuisance threshold is exceeded
        #equal to 0
        for temp_metric, temp_thresh in scrubbing_dictionary.items():

            temp_values = parc_object['confounds'][temp_metric]
            bad_inds = np.where(temp_values > temp_thresh)[0]

            for temp_ind in bad_inds:

                temp_ind_pre = temp_ind - 1
                temp_ind_post = temp_ind + 1

                good_inds = good_inds[good_inds != temp_ind_pre]
                good_inds = good_inds[good_inds != temp_ind]
                good_inds = good_inds[good_inds != temp_ind_post]

        good_arr = np.zeros(temp_val.shape)
        good_arr[good_inds.astype(int)] = 1
        good_arr[0:parc_object['general_info.json']['n_skip_vols']] = 0

        return good_arr
