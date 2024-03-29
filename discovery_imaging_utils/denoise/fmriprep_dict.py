import numpy as np
from discovery_imaging_utils.denoise.general import run_denoising
from discovery_imaging_utils.dictionary_utils import general as gen_dict_utils
import shutil
import h5py
import os


def denoise_hdf5(hdf5_input_path, hdf5_output_path, hpf_before_regression, scrub_criteria_dictionary, interpolation_method, noise_comps_dict, clean_comps_dict, high_pass, low_pass, max_batch_size = 1000, filter_order = 6):

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

    batch_size : int, or None






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

    shutil.copyfile(hdf5_input_path, hdf5_output_path)


    with h5py.File(hdf5_input_path, 'r') as f:

        time_series = f['data']
        fmriprep_metadata_group = f['fmriprep_metadata']
        n_skip_vols = fmriprep_metadata_group.attrs['n_skip_vols']
        TR = fmriprep_metadata_group.attrs['TR']


        if scrub_criteria_dictionary != False:
            inds_to_include = _hdf5_find_timepoints_to_scrub(fmriprep_metadata_group, scrub_criteria_dictionary)
        else:
            inds_to_include = np.ones(time_series.shape[1], dtype=int)
            inds_to_include[0:n_skip_vols] = 0

        if noise_comps_dict != False:
            noise_comps = _hdf5_load_comps_dict(fmriprep_metadata_group, noise_comps_dict)
        else:
            noise_comps = False

        if clean_comps_dict != False:
            clean_comps = _hdf5_load_comps_dict(fmriprep_metadata_group, clean_comps_dict)
        else:
            clean_comps = False


        with h5py.File(hdf5_output_path, 'a') as nf:

            del nf['data']
            new_data = nf.create_dataset('data', time_series.shape, dtype=type(time_series[0,0]))

            num_batches = int(time_series.shape[0]/max_batch_size) + 1

            #Run the denoising in batches
            next_ind_to_clean = 0
            i = 1
            while next_ind_to_clean < time_series.shape[0]:

                if (next_ind_to_clean + max_batch_size) < time_series.shape[0]:

                    print('Running New Batch Iteration (#' + str(i) + '/' + str(num_batches) + ')')
                    temp_time_series = time_series[next_ind_to_clean:(next_ind_to_clean + max_batch_size),:]
                    temp_out_dict = run_denoising(temp_time_series,
                                                    hpf_before_regression,
                                                    inds_to_include,
                                                    interpolation_method,
                                                    noise_comps,
                                                    clean_comps,
                                                    high_pass,
                                                    low_pass,
                                                    n_skip_vols,
                                                    TR)

                    new_data[next_ind_to_clean:(next_ind_to_clean + max_batch_size),:] = temp_out_dict['cleaned_timeseries'][:,:]
                    next_ind_to_clean += max_batch_size
                    i += 1

                else:

                    print('Last Batch Iteration')
                    temp_time_series = time_series[next_ind_to_clean:,:]
                    temp_out_dict = run_denoising(temp_time_series,
                                                    hpf_before_regression,
                                                    inds_to_include,
                                                    interpolation_method,
                                                    noise_comps,
                                                    clean_comps,
                                                    high_pass,
                                                    low_pass,
                                                    n_skip_vols,
                                                    TR,
                                                    filter_order = filter_order)

                    new_data[next_ind_to_clean:,:] = temp_out_dict['cleaned_timeseries'][:,:]
                    next_ind_to_clean = time_series.shape[0]

            #Manually reformat the scrubbing dictionary if 'Uniform' scrubbing is used
            #prior to storing as metadata
            reformatted_scrubbing_dict = {}
            if type(scrub_criteria_dictionary) == dict:
                if 'Uniform' in scrub_criteria_dictionary.keys():

                    reformatted_scrubbing_dict['Uniform_Thresh'] = scrub_criteria_dictionary['Uniform'][0]
                    reformatted_scrubbing_dict['Uniform_Evaluation_Metrics'] = scrub_criteria_dictionary['Uniform'][1]
                    scrub_criteria_dictionary = reformatted_scrubbing_dict

            #Now update all the metadata
            denoising_settings = {'/denoise_settings/hpf_before_regression' : hpf_before_regression,
                                  '/denoise_settings/scrub_criteria_dictionary' : scrub_criteria_dictionary,
                                  '/denoise_settings/interpolation_method' : interpolation_method,
                                  '/denoise_settings/noise_comps_dict' : noise_comps_dict,
                                  '/denoise_settings/clean_comps_dict' : clean_comps_dict,
                                  '/denoise_settings/high_pass' : high_pass,
                                  '/denoise_settings/low_pass' : low_pass}



            denoising_settings = gen_dict_utils.flatten_dictionary(denoising_settings, flatten_char = '/')


            denoising_info = nf.create_group('denoising_info')
            gen_dict_utils._dict_to_hdf5_attrs(denoising_info, denoising_settings, base_path = '')

            mean_roi_signal_intensities = {'/mean_sig_intens/global_signal' : np.nanmean(nf['fmriprep_metadata/global_signal']),
                                           '/mean_sig_intens/white_matter' : np.nanmean(nf['fmriprep_metadata/white_matter']),
                                           '/mean_sig_intens/csf' : np.nanmean(nf['fmriprep_metadata/csf'])}

            gen_dict_utils._dict_to_hdf5_attrs(denoising_info, mean_roi_signal_intensities, base_path = '')
            denoising_info['inclusion_inds'] = np.where(inds_to_include > 0.5)[0]
            denoising_info['ratio_vols_remaining'] = len(denoising_info['inclusion_inds'])/len(inds_to_include)


            nf.flush()


        print('Ran Denoising')



    return

def denoise_hdf5_from_json(hdf5_input_path, hdf5_output_path, denoising_json_path = None, preset = None, max_batch_size = 1000):

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

    batch_size : int, or None






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
    ###########################################################################
    ###########################################################################
    if type(denoising_json_path) == type(preset):
        raise NameError('Error: EITHER denoising_json_path or preset must be defined')

    if type(preset) == type('string'):
        current_directory = '/'.join(os.path.realpath(__file__).split('/')[:-1])
        denoising_json_path = os.path.join(current_directory, 'presets', preset)

    if os.path.exists(denoising_json_path) == False:
        raise NameError('Error: check path/name of denoising settings')


    #Load denoising settings from JSON

    denoising_dict = gen_dict_utils.json_path_to_dict(denoising_json_path)
    required_fields = ['hpf_before_regression',
                       'scrub_criteria_dictionary',
                       'interpolation_method',
                       'noise_comps_dict',
                       'clean_comps_dict',
                       'high_pass',
                       'low_pass']

    for temp_field in required_fields:
        if temp_field in denoising_dict:
            continue
        else:
            raise NameError('Error: ' + temp_field + ' ' + 'must be present in denoising dictionary')

    hpf_before_regression = denoising_dict['hpf_before_regression']
    scrub_criteria_dictionary = denoising_dict['scrub_criteria_dictionary']
    interpolation_method = denoising_dict['interpolation_method']
    noise_comps_dict = denoising_dict['noise_comps_dict']
    clean_comps_dict = denoising_dict['clean_comps_dict']
    high_pass = denoising_dict['high_pass']
    low_pass = denoising_dict['low_pass']

    ############################################################################
    ############################################################################


    shutil.copyfile(hdf5_input_path, hdf5_output_path)

    with h5py.File(hdf5_input_path, 'r') as f:

        time_series = f['data']
        fmriprep_metadata_group = f['fmriprep_metadata']
        n_skip_vols = fmriprep_metadata_group.attrs['n_skip_vols']
        TR = fmriprep_metadata_group.attrs['TR']


        if scrub_criteria_dictionary != False:
            inds_to_include = _hdf5_find_timepoints_to_scrub(fmriprep_metadata_group, scrub_criteria_dictionary)
        else:
            inds_to_include = np.ones(time_series.shape[1], dtype=int)
            inds_to_include[0:n_skip_vols] = 0

        if noise_comps_dict != False:
            noise_comps = _hdf5_load_comps_dict(fmriprep_metadata_group, noise_comps_dict)
        else:
            noise_comps = False

        if clean_comps_dict != False:
            clean_comps = _hdf5_load_comps_dict(fmriprep_metadata_group, clean_comps_dict)
        else:
            clean_comps = False


        with h5py.File(hdf5_output_path, 'a') as nf:

            del nf['data']
            new_data = nf.create_dataset('data', time_series.shape, dtype=type(time_series[0,0]))

            num_batches = int(time_series.shape[0]/max_batch_size) + 1

            #Run the denoising in batches
            next_ind_to_clean = 0
            i = 1
            while next_ind_to_clean < time_series.shape[0]:

                if (next_ind_to_clean + max_batch_size) < time_series.shape[0]:

                    print('Running New Batch Iteration (#' + str(i) + '/' + str(num_batches) + ')')
                    temp_time_series = time_series[next_ind_to_clean:(next_ind_to_clean + max_batch_size),:]
                    temp_out_dict = run_denoising(temp_time_series,
                                                    hpf_before_regression,
                                                    inds_to_include,
                                                    interpolation_method,
                                                    noise_comps,
                                                    clean_comps,
                                                    high_pass,
                                                    low_pass,
                                                    n_skip_vols,
                                                    TR)

                    new_data[next_ind_to_clean:(next_ind_to_clean + max_batch_size),:] = temp_out_dict['cleaned_timeseries'][:,:]
                    next_ind_to_clean += max_batch_size
                    i += 1

                else:

                    print('Last Batch Iteration')
                    temp_time_series = time_series[next_ind_to_clean:,:]
                    temp_out_dict = run_denoising(temp_time_series,
                                                    hpf_before_regression,
                                                    inds_to_include,
                                                    interpolation_method,
                                                    noise_comps,
                                                    clean_comps,
                                                    high_pass,
                                                    low_pass,
                                                    n_skip_vols,
                                                    TR)

                    new_data[next_ind_to_clean:,:] = temp_out_dict['cleaned_timeseries'][:,:]
                    next_ind_to_clean = time_series.shape[0]


            #Now update all the metadata
            denoising_settings = {'/denoise_settings/hpf_before_regression' : hpf_before_regression,
                                  '/denoise_settings/scrub_criteria_dictionary' : scrub_criteria_dictionary,
                                  '/denoise_settings/interpolation_method' : interpolation_method,
                                  '/denoise_settings/noise_comps_dict' : noise_comps_dict,
                                  '/denoise_settings/clean_comps_dict' : clean_comps_dict,
                                  '/denoise_settings/high_pass' : high_pass,
                                  '/denoise_settings/low_pass' : low_pass}

            denoising_settings = gen_dict_utils.flatten_dictionary(denoising_settings, flatten_char = '/')


            denoising_info = nf.create_group('denoising_info')
            gen_dict_utils._dict_to_hdf5_attrs(denoising_info, denoising_settings, base_path = '')

            mean_roi_signal_intensities = {'/mean_sig_intens/global_signal' : np.nanmean(nf['fmriprep_metadata/global_signal']),
                                           '/mean_sig_intens/white_matter' : np.nanmean(nf['fmriprep_metadata/white_matter']),
                                           '/mean_sig_intens/csf' : np.nanmean(nf['fmriprep_metadata/csf'])}

            gen_dict_utils._dict_to_hdf5_attrs(denoising_info, mean_roi_signal_intensities, base_path = '')
            denoising_info['inclusion_inds'] = np.where(inds_to_include > 0.5)[0]
            denoising_info['ratio_vols_remaining'] = len(denoising_info['inclusion_inds'])/len(inds_to_include)


            nf.flush()


        print('Ran Denoising')



    return



def _load_comps_dict(fmriprep_out_dict, comps_dict):
    """
    #Internal function, which is given a "fmriprep_out_dict",
    #with different useful resting-state properties
    #(made by module parc_ts_dictionary), and accesses
    #different components specified by comp_dict, and
    #outputs them as a 2d array.

    #All variables specified must be a key in the dictionary
    #accessed by fmriprep_out_dict['confounds']

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
        temp_arr = fmriprep_out_dict['confounds'][key]

        #If temp_arr is only 1d, at a second dimension for comparison
        if len(temp_arr.shape) == 1:

            temp_arr = np.reshape(temp_arr, (temp_arr.shape[0],1))

        #If necessary, use PCA on the temp_arr
        if value != False:

            temp_arr = reduce_ics(temp_arr, value, fmriprep_out_dict['general_info.json']['n_skip_vols'])

        #Either start a new array or stack to existing
        if comps_matrix == []:

            comps_matrix = temp_arr

        else:

            comps_matrix = np.vstack((comps_matrix, temp_arr))

    return comps_matrix



def _hdf5_load_comps_dict(fmriprep_metadata_group, comps_dict):
    """
    #Internal function, which is given a "fmriprep_out_dict",
    #with different useful resting-state properties
    #(made by module parc_ts_dictionary), and accesses
    #different components specified by comp_dict, and
    #outputs them as a 2d array.

    #All variables specified must be a key in the dictionary
    #accessed by fmriprep_out_dict['confounds']

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

    if type(comps_dict) == type(False):
        return False
    comps_matrix = None

    n_skip_vols = fmriprep_metadata_group.attrs['n_skip_vols']

    #Iterate through all key value pairs
    for key, value in comps_dict.items():

        #Load the current attribute of interest
        temp_arr = fmriprep_metadata_group[key][:]

        #If necessary, use PCA on the temp_arr
        if value != False:

            temp_arr = reduce_ics(temp_arr, value, n_skip_vols)

        #Either start a new array or stack to existing
        if type(comps_matrix) == type(None):

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


def _find_timepoints_to_scrub(fmriprep_out_dict, scrubbing_dictionary):
    """" Internal function used to find timepoints to scrub.

    Function that takes a parcellated dictionary object and
    another dictionary to specify the scrubbing settings, and
    uses this to find which timepoints to scrub.

    If scrubbing dictionary is set to False, then the initial timepoints
    to remove at the beginning of the scan (specified under fmriprep_out_dict)
    will be the only ones specified for removal. If scrubbing dictioanary
    is defined, either hard thresholds or Uniform scrubbing based on
    criteria specified under the scrubbing dictionary will be used for
    determining bad timepoints. All timepoints identified as bad (outside
    of the initial timepoints) will be padded, and because of this the output
    to the uniform scrubbing may differ by a couple of timepoints depending on
    how the overlap among bad timepoints happens to fall.

    Parameters
    ----------
    fmriprep_out_dict : dict
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

            temp_val = fmriprep_out_dict['confounds']['framewise_displacement']
            good_arr = np.ones(temp_val.shape)
            good_arr[0:fmriprep_out_dict['general_info.json']['n_skip_vols']] = 0
            return good_arr

        else:

            raise NameError ('Error, if scrubbing dictionary is a boolean it must be False')


    if 'Uniform' in scrubbing_dictionary:

        amount_to_keep = scrubbing_dictionary.get('Uniform')[0]
        evaluation_metrics = scrubbing_dictionary.get('Uniform')[1]


        evaluation_array = []

        for temp_metric in evaluation_metrics:

            if evaluation_array == []:

                evaluation_array = demean_normalize(fmriprep_out_dict['confounds'][temp_metric])

            else:

                temp_val = np.absolute(demean_normalize(fmriprep_out_dict['confounds'][temp_metric]))
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
        good_arr[0:fmriprep_out_dict['general_info.json']['n_skip_vols']] = 0

        return good_arr



    #If neither of the first two options were used, we will assume
    #they dictionary has appropriate key/value pairs describing scrubbing
    #criteria
    else:
        temp_val = fmriprep_out_dict['confounds']['framewise_displacement']
        good_inds = np.linspace(0, temp_val.shape[0] - 1, temp_val.shape[0])

        #Iterate through all key/value pairs and set the good_arr
        #value for indices which the nuisance threshold is exceeded
        #equal to 0
        for temp_metric, temp_thresh in scrubbing_dictionary.items():

            temp_values = fmriprep_out_dict['confounds'][temp_metric]
            bad_inds = np.where(temp_values > temp_thresh)[0]

            for temp_ind in bad_inds:

                temp_ind_pre = temp_ind - 1
                temp_ind_post = temp_ind + 1

                good_inds = good_inds[good_inds != temp_ind_pre]
                good_inds = good_inds[good_inds != temp_ind]
                good_inds = good_inds[good_inds != temp_ind_post]

        good_arr = np.zeros(temp_val.shape)
        good_arr[good_inds.astype(int)] = 1
        good_arr[0:fmriprep_out_dict['general_info.json']['n_skip_vols']] = 0

        return good_arr


def _hdf5_find_timepoints_to_scrub(fmriprep_metadata_group, scrubbing_dictionary):
    """" Internal function used to find timepoints to scrub.

    Function that takes a parcellated dictionary object and
    another dictionary to specify the scrubbing settings, and
    uses this to find which timepoints to scrub.

    If scrubbing dictionary is set to False, then the initial timepoints
    to remove at the beginning of the scan (specified under fmriprep_out_dict)
    will be the only ones specified for removal. If scrubbing dictioanary
    is defined, either hard thresholds or Uniform scrubbing based on
    criteria specified under the scrubbing dictionary will be used for
    determining bad timepoints. All timepoints identified as bad (outside
    of the initial timepoints) will be padded, and because of this the output
    to the uniform scrubbing may differ by a couple of timepoints depending on
    how the overlap among bad timepoints happens to fall.

    Parameters
    ----------
    fmriprep_out_dict : dict
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

    n_skip_vols = fmriprep_metadata_group.attrs['n_skip_vols']

    if type(scrubbing_dictionary) == type(False):

        if scrubbing_dictionary == False:

            temp_val = fmriprep_metadata_group['framewise_displacement']
            good_arr = np.ones(temp_val.shape)
            good_arr[0:n_skip_vols] = 0
            return good_arr

        else:

            raise NameError ('Error, if scrubbing dictionary is a boolean it must be False')


    if 'Uniform' in scrubbing_dictionary:

        amount_to_keep = scrubbing_dictionary.get('Uniform')[0]
        evaluation_metrics = scrubbing_dictionary.get('Uniform')[1]


        evaluation_array = []

        for temp_metric in evaluation_metrics:

            if evaluation_array == []:

                evaluation_array = demean_normalize(fmriprep_metadata_group[temp_metric])

            else:

                temp_val = np.absolute(demean_normalize(fmriprep_metadata_group[temp_metric]))
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
        good_arr[0:n_skip_vols] = 0

        return good_arr



    #If neither of the first two options were used, we will assume
    #they dictionary has appropriate key/value pairs describing scrubbing
    #criteria
    else:
        temp_val = fmriprep_metadata_group['framewise_displacement']
        good_inds = np.linspace(0, temp_val.shape[0] - 1, temp_val.shape[0])

        #Iterate through all key/value pairs and set the good_arr
        #value for indices which the nuisance threshold is exceeded
        #equal to 0
        temp_values = np.zeros(temp_val.shape)
        for temp_metric, temp_thresh in scrubbing_dictionary.items():

            temp_values[:] = fmriprep_metadata_group[temp_metric]
            bad_inds = np.where(temp_values > temp_thresh)[0]

            for temp_ind in bad_inds:

                temp_ind_pre = temp_ind - 1
                temp_ind_post = temp_ind + 1

                good_inds = good_inds[good_inds != temp_ind_pre]
                good_inds = good_inds[good_inds != temp_ind]
                good_inds = good_inds[good_inds != temp_ind_post]

        good_arr = np.zeros(temp_val.shape)
        good_arr[good_inds.astype(int)] = 1
        good_arr[0:n_skip_vols] = 0

        return good_arr
