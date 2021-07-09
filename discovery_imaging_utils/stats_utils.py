import numpy as np
from discovery_imaging_utils import imaging_utils
import statsmodels
import h5py
import warnings
import json
from scipy import stats
from scipy.integrate import quad


def fast_regression_beta_resids(Y, X_PINV, X):

    '''Function that calculates regression

    Based on the pseudo-inverse, calculates
    beta-weights and residuals for linear
    regression problem to predict values
    of Y

    Parameters
    ----------

    Y : numpy.ndarray
        Dependent variables to be predicted.
        Shape <n> or <n,1>
    X_PINV : numpy.ndarray
        The pseudo-inverse for independent variables
        X used as predictors in linear regression.
        This is the result of np.linalg.pinv(X)
    X : numpy.ndarray

    Returns
    -------

    B : numpy.ndarray
        The beta-weights for the linear regression
    residuals : numpy.ndarray
        The residuals of the least squares model
    '''


    B = np.matmul(X_PINV, Y)
    residuals = Y - np.matmul(X, B)
    return B, residuals


def calc_matrix_lms_fast(net_mats, regressors, include_diagonals = False,
                    reinsert_mean = True, tstat_map = None, pval_map = False, net_vecs = False):

    """Function to remove unwanted effects from connectivity matrices

    This function will move the effects of specified regressors from
    the functional connectivity matrix, and either return the
    matrices after these effects have been removed, or the test-statistic
    for a particular regressor/s. The input net_mats can also supper
    net_vectors (i.e. subj by region) if net_vecs is set to true.


    Parameters
    ----------

    net_mats : numpy.ndarray
        Array with shape <n_subjs, n_regions, n_regions> of the
        data to be cleaned.
    regressors : numpy.ndarray
        Array with shape <n_subjs, n_regressors> whose content
        will be removed from the net_mats
    include_diagonals : bool, default False
        If False, diagonals will be skipped/set to 0
    reinsert_mean : bool, default True
        Whether the mean should be reinserted after
        regression
    tstat_map: list of ints, or None, default None
        If something outside of False is provided here,
        then instead of returning the cleaned net_mats
        the function will return t-stat maps associated
        with the parameter estimates for the contrast
        or contrasts of interest. [0] would return the first
        contrast's tstat map, and [0,1] would return the
        first two contrast's tstat maps
    pval_map : bool, default False
        Only used if tstat_map not None. If True, function also outputs pvals
        as second output.
    net_vecs: bool, default False
        Whether the function should expect net_vectors
        instead of matrices (i.e. shape <n_subjs, n_regions>)


    Returns
    -------

    If tstat_map is None, then returns a matrix with the same
    shape as net_mats with influence of regressors removed through
    linear regression. If tstat_map is a list of integers,
    then returns the tstat maps associated with the specified
    parameter estimates as list of numpy.ndarrays. If tstat_map
    is used, and pval_map is set to True, a second output will be
    included with p-values for the contrasts of interest.


    """

    #Check if cleaned data or tstat maps should be returned
    if type(tstat_map) == type(None):

        cleaned_net_mats = np.zeros(net_mats.shape)

    else:

        tstat_maps = []
        for i in range(len(tstat_map)):

            tstat_maps.append(np.zeros(net_mats.shape[1:]))

        #see if p-val maps should be returned
        if pval_map == True:
            pval_maps = []
            for i in range(len(tstat_map)):

                pval_maps.append(np.zeros(net_mats.shape[1:]))


    #calculate pinv for regressors
    pinv_mat = np.linalg.pinv(regressors)
    df = regressors.shape[0] - regressors.shape[1]
    c = np.linalg.inv(np.matmul(regressors.transpose(), regressors))



    #Process in case of net_mats
    if net_vecs == False:

        if net_mats.ndim != 3:

            raise NameError('Error: if net_vecs set to False, then net_mats must be shape <subjects, regions, regions>')

        else:

            for i in range(net_mats.shape[1]):
                for j in range(net_mats.shape[1]):


                    if include_diagonals == False:
                        if i == j:
                            continue

                    coefficients, residuals = fast_regression_beta_resids(net_mats[:,i,j].squeeze(), pinv_mat, regressors)

                    if type(tstat_map) == type(None):

                        cleaned_net_mats[:,i,j] = residuals

                        if reinsert_mean:

                            cleaned_net_mats[:,i,j] = cleaned_net_mats[:,i,j] + np.mean(net_mats[:,i,j])

                    else:

                        ssr = np.matmul(residuals.transpose(), residuals)
                        for iteration, contrast in enumerate(tstat_map):

                            temp_tstat = coefficients[contrast]/np.sqrt((ssr/df)*c[contrast,contrast])
                            tstat_maps[iteration][i,j] = temp_tstat


    #Process in case of net_vecs
    else:

        if net_mats.ndim != 2:

            raise NameError('Error: if net_vecs set to True, then net_mats must be shape <subjects, regions>')

        else:

            for i in range(net_mats.shape[1]):

                    coefficients, residuals = fast_regression_beta_resids(net_mats[:,i].squeeze(), pinv_mat, regressors)

                    if type(tstat_map) == type(None):

                        cleaned_net_mats[:,i] = residuals

                        if reinsert_mean:

                            cleaned_net_mats[:,i] = cleaned_net_mats[:,i] + np.mean(net_mats[:,i])

                    else:

                        ssr = np.matmul(residuals.transpose(), residuals)
                        for iteration, contrast in enumerate(tstat_map):

                            temp_tstat = coefficients[contrast]/np.sqrt((ssr/df)*c[contrast,contrast])
                            tstat_maps[iteration][i] = temp_tstat


    if type(tstat_map) != type(None):

        if pval_map == False:

            return tstat_maps

        else:

            for iteration, contrast in enumerate(tstat_map):
                pval_maps[iteration] = stats.t.sf(np.abs(tstat_maps[iteration]), df)*2

            return tstat_maps, pval_maps

    else:

        return cleaned_net_mats


def calc_matrix_lms_OLD(net_mats, regressors, include_diagonals = False,
                    reinsert_mean = True, tstat_map = None, pval_map = False, net_vecs = False):

    """Function to remove unwanted effects from connectivity matrices

    This function will move the effects of specified regressors from
    the functional connectivity matrix, and either return the
    matrices after these effects have been removed, or the test-statistic
    for a particular regressor/s. The input net_mats can also supper
    net_vectors (i.e. subj by region) if net_vecs is set to true.


    Parameters
    ----------

    net_mats : numpy.ndarray
        Array with shape <n_subjs, n_regions, n_regions> of the
        data to be cleaned.
    regressors : numpy.ndarray
        Array with shape <n_subjs, n_regressors> whose content
        will be removed from the net_mats
    include_diagonals : bool, default False
        If False, diagonals will be skipped/set to 0
    reinsert_mean : bool, default True
        Whether the mean should be reinserted after
        regression
    tstat_map: list of ints, or None, default None
        If something outside of False is provided here,
        then instead of returning the cleaned net_mats
        the function will return t-stat maps associated
        with the parameter estimates for the contrast
        or contrasts of interest. [0] would return the first
        contrast's tstat map, and [0,1] would return the
        first two contrast's tstat maps
    pval_map : bool, default False
        Only used if tstat_map not None. If True, function also outputs pvals
        as second output.
    net_vecs: bool, default False
        Whether the function should expect net_vectors
        instead of matrices (i.e. shape <n_subjs, n_regions>)


    Returns
    -------

    If tstat_map is None, then returns a matrix with the same
    shape as net_mats with influence of regressors removed through
    linear regression. If tstat_map is a list of integers,
    then returns the tstat maps associated with the specified
    parameter estimates as list of numpy.ndarrays. If tstat_map
    is used, and pval_map is set to True, a second output will be
    included with p-values for the contrasts of interest.


    """

    #Check if the regressors have full rank
    if np.linalg.matrix_rank(regressors) != regressors.shape[1]:

        raise NameError('Error: the regressors dont have full rank (i.e. at least one is redundant)')


    #Check if cleaned data or tstat maps should be returned
    if type(tstat_map) == type(None):

        cleaned_net_mats = np.zeros(net_mats.shape)

    else:

        tstat_maps = []
        for i in range(len(tstat_map)):

            tstat_maps.append(np.zeros(net_mats.shape[1:]))

        #see if p-val maps should be returned
        if pval_map == True:
            pval_maps = []
            for i in range(len(tstat_map)):

                pval_maps.append(np.zeros(net_mats.shape[1:]))


    #Process in case of net_mats
    if net_vecs == False:

        if net_mats.ndim != 3:

            raise NameError('Error: if net_vecs set to False, then net_mats must be shape <subjects, regions, regions>')

        else:

            for i in range(net_mats.shape[1]):
                for j in range(net_mats.shape[1]):


                    if include_diagonals == False:
                        if i == j:
                            continue

                    model = statsmodels.regression.linear_model.OLS(net_mats[:,i,j], exog=regressors)
                    results = model.fit()

                    if type(tstat_map) == type(None):

                        cleaned_net_mats[:,i,j] = net_mats[:,i,j] - model.predict(params=results.params, exog = regressors)

                        if reinsert_mean:

                            cleaned_net_mats[:,i,j] = cleaned_net_mats[:,i,j] + np.mean(net_mats[:,i,j])

                    else:

                        for iteration, contrast in enumerate(tstat_map):

                            tstat_maps[iteration][i,j] = results.tvalues[contrast]

                            if pval_map == True:

                                pval_maps[iteration][i,j] = results.pvalues[contrast]


    #Process in case of net_vecs
    else:

        if net_mats.ndim != 2:

            raise NameError('Error: if net_vecs set to True, then net_mats must be shape <subjects, regions>')

        else:

            for i in range(net_mats.shape[1]):

                    model = statsmodels.regression.linear_model.OLS(net_mats[:,i], exog=regressors)
                    results = model.fit()

                    if type(tstat_map) == type(None):

                        cleaned_net_mats[:,i] = net_mats[:,i] - model.predict(params=results.params, exog = regressors)

                        if reinsert_mean:

                            cleaned_net_mats[:,i] = cleaned_net_mats[:,i] + np.mean(net_mats[:,i])

                    else:

                        for iteration, contrast in enumerate(tstat_map):

                            tstat_maps[iteration][i] = results.tvalues[contrast]

                            if pval_map == True:

                                pval_maps[iteration][i] = results.pvalues[contrast]


    if type(tstat_map) != type(None):

        if pval_map == False:

            return tstat_maps

        else:

            return tstat_maps, pval_maps

    else:

        return cleaned_net_mats


def construct_contrast_matrix(dict_of_features, add_constant = True):
    '''Function to create a contrast dictionary/matrix for a glm

    This function will take a dictionary with values consisting of
    either numpy.ndarrays of numbers or lists of strings and converts
    it to a list with contrast names and contrast contents.

    This function:

    (1) Checks that all features have the same number of elements
    (2) Squeezes numpy arrays to one-dimension and throws error
    if they are more than one dimension
    (3) Takes the content of dict_of_features and converts it
    to either a new dictionary or list. With the following conventions:
    (a) When the value in a key/value pair of the dict_of_features is
    a numpy array, it will be copied directly to the output assuming it
    has the proper size.
    (b) When the value is a list of strings, the function will look for
    unique entries and make them into categorical variables. If there is
    only one unique string, then the variable will be excluded. If there
    are two, then one will be coded as 1 and the other as 0. The string
    coded as 1 will be referenced in the name of the function output. If
    there are more than two unique strings, then there will be a binary
    nominal variable made for each string.
    (c) By default a constant will be added but this can be changed by
    setting add_constant to False

    (4) Outputs a list with the first element being a list of strings that
    contain names for each of the variables and the second element being
    a numpy.ndarray with shape <n_categories, n_features>

    Parameters
    ----------

    dict_of_features : dict
        Dictionary whose keys will be used as variable names, and
        values will be used to construct contrasts
    add_constant : bool, default True
        whether or not to add a constant entry to represent the intercept


    Returns
    -------

    list_of_confounds : list
        First element is confound names, second element is
        numpy.ndarray containing confounds




    '''



    #First check that all dict values have the
    #right size/type
    num_features = []
    for key, value in dict_of_features.items():

        if type(value) == np.ndarray:

            dict_of_features[key] = np.squeeze(value)
            num_features.append(dict_of_features[key].size)

        elif type(value) == list:

            num_features.append(len(value))
            pass

        else:

            print(type(value))
            print(value)
            raise NameError('Error: all values in dict_of_features must be type numpy.ndarray or list')

        if np.unique(num_features)[0].size != 1:

            raise NameError('Error: all values in dict_of_features must have the same number of elements')

    num_features = np.unique(num_features)[0]


    #Now construct the regression dict

    new_dict = {}
    for key, temp_item in dict_of_features.items():

        if type(temp_item[0]) == str:

            unique_entries = []
            for temp_entry in temp_item:

                if temp_entry not in unique_entries:
                    unique_entries.append(temp_entry)

            if len(unique_entries) < 2:

                #Ignore if there is only one category
                pass

            elif len(unique_entries) == 2:

                temp_nominal = np.zeros(num_features)

                for i, temp_entry in enumerate(temp_item):

                    if temp_entry == unique_entries[0]:

                        temp_nominal[i] = 1

                new_dict[key + '_val_' + str(unique_entries[0])] = temp_nominal.copy()

            else:


                for temp_unique in unique_entries[1:]:
                    temp_nominal = np.zeros(num_features)

                    for i, temp_entry in enumerate(temp_item):

                        if temp_entry == temp_unique:

                            temp_nominal[i] = 1

                    new_dict[key + '_val_' + str(temp_unique)] = temp_nominal.copy()


        else:

            new_dict[key] = temp_item.copy()

    if add_constant == True:

        new_dict['constant'] = np.ones(num_features)

    keys = []
    values = []
    for temp_key, temp_value in new_dict.items():

        keys.append(temp_key)
        values.append(temp_value)

    values = np.vstack(values).transpose()

    return [keys, values]



def conn_from_hdf5s(hdf5_paths, output_path = None, method = 'pearson_r_to_z', grab_metadata = False, required_proportion_of_volumes = None):

    '''Produces connectivity matrix from hdf5 file(s) with cleaned timeseries


    Parameters
    ----------

    hdf5_paths: str or list
        Path to hdf5 file with cleaned fMRI data, or a list of file paths.
        If list of file paths was given, connectivity will be generated based
        on the average of the two sets of timeseries.

    output_path: str or None
        Defaults to None, otherwise string specifying output file to be created
        with connectivity data

    method: str
        Type of connectivity method to use. Defaults to pearson_r_to_z which is the
        pearson correlation coefficient fisher transformed to z scores

        All options
            - 'pearson_r_to_z'
            - 'pearson_r'

    grab_metadata: bool, default False
        If true (and if output_path is specified), then this will output a json
        file (same name as output_path with .json at end) with parcel ids, plus
        fd and dvars summary stats


    Returns
    -------

    conn_mat: numpy.ndarray
        A square numpy array with functional connectivity data. If output_path is
        specified, the function will also save functional connectivity data using numpy

    '''




    hdf5_files = []
    if type(hdf5_paths) == type('string'):

        hdf5_files.append(hdf5_paths)

    elif type(hdf5_paths) == type([]):

        for temp_file in hdf5_paths:

            hdf5_files.append(temp_file)

    else:

        raise NameError('Error: Input must be string or list')


    timeseries = []
    good_inds = []
    metadata_dict = {}
    metadata_dict['mean_dvars'] = []
    metadata_dict['mean_fd'] = []
    metadata_dict['num_defined_vols'] = []

    for temp_file in hdf5_files:
        with h5py.File(temp_file, 'r') as f:
            timeseries.append(f['data'][:])
            good_inds.append(f['denoising_info']['inclusion_inds'][:])


            if grab_metadata:
                try:
                    lh_parcel_names = f['ids']['lh_ids'].attrs['parcel_names']
                    rh_parcel_names = f['ids']['rh_ids'].attrs['parcel_names']
                    parcel_names = list(np.hstack((lh_parcel_names, rh_parcel_names)))
                    mean_dvars = f['fmriprep_metadata'].attrs['mean_dvars']
                    mean_fd = f['fmriprep_metadata'].attrs['mean_fd']
                    num_vols = f['denoising_info']['inclusion_inds'][:].shape[0]

                    metadata_dict['parcel_names'] = parcel_names
                    metadata_dict['mean_dvars'].append(mean_dvars)
                    metadata_dict['mean_fd'].append(mean_fd)
                    metadata_dict['num_defined_vols'].append(num_vols)

                except:
                    pass


    if method == 'pearson_r_to_z':

        conn_mats = np.zeros((timeseries[0].shape[0],timeseries[0].shape[0],len(timeseries)))
        for i, temp_timeseries in enumerate(timeseries):

            temp_conn = np.corrcoef(temp_timeseries[:,good_inds[i]])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                temp_z = np.arctanh(temp_conn)

            conn_mats[:,:,i] = temp_z

        conn_mat = np.mean(conn_mats, axis = -1)
        np.fill_diagonal(conn_mat, 0)


    elif method == 'pearson_r':

        conn_mats = np.zeros((timeseries[0].shape[0],timeseries[0].shape[0],len(timeseries)))
        for i, temp_timeseries in enumerate(timeseries):

            temp_conn = np.corrcoef(temp_timeseries[:,good_inds[i]])
            conn_mats[:,:,i] = temp_conn

        conn_mat = np.mean(conn_mats, axis = -1)
        np.fill_diagonal(conn_mat, 0)


    else:

        raise NameError('Error: Use Defined Connectivity Method')





    if type(output_path) != type(None):

        np.save(output_path, conn_mat)

        if grab_metadata == True:

            json_text = json.dumps(metadata_dict, indent = 4)
            json_out_path = output_path + '.json'
            with open(json_out_path,'w') as f:
                f.write(json_text)



    return conn_mat


def calculate_white_noise_estimate(power_spectrum, increment = .01 , required_segment_length = 5):

    '''Function to estimate the level of gaussian noise.

    This function uses a simple technique to estimate the
    baseline level of gaussian noise in a signal. Given
    a power spectrum, the function starts at the lowest
    observed power, and increments up by a factor of
    (1 + increment) until the function reaches an
    estimated level of noise that results in a number
    of consevutive power estimates (in order of the
    power spectrum) surpassing the count set by
    required_segment_length

    Note - the presets determined here happen to
    produce good results for the data I am working
    with at the moment.. check to see that this produces
    reasonable results for your data...

    Parameters
    ----------

    power_spectrum: numpy.ndarray
        power spectrum (all values must be above 0 for this technique to work)
    increment: float, default .01
        the amount (relative to baseline) to increase the baseline power
        spectrum estimate between each iteration
    required_segment_length: int, default 5
        the number of consecutive power_spectrum elements that must be
        observed to have power be below the final estimate of gaussian noise

    Returns
    -------

    current_white_noise_estimate : float
        an estimate of the power of white noise



    '''

    if np.min(power_spectrum) <= 0:
        raise NameError('Error: minimum power spectrum value must be at least zero for this function to work')

    current_white_noise_estimate = np.min(power_spectrum)
    max_segment_found = 0
    while max_segment_found < required_segment_length:
        current_white_noise_estimate = current_white_noise_estimate  * (1 + increment)
        current_segment_length = 0
        for temp_pow in power_spectrum:

            if temp_pow <= current_white_noise_estimate:
                current_segment_length += 1
            else:
                if current_segment_length > max_segment_found:
                    max_segment_found = current_segment_length
                current_segment_length = 0

    return current_white_noise_estimate


def calc_tau_mat(data):

    """Calculates Kendall's tau on a matrix

    This function calculate's Kendall's tau on a matrix.
    The reason for this function is that scipy's stats
    implementation of Kendall's tau only takes an array
    x and y as input, making the calculation of a tau
    matrix indexing the similarity between many variables
    computationally inefficient. It is possible that this
    program will take too much RAM for large matrices. The
    program has so far been tested against matrices of shape
    ~200x500.

    Parameters
    ----------
    data : numpy.ndarray
        Array with shape <n_variables, n_observations>

    Returns
    -------

    An numpy.ndarray shape <n_variables, n_variables> with
    the kendall's tau between each combination of variables

    """

    num_voxels = data.shape[0]
    num_obs = data.shape[1]

    internal_concordance_mat = np.zeros((num_voxels, int(num_obs*(num_obs - 1)/2)))
    denominator  = num_obs*(num_obs - 1)/2

    for temp_vox in range(num_voxels):

        temp_data = data[temp_vox]
        internal_concordance_mat[temp_vox,:] = diu.imaging_utils.convert_to_upper_arr((temp_data[:,None] > temp_data[None,:]))

    cordance_mat = np.zeros((num_voxels, num_voxels))
    binary_internal_con_mat = internal_concordance_mat*2 - 1
    cordance_mat = np.matmul(binary_internal_con_mat, binary_internal_con_mat.transpose())
    tau_mat = cordance_mat/denominator

    return tau_mat


def optimal_SVHT_coef(num_dimensions, num_samples):
    '''Code to find cutoff threshold for PCA

    This code implements the Matlab supplement from
    the paper "The optimal hard threshold for singular
    values is 4/sqrt(3)" by Gavish and Donoho. This
    code then generates a by-product that (when used
    with the output of PCA) allows you to determine
    how many principal components to keep when using
    PCA for denoising. This formulation assumes that
    you have some data matrix that is some low-rank
    signal + noise. To reconstruct the low-rank
    structure of the signal without noise, we then take
    the output of this function (omega) and combine it
    with the median of singular values from the data matrix
    following PCA to determine which components to preserve:

    singular_values > omega*np.median(singular_values)

    Any singular values satisfying the above condition
    can then be used to create a reconstructed version
    of the matrix.

    (it doesn't matter the order that you input the two
    variables)

    Parameters
    ----------

    num_dimensions : int
        The number of dimensions in the data array
    num_samples : int
        The number of samples

    Returns
    ------

    omega : float
        The constant to combine with SVD output for
        determining which PCs to truncate

    '''



    def optimal_SVHT_coef_sigma_unknown(B):

        # Beta = np.min(data.shape)/np.max(data.shape)

        if B < 0:
            raise NameError('Error: B must be greater than 0')
        elif B > 1:
            raise NameError('Error: B must be between 0 and 1')

        B = np.array([B])[:,None]
        w = 8*B / (B + 1 + np.sqrt(B**2 + 14 * B + 1))
        coef = np.sqrt(2 * (B + 1) + w)

        MPmedian = MedianMarcenkoPastur(B)
        omega = coef/np.sqrt(MPmedian)

        return omega[0][0]


    def MedianMarcenkoPastur(B):

        def MarPas(x, B):

            temp = incMarPas(x,B)
            return np.subtract(1, temp)

        lobnd = (1 - np.sqrt(B))**2;
        hibnd = (1 + np.sqrt(B))**2;
        change = 1;

        while change and (hibnd - lobnd > .001):
            change = 0
            x = np.linspace(lobnd,hibnd,5);
            y = np.zeros(x.shape[0])

            #needs fixing in here somewhere.......
            for i in range(x.shape[0]):
                temp = MarPas(x[i], B)
                y[i] = temp
            if np.any(y < 0.5):
                lobnd = np.max(x[np.where(y < 0.5)])
                #lobnd = np.max(x[0])
                change = 1
            if np.any(y > 0.5):
                hibnd = np.min(x[np.where(y > 0.5)])
                #hibnd = np.min(x[1])
                change = 1;

        med = (hibnd+lobnd)/2;
        return med


    def incMarPas(x0,B):

        topSpec = (1 + np.sqrt(B))**2;
        botSpec = (1 - np.sqrt(B))**2;

        #print(topSpec)
        #print(botSpec)



        def IfElse(Q,point):
            y = point;
            return y

        def MarPas(x, topSpec, botSpec):

            #print((np.multiply(topSpec-x, x-botSpec) > 0).shape)
            #print(np.sqrt(np.divide(np.multiply(topSpec-x, x-botSpec),np.multiply(B, x)/(2 * np.pi))).shape)

            temp = IfElse(np.multiply(topSpec-x, x-botSpec) > 0,
                          np.divide(np.sqrt(np.multiply(topSpec-x, x-botSpec)),B*x*2*np.pi))

            #print(temp)
            return(temp)

        #print(MarPas(x0,topSpec,botSpec))
        #print(np.multiply(topSpec-x0, x0-botSpec) > 0)
        #print(np.divide(np.sqrt(np.multiply(topSpec-x0, x0-botSpec)),B*x0*2*np.pi))

        I = quad(MarPas, x0, topSpec, args = (topSpec, botSpec))[0]

        return I

    B = np.min([num_dimensions, num_observations])/np.max([num_dimensions, num_observations])
    return optimal_SVHT_coef_sigma_unknown(B)

    def pca_denoise(data):
    	'''Function that uses PCA to denoise data

    	This function assumes that the data matrix
    	has some low-rank feature set + gaussian
    	noise. Given that, this function uses SVD
    	to decompose the matrix, then reconstructs
    	the matrix using a subset of the components
    	from SVD/PCA with the goal of retaining the
    	low-rank features without the noise.

    	See Gavish and Donoho 2014 for reference.

    	Parameters
    	----------

    	data : numpy.ndarray
    		A 2-d array with low rank data + noise

    	Returns
    	-------

    	cleaned : numpy.ndarrray
    		A denoised version of the data array with
    		the same dimensions

    	num_good_svs : int
    		The estimated rank used for reconstruction

    	'''

    	u, s, vh = scipy.linalg.svd(data, full_matrices = False)
    	y = np.diagonal(data).copy()
    	omega = optimal_SVHT_coef(data.shape[0], data.shape[1])
    	cutoff = omega * np.median(s)
    	num_good_svs = np.max(np.where(s > cutoff))
    	cleaned = u[:,:(num_good_svs + 1)] @ np.diag(s[:(num_good_svs+1)]) @ vh[:(num_good_svs+1),:]

    return cleaned, num_good_svs
