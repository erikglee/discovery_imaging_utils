import numpy as np
from discovery_imaging_utils import imaging_utils
import statsmodels


def calc_matrix_lms(net_mats, regressors, include_diagonals = False,
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


                for temp_unique in unique_entries:
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
