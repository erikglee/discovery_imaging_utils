import numpy as np
from discovery_imaging_utils import imaging_utils


def remove_infleunces_from_connectivity_matrix(features_to_clean, covariates_to_regress):
    """"A function to remove the influence of different variables from a connectivity matrix

    Function that takes functional connectivity matrices, and a list of regressors,
    and removes regresses out the (linear) influence of the regressors from the connectivity
    matrix

    Parameters
    ----------

    features_to_clean : numpy.ndarray
        The input features to be cleaned with shape <n_observations, n_regions, n_regions>,

    covariates_to_regress : list
        A list containing the covariates whose influence should be removed from features_to_clean.
        The entries to the list can either be a numpy.ndarray containing continious values, or
        can be a list of strings that will be used to define different groups whose influence
        will be removed.

    Returns
    -------

    numpy.ndarray
        The features_to_clean, after covariates_to_regress have been regressed through a linear
        model.


    """

    raise NameError('Error: confirm this function works before using')
    #First construct the regression matrix
    for temp_item in covariates_to_regress:

        num_features = features_to_clean.shape[0]

        if type(temp_item[0]) == str:

            unique_entries = []
            for temp_entry in temp_item:

                if temp_entry not in unique_entries:
                    unique_entries.append(temp_entry)

            if len(unique_entries) < 2:

                #Ignore if there aren't at least two categories
                pass

            elif len(unique_entries) == 2:

                temp_nominal = np.zeros(num_features)

                for i, temp_entry in enumerate(temp_item):

                    if temp_entry == temp_nominal[0]:

                        temp_nominal[i] = 1

                formatted_covariates.append(temp_nominal.copy())

            else:

                temp_nominal = np.zeros((num_features, len(unique_entries)))

                for i, temp_entry in enumerate(temp_item):
                    for j, temp_unique in enumerate(temp_item):

                        if temp_unique == temp_entry:

                            temp_nominal[i,j] = 1

                formatted_covariates.append(temp_nominal.copy())

        else:

            formatted_covariates.append(temp_item)

    regressors = np.vstack(formatted_covariates).transpose()

    #Second remove the influence of covariates
    XT_X_Neg1_XT = imaging_utils.calculate_XT_X_Neg1_XT(regressors)
    print(XT_X_Neg1_XT.shape)

    for i in range(features_to_clean.shape[1]):
        for j in range(features_to_clean.shape[2]):

            cleaned_features[:,i,j] = np.squeeze(imaging_utils.partial_clean_fast(features_to_clean[:,i,j][:,None], XT_X_Neg1_XT, regressors))


    return cleaned_features


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
