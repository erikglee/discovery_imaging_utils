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