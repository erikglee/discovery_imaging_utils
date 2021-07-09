import numpy as np
import nibabel as nib





def arr2nifti(array, affine, output_path):

    """Function to convert numpy array to nifti

    Parameters
    ----------
    array : numpy.ndarray
        array whose data will be the contents of a nifti image

    affine : numpy.ndarray
        array with shape <4, 4> that will serve as the affine
        for the generated nifti file

    output_path : str
        path to nifti file to be created (choose your own extension)


    """

    array_img = nib.Nifti1Image(array, affine)
    nib.save(array_img, output_path)

    return


def convert_spherical_roi_coords_to_nifti(template_nifti_path, spherical_coords, radius, output_nifti_path, spherical_labels=None):
    """
    #Template_nifti_path should point to a nifti with the desired
    #affine matrix/size, should be 3d, not a timeseries. Spherical_coords
    #should be a list of RAS coordinates for the spheres, radius should be a list
    #of radii for the different spheres, output_nifti_path is where the mask file
    #will be saved. Spherical labels is an optional list that can specify the number
    #assigned to values for different spheres. If this isn't set, spheres will be labeled
    #1, 2, 3 ... etc.
    """

    template_nifti = nib.load(template_nifti_path)
    affine = template_nifti.affine
    mask_vol = np.zeros(template_nifti.get_fdata().shape)

    for i in range(mask_vol.shape[0]):
        for j in range(mask_vol.shape[1]):
            for k in range(mask_vol.shape[2]):

                temp_ras = np.matmul(affine,[i, j, k, 1])[0:3]
                for l in range(len(spherical_coords)):

                    distance = np.linalg.norm(spherical_coords[l] - temp_ras)

                    if radius[l] <= distance:

                        if type(spherical_labels) == None:
                            mask_vol[i,j,k] = l + 1
                        else:
                            mask_vol[i,j,k] = spherical_labels[l]
                        break

    template_header = template_nifti.header
    img = nib.nifti1Image(mask_vol, affine, header = template_header)
    nib.save(img, output_nifti_path)

    print(output_nifti_path)


    return






def nifti_rois_to_time_signals(input_timeseries_nii_path, input_mask_nii_path, demedian_before_averaging = True):

    """
    #Function that takes a 4d nifti file with path input_timeseries_nii_path,
    #and a 3d mask registered to the 4d timeseries (input_mask_nii_path) who has,
    #different values which specify different regions, and returns average time
    #signals for the different regions. By default, each voxel in a given ROI will be
    #divided by its median prior to averaging. The mean of all voxel medians will be
    #provided as output. To turn off this normalization, set demedian_before_averaging
    #to False.
    #
    #Output nifti_time_series - size n_regions, n_timepoints
    #unique_mask_vals - size n_regions (specifying the ID for each mask)
    #parc_mean_median_signal_intensities - size n_regions
    """

    input_ts_nii = nib.load(input_timeseries_nii_path)
    input_mask_nii = nib.load(input_mask_nii_path)

    input_mask_matrix = input_mask_nii.get_fdata()
    input_ts_matrix = input_ts_nii.get_fdata()

    if input_mask_matrix.shape[0:3] != input_ts_matrix.shape[0:3]:
        raise NameError('Error: the first three dimensions of the input mask and input timeseries must be the same.')

    unique_mask_vals = np.unique(input_mask_matrix)
    unique_mask_vals.sort()
    unique_mask_vals = unique_mask_vals[1:]

    nifti_time_series = np.zeros((unique_mask_vals.shape[0], input_ts_matrix.shape[3]))
    parc_mean_median_signal_intensities = np.zeros(unique_mask_vals.shape[0])



    for i in range(len(unique_mask_vals)):

        inds = np.where(input_mask_matrix == unique_mask_vals[i])
        temp_timeseries = input_ts_matrix[inds]

        voxel_medians = np.nanmedian(temp_timeseries, axis=1)
        voxel_medians[np.where(voxel_medians < 0.001)] = np.nan

        if demedian_before_averaging:
            temp_timeseries = temp_timeseries/voxel_medians[:,None]



        nifti_time_series[i,:] = np.nanmean(temp_timeseries, axis=0)
        parc_mean_median_signal_intensities[i] = np.nanmean(voxel_medians)

    return nifti_time_series, unique_mask_vals, parc_mean_median_signal_intensities


def parcellate_nifti(nifti_data_to_parcellate, parcellation_path, demean_before_averaging = True):

	"""

	Parameters
	----------

	loaded_nifti_to_parcellate : numpy.ndarray
	3d or 4d ndarray with data to parcellate
	parcellation_path : str
	path to the 3d parcellation nifti to use
	demean_before_averaging : bool, optional
	whether or not to demean voxels before averaging

	#Function that takes a 4d nifti file with path input_timeseries_nii_path,
	#and a 3d mask registered to the 4d timeseries (input_mask_nii_path) who has,
	#different values which specify different regions, and returns average time
	#signals for the different regions. By default, each voxel in a given ROI will be
	#divided by its median prior to averaging. The mean of all voxel medians will be
	#provided as output. To turn off this normalization, set demedian_before_averaging
	#to False.
	#
	#Output nifti_time_series - size n_regions, n_timepoints
	#unique_mask_vals - size n_regions (specifying the ID for each mask)
	#parc_mean_median_signal_intensities - size n_regions
	"""

	#Load the parcellation nifti and find the
	#ids for the different parcels
	input_mask_nii = nib.load(parcellation_path)
	input_mask_matrix = input_mask_nii.get_fdata()
	unique_mask_vals = np.unique(input_mask_matrix)
	unique_mask_vals.sort()
	unique_mask_vals = unique_mask_vals[1:]



	#extract data from nifti to be parcellated
	input_ts_matrix = nifti_data_to_parcellate


	#Create array to store parcellated output
	if input_ts_matrix.ndim == 4:
		depth = input_ts_matrix.shape[3]
	else:
		depth = 1

	parcellated_nifti_data = np.zeros((unique_mask_vals.shape[0], depth))
	parc_mean_signal_intensities = np.zeros(unique_mask_vals.shape[0])
	parcel_dictionary = {}

	if input_mask_matrix.shape[0:3] != input_ts_matrix.shape[0:3]:
		raise NameError('Error: the first three dimensions of the input mask and input timeseries must be the same.')


	for i in range(len(unique_mask_vals)):

		inds = np.where(input_mask_matrix == unique_mask_vals[i])
		parcel_dictionary['nii_' + str(unique_mask_vals[i])] = inds
		temp_timeseries = input_ts_matrix[inds]

		if depth > 1:
			voxel_means = np.nanmean(temp_timeseries, axis=1)
		else:
			voxel_means = temp_timeseries

		voxel_means[np.where(np.abs(voxel_means) < 0.000001)] = np.nan
		parcel_mean = np.nanmean(voxel_means)

		if demean_before_averaging:

			if depth > 1:
				temp_timeseries = temp_timeseries/voxel_means[:,None]
				parcellated_nifti_data[i,:] = np.nanmean(temp_timeseries, axis=0)*parcel_mean
			else:
				parcellated_nifti_data[i,:] = np.nanmean(temp_timeseries)

		else:

			if depth > 1:
				parcellated_nifti_data[i,:] = np.nanmean(temp_timeseries, axis=0)
			else:
				parcellated_nifti_data[i,:] = np.nanmean(temp_timeseries)


	unique_mask_vals = unique_mask_vals.tolist()



	return parcellated_nifti_data, unique_mask_vals, parcel_dictionary

def incorporate_nifti_inclusion_mask(func_data, inclusion_mask_path, cutoff = 0.5):

	inclusion_mask_data = nib.load(inclusion_mask_path).get_fdata()

	if inclusion_mask_data.shape[0:3] != func_data.shape[0:3]:
		raise NameError('Error: the first three dimensions of the input mask and input timeseries must be the same.')

	inds_to_include = np.where(inclusion_mask_data > cutoff)
	inds_to_exclude = np.where(inclusion_mask_data <= cutoff)

    masked_func_data = func_data[inds_to_include]
	func_data[inds_to_exclude] = np.nan

	return func_data, inds_to_include
