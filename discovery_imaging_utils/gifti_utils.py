import numpy as np
import nibabel as nib
from nibabel import load as nib_load

def load_gifti_func(path_to_file):
    """
    #Wrapper function to load functional data from
    #a gifti file using nibabel. Returns data in shape
    #<num_verts x num_timepoints>
    """

    gifti_img = nib_load(path_to_file)
    gifti_list = [x.data for x in gifti_img.darrays]
    gifti_data = np.vstack(gifti_list).transpose()

    return gifti_data

def arr2gifti(array, output_path, hemi = ''):

    """Function to convert numpy array to gifti

    Parameters
    ----------
    array : numpy.ndarray
        1d or 2d array with features <n_vertices, n_dims> to be put in
        gifti image

    output_path : str
        path to gifti file to be created (choose your own extension)

    hemi : str
        TO BE IMPLEMENTED - 'LH' or 'RH' will specify which surface gifti
        file will match


    """

    darrays = []

    if array.ndim == 1:
        temp_darray = nib.gifti.gifti.GiftiDataArray(data=array.astype(np.float32))
        darrays.append(temp_darray)
    else:
        for i in range(array.shape[1]):
            temp_darray = nib.gifti.gifti.GiftiDataArray(data=array[:,i].astype(np.float32))
            darrays.append(temp_darray)

    #How to upgrade to have anat built in:
    #saved_from_wb._meta.data[0].__dict__
    #{'name': 'AnatomicalStructurePrimary', 'value': 'CortexLeft'}


    img = nib.gifti.gifti.GiftiImage(darrays=darrays)
    nib.save(img, output_path)
    print(output_path)

    return



def parcellate_gifti(func_data, parcellation_path, demean_before_averaging = True):
	"""Internal function to parcelate gifti data.

	Function assumes first region in parcellation is the medial wall and Should
	be ignored. If demean option is used, vertex data is normalized to temporal
	mean prior to spatial averaging, and the parcel-wise mean of all vertices is
	put back to the parcel data afterwards (so demeaning makes vertices have
	equal weight, but parcel still has mean at end)

	Parameters
	----------

	func_data : numpy.ndarray
	data to be parcellated with shape <n_vertices, n_timepoints> or
	shape <n_vertices>. NaN and values less than 0.000001 will be ignored
	parcellation_path : str
	path to FreeSurfer .annot file containing regions
	demean_before_averaging : bool, optional
	whether or not to demean vertices before parcellation so that
	each vertex is weighted evenly. If used, the mean will be
	put back into the data at the parcel level


	Returns
	-------

	parcellated_giti_data : numpy.ndarray
	parcellated data with shape <n_regions, n_features>, where n_regions is
	the number of parcels from the parcellation (minus medial wall) and
	n_features is the shape of the second dimension of func_data
	parcel_labels : list of strings
	the names for the different parcels
	parcel_dictionary : dict
	dictionary whose keys are parcel names and values are zero-indexed
	indices of vertices belonging to each parcel (in case you want to
	project parcel data back to the surface later after manipulations)


	"""

	#Load annotation
	#Output will be tuple of format [labels, ctab, names]
	parcellation = nib.freesurfer.io.read_annot(parcellation_path)

	func_data = func_data.astype(float)


	#Then concatenate parcel labels and parcel timeseries between the left and right hemisphere
	#and drop the medial wall from label list
	parcel_labels = parcellation[2][1:]

	#Try to convert the parcel labels from bytes to normal string
	for i in range(0, len(parcel_labels)):
		parcel_labels[i] = parcel_labels[i].decode("utf-8")

	#Make array to store parcellated data with shape <num_parcels, num_timepoints>
	if func_data.ndim > 1:
		depth = func_data.shape[1]
	else:
		depth = 1

	parcellated_gifti_data = np.zeros((len(parcellation[2]) - 1, depth))
	parcel_dictionary = {}

	#Skip the first parcel (should be medial wall)
	for i in range(1,len(parcellation[2])):

		#Find the vertices for the current parcel
		vois = np.where(parcellation[0] == i)
		parcel_dictionary[parcel_labels[i-1]] = vois
		temp_timeseries = func_data[vois]

		if depth > 1:
			vertex_means = np.nanmean(temp_timeseries, axis=1)
		else:
			vertex_means = temp_timeseries

			vertex_means[np.where(np.abs(vertex_means) < 0.000001)] = np.nan

		parcel_mean = np.nanmean(vertex_means)

		if demean_before_averaging:

			if depth > 1:
				temp_timeseries = temp_timeseries/vertex_means[:,None]
				parcellated_gifti_data[i - 1,:] = np.nanmean(temp_timeseries, axis=0)*parcel_mean
			else:
				parcellated_gifti_data[i - 1,:] = np.nanmean(temp_timeseries)

		else:

			if depth > 1:
				parcellated_gifti_data[i - 1,:] = np.nanmean(temp_timeseries, axis=0)
			else:
				parcellated_gifti_data[i - 1,:] = np.nanmean(temp_timeseries)


	return parcellated_gifti_data, parcel_labels, parcel_dictionary


def incorporate_gifti_inclusion_mask(func_data, inclusion_mask_path, cutoff = 0.5):

	inclusion_mask_data = imaging_utils.load_gifti_func(inclusion_mask_path)
	inds_to_include = np.where(inclusion_mask_data > cutoff)
	inds_to_exclude = np.where(inclusion_mask_data <= cutoff)

	func_data[inds_to_exclude] = np.nan


	return func_data, inds_to_include
