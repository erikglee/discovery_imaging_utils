import pandas as pd
import numpy as np
import os
import glob
import nibabel
from discovery_imaging_utils import imaging_utils
from discovery_imaging_utils import nifti_utils
from discovery_imaging_utils import gifti_utils
import json

#Run in this order:
#(1) file_paths_dict = generate_file_paths(....)
#(2) check if files present with all_file_paths_exist(...)
#(3) if files present, parc_ts_dict = populate_parc_dictionary(....)
#(4) then use save/load functions to store in directory structure


def generate_file_paths(lh_gii_data_path=None,
						lh_inclusion_mask_path=None,
						lh_parcellation_path=None,
						nifti_data_path=None,
						nifti_inclusion_mask_path=None,
						nifti_parcellation_path=None,
						aroma_included=True):
	"""Function to generate paths for an image_data_dictionary.

	Function that populates file paths for an image_data_dictionary.
	This image_data_dictionary can support nifti data, surface data (gifti),
	either in input image space or reduced through a parcellation scheme. At a minimum,
	the function can simply take a path to a nifti file or alternatively a gifti file (or both).
	Beyond this functionality, if only certain voxels/vertices should be included in the
	image_data_dictionary, then then nifti/gifti masks specifying which elements should be included
	can be specified. Further, if either the gifti or nifti images should be reduced via a parcellation
	scheme, that can be specified also. For spaces where a parcellation is specified, only data
	averages within parcels/rois will be propogated to the image_data_dictionary.

	At minimum - either lh_gii_path or nifti_path must be defined to run function.


	Parameters
	----------

	lh_gii_data_path : str, optional
	path to a lh gifti image that will be used to define an image_data_dictionary.
	All remaining gifti parameters will only be used if the lh_gii_path is defined.
	The rh gifti path will be inferred based on the lh_gii_path.

	lh_inclusion_mask_path : str, optional
	path to a lh gifti image that will be used to specify which vertices should be
	included in the image_data_dictionary. Non-zero values will be included in final
	dictionary. If this argument is not set, then all vertices will be included.
	Again rh path will be inferred.

	lh_parcellation_path : str, optional
	path to an annotation image that will be used to parcellate the data. If this
	option is used, only parcel level data and not vertex level data will be included
	in the image_data_dictionary. rh_path will be inferred

	nifti_data_path : str, optional
	path to a nifti image that will be used to define an image_data_dictionary. All
	remaining nifti parameters will only be used if the nifti_path is defined.

	nifti_inclusion_mask_path : str, optional
	path to a nifti image whose non-zero locations will be used to define which voxels
	will be included in the image_data_dictionary. If not specified, then all voxels
	will be included.

	nifti_parcellation_path : str, optional
	path to a nifti image whose unique (non-zero) values will be used to parcellate the
	input nifti data. If this option is used, only region level data and not voxel level
	data will be included in the image_data_dictionary.

	aroma_included : bool, optional
	defaults to True, specifies whether or not aroma files should be included in the dictionary.
	If fmriprep was ran without the AROMA flag, set to False.

	Returns
	-------

	dict
	Dictionary with file paths that will be used to generate image_data_dictionary after running
	all_file_paths_exist, then populate_image_data_dictionary.


	Examples
	--------



	"""


	#Check that there is input data
	if type(lh_gii_data_path) == type(None):
		if type(nifti_data_path) == type(None):
			raise NameError('Error: At minimum must specify lh_gii_data_path or nifti_data_path')


	path_dictionary = {}
	prefix = ''

	#Gifti related paths
	if type(lh_gii_data_path) != type(None):

		path_dictionary['lh_gii_data_path'] = lh_gii_data_path

		#For new fMRIPREP naming convention
		if 'hemi-L' in lh_gii_data_path:
			path_dictionary['rh_gii_data_path'] = lh_gii_data_path.replace('hemi-L', 'hemi-R')

			#For old fMRIPREP naming convention
		else:
			path_dictionary['lh_gii_data_path'] = lh_gii_data_path.replace('L.func.gii','R.func.gii')

			#For finding aroma/nuisance files
		prefix = lh_gii_data_path[0:lh_gii_data_path.find('_space')]


	#MAYBE SHOULDN'T RELY ON NAME STARTING WITH lh....???
	#Paths for potential surface inclusion masks
	if type(lh_inclusion_mask_path) != type(None):

		path_dictionary['lh_inclusion_mask_path'] = lh_inclusion_mask_path
		path_dictionary['rh_inclusion_mask_path'] = '/'.join(lh_inclusion_mask_path.split('/')[0:-1]) + '/rh' + lh_inclusion_mask_path.split('/')[-1][2:]

	#Paths for potential surface inclusion masks
	if type(lh_parcellation_path) != type(None):

		path_dictionary['lh_parcellation_path'] = lh_parcellation_path
		path_dictionary['rh_parcellation_path'] = '/'.join(lh_parcellation_path.split('/')[0:-1]) + '/rh' + lh_parcellation_path.split('/')[-1][2:]




	#Nifti related paths
	if type(nifti_data_path) != type(None):

		path_dictionary['nifti_data_path'] = nifti_data_path
		nifti_prefix = nifti_data_path[0:nifti_data_path.find('_space')]

		if prefix != '':
			if nifti_prefix != prefix:
				raise NameError('Error: It doesnt look like the nifti and gifti timeseries point to the same run')
			else:
				prefix = nifti_prefix

			if type(nifti_inclusion_mask_path) != type(None):

				path_dictionary['nifti_inclusion_mask_path'] = nifti_inclusion_mask_path

			if type(nifti_parcellation_path) != type(None):

				path_dictionary['nifti_parcellation_path'] = nifti_parcellation_path

	#Aroma related paths
	if aroma_included:
		path_dictionary['melodic_mixing_path'] = prefix + '_desc-MELODIC_mixing.tsv'
		path_dictionary['aroma_noise_ics_path'] = prefix + '_AROMAnoiseICs.csv'

		#Confounds path
		path_dictionary['confounds_regressors_path'] = prefix + '_desc-confounds_regressors.tsv'


	return path_dictionary

def all_file_paths_exist(file_path_dictionary):
	"""Takes a dictionary of file paths and checks if they exist.

	Takes a dictionary where each entry is a string representing a file
	path, and iterates over all entries checking whether or not they each
	point to a file.

	Parameters
	----------
	file_path_dictionary : dict
	a dictionary where all entries are paths to files

	Returns
	-------
	files_present : bool
	a boolean saying whether or not all files in the dictionary were found

	"""

	#Check if all files exist, and if they don't
	#return False
	files_present = True

	for temp_field in file_path_dictionary:

		if os.path.exists(file_path_dictionary[temp_field]) == False:
			print('File not found: ' + file_path_dictionary[temp_field])
			files_present = False

	return files_present


def populate_dict_for_denoising(file_path_dictionary, TR, normalize = True):
	"""Function to populate a dictionary with data to use in denoising

	Parameters
	----------

	file_path_dictionary : dict of str
	dictionary with file paths (created by generate_file_paths)
	TR : float
	repetition time of acquisition in seconds
	normalize : bool
	whether or not all output data elements should be set to have a temporal
	mean of 10000

	"""

	if 'aroma_noise_ics_path' in file_path_dictionary.keys():
		aroma_used = True


	dict_for_denoising = {}
	dict_for_denoising['file_path_dictionary.json'] = file_path_dictionary
	dict_for_denoising['image_data_dictionary'] = populate_image_data_dictionary(file_path_dictionary, normalize = normalize)
	dict_for_denoising['confounds'] = _populate_confounds_dict(file_path_dictionary, aroma_used = aroma_used)
	dict_for_denoising['general_info.json'] = _populate_general_info_dict(dict_for_denoising['confounds'], dict_for_denoising['file_path_dictionary.json'], TR)


	return dict_for_denoising


def populate_image_data_dictionary(file_path_dictionary, normalize = False):
	"""Takes a file_path_dictionary and uses it to populate an image_data_dictionary

	Takes a file_path_dictionary generated by, generate_file_paths, and creates an
	image_data_dictionary.


	Parameters
	----------

	file_path_dictionary : dict
	dictionary created by generate_file_paths

	TR : float
	the repitition time in seconds

	normalize : bool
	whether data should be normalized (defaults to true). Automatically
	done prior to parcellation, but this determines if the output data
	will be mean of 10k. Only works for 4d nifti or 2d surface data.


	Returns
	-------

	image_data_dictionary : dict
	.....input details



	"""



	image_data_dict = {}
	metadata_dict = {}

	lh_data = None
	nifti_data = None

	has_gifti = False
	has_gifti_parcellation = False
	has_nifti = False
	has_nifti_parcellation = False

	#If there is surface data
	if 'lh_gii_data_path' in file_path_dictionary.keys():

		has_gifti = True
		lh_data = imaging_utils.load_gifti_func(file_path_dictionary['lh_gii_data_path'])
		rh_data = imaging_utils.load_gifti_func(file_path_dictionary['rh_gii_data_path'])

		image_data_dict['lh_gifti_shape'] = lh_data.shape
		image_data_dict['rh_gifti_shape'] = rh_data.shape
		metadata_dict['lh_gifti_data_path'] = file_path_dictionary['lh_gii_data_path']
		metadata_dict['rh_gifti_data_path'] = file_path_dictionary['rh_gii_data_path']


		lh_gifti_ids = np.arange(0, lh_data.shape[0], 1, dtype=int)
		rh_gifti_ids = np.arange(0, rh_data.shape[0], 1, dtype=int)

		#If inclusion mask is specified, set zero values in inclusion
		#mask to NaN.. the _parcellate_gifti function knows how to handle this
		if 'lh_inclusion_mask_path' in file_path_dictionary.keys():

			#Make function to set appropriate values to NaN.....
			lh_data, lh_inclusion_inds = _incorporate_gifti_inclusion_mask(lh_data, file_path_dictionary['lh_inclusion_mask_path'])
			rh_data, rh_inclusion_inds = _incorporate_gifti_inclusion_mask(rh_data, file_path_dictionary['rh_inclusion_mask_path'])
			metadata_dict['lh_gifti_inclusion_mask_path'] = file_path_dictionary['lh_inclusion_mask_path']
			metadata_dict['rh_gifti_inclusion_mask_path'] = file_path_dictionary['rh_inclusion_mask_path']

			lh_gifti_ids = lh_inclusion_inds
			rh_gifti_ids = rh_inclusion_inds


			if 'lh_parcellation_path' in file_path_dictionary.keys():

				has_gifti_parcellation = True
				#Need to (1) parcellate data, (2) return parcel labels, (3) save info to recreate parcels
				lh_data, lh_labels, lh_parcels_dict = _parcellate_gifti(lh_data, file_path_dictionary['lh_parcellation_path'])
				rh_data, rh_labels, rh_parcels_dict = _parcellate_gifti(rh_data, file_path_dictionary['rh_parcellation_path'])
				metadata_dict['lh_gifti_parcellation_path'] = file_path_dictionary['lh_gii_parcellation_path']
				metadata_dict['rh_gifti_parcellation_path'] = file_path_dictionary['rh_gii_parcellation_path']

				lh_gifti_ids = lh_labels
				rh_gifti_ids = rh_labels

				image_data_dict['lh_parcels_dict'] = lh_parcels_dict
				image_data_dict['rh_parcels_dict'] = rh_parcels_dict


	#If there is nifti data
	if 'nifti_data_path' in file_path_dictionary.keys():

		has_nifti = True
		nifti_img = nib.load(file_path_dictionary['nifti_data_path'])
		metadata_dict['nifti_data_path'] = file_path_dictionary['nifti_data_path']
		nifti_data = nifti_img.get_fdata()
		nifti_ids = np.indices(nifti_data.shape)
		nifti_inclusions_inds = None

		image_data_dict['nifti_affine'] = nifti_img.affine
		image_data_dict['nifti_shape'] = nifti_data.shape

		if 'nifti_inclusion_mask_path' in file_path_dictionary.keys():

			nifti_data, nifti_inclusions_inds = _incorporate_nifti_inclusion_mask(nifti_data, file_path_dictionary['nifti_inclusion_mask'])
			metadata_dict['nifti_inclusion_mask_path'] = file_path_dictionary['nifti_inclusion_mask']
			nifti_ids = nifti_inclusion_inds

		if 'nifti_parcellation_path' in file_path_dictionary.keys():

			has_nifti_parcellation = True
			nifti_data, nifti_labels, nifti_parcels_dict = _parcellate_nifti(nifti_data, file_path_dictionary['nifti_parcellation_path'])
			metadata_dict['nifti_parcellation_path'] = file_path_dictionary['nifti_parcellation_path']
			nifti_ids = nifti_labels

			image_data_dict['nifti_parcels_dict'] = nifti_parcels_dict

		#If the data hasn't already been brought down to 2d, then do that now
		if nifti_data.ndim > 2:

			#Check what the final dimension should be
			if nifti_data.ndim == 3:
				depth = 1
			else:
				depth = nifti_data.shape[4]


			if type(nifti_inclusion_inds) != type(None):

				nifti_data = np.reshape(nifti_data[nifti_inclusion_inds], (nifti_inclusion_inds[0].shape[0], depth))

			else:

				nifti_data = np.reshape(nifti_data, (nifti_data.shape[0]*nifti_data.shape[1]*nifti_data.shape[2], depth))


	data = None
	lh_data_inds = None
	rh_data_inds = None
	nifti_data_inds = None

	#FYI...
	#data inds will specify how to access lh/rh/nifti elements from the data
	#key, and the ids will alternatively specify what those inds mean in terms
	#of the lh/rh/nifti or parcellation schemes


	#Add gifti data
	rh_data_inds = None #for nifti later
	if type(lh_data) != type(None):
		data = np.vstack((lh_data, rh_data))
		image_data_dict['lh_data_inds'] = np.arange(0, lh_data_inds.shape[0], 1, dtype=int)
		image_data_dict['rh_data_inds'] = np.arange(lh_data_inds.shape[0], lh_data_inds.shape[0] + rh_data_inds.shape[0], 1, dtype=int)
		image_data_dict['lh_ids'] = lh_gifti_ids
		image_data_dict['rh_ids'] = rh_gifti_ids


	#Add nifti data
	if type(nifti_data) != type(None):
		if type(data) != type(None):
			data = np.vstack((data, nifti_data))
			nifti_data_inds = np.arange(rh_data_inds[-1], rh_data_inds[-1] + nifti_data.shape[0], 1, dtype=int)
		else:
			data = nifti_data
			nifti_data_inds = np.arange(0, nifti_data.shape[0], 1, dtype=int)

			image_data_dict['nifti_data_inds'] = nifti_data_inds
			image_data_dict['nifti_ids'] = nifti_ids

	#Normaize data if necessary
	if normalize == True:
		if data.shape[1] > 1:

			image_data_dict['data_means'] = np.mean(data,axis=1)
			data = data/image_data_dict['data_means']*10000


			image_data_dict['data'] = data

			return image_data_dict


def _parcellate_gifti(func_data, parcellation_path, demean_before_averaging = True):
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


def _parcellate_nifti(nifti_data_to_parcellate, parcellation_path, demean_before_averaging = True):

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

	for i in range(len(unique_mask_vals)):

		inds = np.where(input_mask_matrix == unique_mask_vals[i])
		parcel_dictionary[unique_mask_vals[i]] = inds
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


def _incorporate_gifti_inclusion_mask(func_data, inclusion_mask_path, cutoff = 0.5):

	inclusion_mask_data = imaging_utils.load_gifti_func(inclusion_mask_path)
	inds_to_include = np.where(inclusion_mask_data > cutoff)
	inds_to_exclude = np.where(inclusion_mask_data <= cutoff)

	func_data[inds_to_exclude] = np.nan


	return func_data, inds_to_include


def _incorporate_nifti_inclusion_mask(func_data, inclusion_mask_path, cutoff = 0.5):

	inclusion_mask_data = nib.load(inclusion_mask_path).get_fdata()
	inds_to_include = np.where(inclusion_mask_data > cutoff)
	inds_to_exclude = np.where(inclusion_mask_data <= cutoff)

	func_data[inds_to_exclude] = np.nan

	return func_data, inds_to_include


def _populate_general_info_dict(confounds_dict, file_path_dict, TR):

	general_info_dict = {}

	###################################################
	#Calculate the number of timepoints to skip at the beginning for this person.
	#If equal to zero, we will actually call it one so that we don't run into any
	#issues during denoising with derivatives
	general_info_dict['n_skip_vols'] = len(np.where(np.absolute(confounds_dict['a_comp_cor_00']) < 0.00001)[0])
	if general_info_dict['n_skip_vols'] == 0:
		general_info_dict['n_skip_vols'] = 1

		general_info_dict['mean_fd'] = np.mean(confounds_dict['framewise_displacement'][general_info_dict['n_skip_vols']:])
		general_info_dict['mean_dvars'] = np.mean(confounds_dict['dvars'][general_info_dict['n_skip_vols']:])
		general_info_dict['num_std_dvars_above_1p5'] = len(np.where(confounds_dict['std_dvars'][general_info_dict['n_skip_vols']:] > 1.5)[0])
		general_info_dict['num_std_dvars_above_1p3'] = len(np.where(confounds_dict['std_dvars'][general_info_dict['n_skip_vols']:] > 1.3)[0])
		general_info_dict['num_std_dvars_above_1p2'] = len(np.where(confounds_dict['std_dvars'][general_info_dict['n_skip_vols']:] > 1.2)[0])

		general_info_dict['num_fd_above_0p5_mm'] = len(np.where(confounds_dict['framewise_displacement'][general_info_dict['n_skip_vols']:] > 0.5)[0])
		general_info_dict['num_fd_above_0p4_mm'] = len(np.where(confounds_dict['framewise_displacement'][general_info_dict['n_skip_vols']:] > 0.4)[0])
		general_info_dict['num_fd_above_0p3_mm'] = len(np.where(confounds_dict['framewise_displacement'][general_info_dict['n_skip_vols']:] > 0.3)[0])
		general_info_dict['num_fd_above_0p2_mm'] = len(np.where(confounds_dict['framewise_displacement'][general_info_dict['n_skip_vols']:] > 0.2)[0])


	#Set TR
	general_info_dict['TR'] = TR

	#Find session/subject names
	if 'lh_gii_data_path' in file_path_dict.keys():

		temp_path = file_path_dict['lh_gii_data_path'].split('/')[-1]
		split_end_path = temp_path.split('_')

	else:

		temp_path = file_path_dict['nifti_data_path'].split('/')[-1]
		split_end_path = temp_path.split('_')

		general_info_dict['subject'] = split_end_path[0]

	if split_end_path[1][0:3] == 'ses':
		general_info_dict['session'] = split_end_path[1]
	else:
		general_info_dict['session'] = []

	return general_info_dict



def _populate_confounds_dict(file_path_dictionary, aroma_used = True):
	"""Function to populate a confounds dict based on fMRIPREP output

	Parameters
	----------

	file_path_dictionary : dict
	Dictionary with file paths needed to construct confounds dict, at a
	minimum this must have keys 'confounds_regressors_path' and
	if AROMA was used 'melodic_mixing_path' and 'aroma_noise_ics_path'
	aroma_used : bool,
	specifies whether to load AROMA files in the confounds dictionary
	(must have files from fMRIPREP with AROMA info to do this)

	Returns
	-------

	confounds_dict : dict
	Dictionary with different key/value pairs representing groupings of
	useful temporal confounds that can be incorporated into denoising

	"""

	confounds_dictionary = {}

	confounds_regressors_tsv_path = file_path_dictionary['confounds_regressors_path']
	confound_df = pd.read_csv(confounds_regressors_tsv_path, sep='\t')
	for (columnName, columnData) in confound_df.iteritems():
		confounds_dictionary[columnName] = columnData.values


		#For convenience, bunch together some commonly used nuisance components

		#Six motion realignment paramters
		confounds_dictionary['motion_regs_six'] = np.vstack((confounds_dictionary['trans_x'], confounds_dictionary['trans_y'], confounds_dictionary['trans_z'],
		confounds_dictionary['rot_x'], confounds_dictionary['rot_y'], confounds_dictionary['rot_z']))

		#Six motion realignment parameters plus their temporal derivatives
		confounds_dictionary['motion_regs_twelve'] = np.vstack((confounds_dictionary['trans_x'], confounds_dictionary['trans_y'], confounds_dictionary['trans_z'],
		confounds_dictionary['rot_x'], confounds_dictionary['rot_y'], confounds_dictionary['rot_z'],
		confounds_dictionary['trans_x_derivative1'], confounds_dictionary['trans_y_derivative1'],
		confounds_dictionary['trans_z_derivative1'], confounds_dictionary['rot_x_derivative1'],
		confounds_dictionary['rot_y_derivative1'], confounds_dictionary['rot_z_derivative1']))

		#Six motion realignment parameters, their temporal derivatives, and
		#the square of both
		confounds_dictionary['motion_regs_twentyfour'] = np.vstack((confounds_dictionary['trans_x'], confounds_dictionary['trans_y'], confounds_dictionary['trans_z'],
		confounds_dictionary['rot_x'], confounds_dictionary['rot_y'], confounds_dictionary['rot_z'],
		confounds_dictionary['trans_x_derivative1'], confounds_dictionary['trans_y_derivative1'],
		confounds_dictionary['trans_z_derivative1'], confounds_dictionary['rot_x_derivative1'],
		confounds_dictionary['rot_y_derivative1'], confounds_dictionary['rot_z_derivative1'],
		confounds_dictionary['trans_x_power2'], confounds_dictionary['trans_y_power2'], confounds_dictionary['trans_z_power2'],
		confounds_dictionary['rot_x_power2'], confounds_dictionary['rot_y_power2'], confounds_dictionary['rot_z_power2'],
		confounds_dictionary['trans_x_derivative1_power2'], confounds_dictionary['trans_y_derivative1_power2'],
		confounds_dictionary['trans_z_derivative1_power2'], confounds_dictionary['rot_x_derivative1_power2'],
		confounds_dictionary['rot_y_derivative1_power2'], confounds_dictionary['rot_z_derivative1_power2']))

		#white matter, and csf
		confounds_dictionary['wmcsf'] = np.vstack((confounds_dictionary['white_matter'], confounds_dictionary['csf']))

		#white matter, csf, and their temporal derivatives
		confounds_dictionary['wmcsf_derivs'] = np.vstack((confounds_dictionary['white_matter'], confounds_dictionary['csf'],
		confounds_dictionary['white_matter_derivative1'], confounds_dictionary['csf_derivative1']))

		#White matter, csf, and global signal
		confounds_dictionary['wmcsfgsr'] = np.vstack((confounds_dictionary['white_matter'], confounds_dictionary['csf'], confounds_dictionary['global_signal']))

		#White matter, csf, and global signal plus their temporal derivatives
		confounds_dictionary['wmcsfgsr_derivs'] = np.vstack((confounds_dictionary['white_matter'], confounds_dictionary['csf'], confounds_dictionary['global_signal'],
		confounds_dictionary['white_matter_derivative1'], confounds_dictionary['csf_derivative1'],
		confounds_dictionary['global_signal_derivative1']))

		#The first five anatomical comp cor components
		confounds_dictionary['five_acompcors'] = np.vstack((confounds_dictionary['a_comp_cor_00'], confounds_dictionary['a_comp_cor_01'],
		confounds_dictionary['a_comp_cor_02'], confounds_dictionary['a_comp_cor_03'],
		confounds_dictionary['a_comp_cor_04']))


	if aroma_used:
		####################################################
		#Load the melodic IC time series
		melodic_df = pd.read_csv(file_path_dictionary['melodic_mixing_path'], sep="\t", header=None)

		####################################################
		#Load the indices of the aroma ics
		aroma_ics_df = pd.read_csv(file_path_dictionary['aroma_noise_ics_path'], header=None)
		noise_comps = (aroma_ics_df.values - 1).reshape(-1,1)


		####################################################
		#Gather the ICs identified as noise/clean by AROMA
		all_ics = melodic_df.values
		mask = np.zeros(all_ics.shape[1],dtype=bool)
		mask[noise_comps] = True
		confounds_dictionary['aroma_noise_ics'] = np.transpose(all_ics[:,~mask])
		confounds_dictionary['aroma_clean_ics'] = np.transpose(all_ics[:,mask])

		return confounds_dictionary



def convert_to_images(image_data_dict, output_folder, overwrite = False):

	"""Function that converts image data dictionaries back to nifti/gifti

	Takes an image data dictionary, possibly parcellated and or transformed
	data from a gifti and or nifti file and uses information about the base
	files (affine, size, parcel_ids) from the image_data_dict and saves
	the corresponding gifti/nifti files in a new folder
	data, and

	Parameters
	----------

	image_data_dict : dict
	dictionary whose data will be used to reconstruct nifti and
	gifti files when relevant

	output_folder : str
	path to folder that will be created to store the nifti/gifti files

	overwrite : bool
	whether to continue if the output_folder already exists

	"""

	if os.path.exists(output_folder):

		if overwrite == False:

			raise NameError('Error: folder already exists')

		else:

			os.mkdir(output_folder)


	if 'lh_data_inds' in image_data_dict.keys():

		lh_data = image_data_dict['data'][image_data_dict['lh_data_inds']]
		rh_data = image_data_dict['data'][image_data_dict['rh_data_inds']]

		lh_gifti_data = np.zeros(image_data_dict['lh_gifti_shape'])
		rh_gifti_data = np.zeros(image_data_dict['rh_gifti_shape'])

		#Unparcellate the data (this only works because dictionaries
		#are now ordered in python....)
		if 'lh_parcels_dict' in image_data_dict.keys():

			i = 0
			for parcel, inds in image_data_dict['lh_parcels_dict']:
				lh_gifti_data[inds] = lh_data[i]
				i += 1

				i = 0
				for parcel, inds in image_data_dict['rh_parcels_dict']:
					rh_gifti_data[inds] = rh_data[i]
					i += 1

				else:

					lh_gifti_data[image_data_dict['lh_ids']] = lh_data
					rh_gifti_data[image_data_dict['rh_ids']] = rh_data


					lh_gifti_path = os.path.join(output_folder, 'lh.data.func.gii')
					rh_gifti_path = os.path.join(output_folder, 'rh.data.func.gii')
					gifti_utils.arr2gifti(lh_gifti_data, lh_gifti_path)
					gifti_utils.arr2gifti(rh_gifti_data, rh_gifti_path)

	if 'nifti_data_inds' in image_data_dict.keys():

		nifti_partial_data = image_data_dict['data'][image_data_dict['nifti_data_inds']]
		nifti_data = np.zeros(image_data_dict['nifti_shape'])

		#Unparcellate the data
		if 'nifti_parcels_dict' in image_data_dict.keys():

			i = 0
			for parcel, inds in image_data_dict['nifti_parcels_dict']:
				nifti_data[inds] = nifti_partial_data[i]
				i += 1

			else:

				nifti_data[image_data_dict['nifti_ids']] = nifti_partial_data

				nifti_path = os.path.join(output_folder, 'data.nii')
				nifti_utils.arr2nifti(nifti_data, image_data_dict['nifti_affine'], nifti_path)

	return
