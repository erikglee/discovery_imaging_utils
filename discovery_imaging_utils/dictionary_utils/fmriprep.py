import pandas as pd
import numpy as np
import os
import glob
import json
import h5py
from discovery_imaging_utils.dictionary_utils import image_data
from discovery_imaging_utils.dictionary_utils import general

#Run in this order:
#(1) file_paths_dict = generate_file_paths(....)
#(2) check if files present with all_file_paths_exist(...)
#(3) if files present, parc_ts_dict = populate_parc_dictionary(....)
#(4) then use save/load functions to store in directory structure


def generate_paths(lh_gii_data_path=None,
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


def populate_hdf5(file_path_dictionary, hdf5_file_path, TR, normalize_within_parcels = False, normalize_within_dataset = True):
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


	if 'lh_gii_data_path' in file_path_dictionary.keys():
		lh_gii_data_path = file_path_dictionary['lh_gii_data_path']
	else:
		lh_gii_data_path = None
	if 'lh_inclusion_mask_path' in file_path_dictionary.keys():
		lh_inclusion_mask_path = file_path_dictionary['lh_inclusion_mask_path']
	else:
		lh_inclusion_mask_path = None
	if 'lh_parcellation_path' in file_path_dictionary.keys():
		lh_parcellation_path = file_path_dictionary['lh_parcellation_path']
	else:
		lh_parcellation_path = None

	if 'rh_gii_data_path' in file_path_dictionary.keys():
		rh_gii_data_path = file_path_dictionary['rh_gii_data_path']
	else:
		rh_gii_data_path = None
	if 'rh_inclusion_mask_path' in file_path_dictionary.keys():
		rh_inclusion_mask_path = file_path_dictionary['rh_inclusion_mask_path']
	else:
		rh_inclusion_mask_path = None
	if 'rh_parcellation_path' in file_path_dictionary.keys():
		rh_parcellation_path = file_path_dictionary['rh_parcellation_path']
	else:
		rh_parcellation_path = None

	if 'nifti_data_path' in file_path_dictionary.keys():
		nifti_data_path = file_path_dictionary['nifti_data_path']
	else:
		nifti_data_path = None
	if 'nifti_inclusion_mask_path' in file_path_dictionary.keys():
		nifti_inclusion_mask_path = file_path_dictionary['nifti_inclusion_mask_path']
	else:
		nifti_inclusion_mask_path = None
	if 'nifti_parcellation_path' in file_path_dictionary.keys():
		nifti_parcellation_path = file_path_dictionary['nifti_parcellation_path']
	else:
		nifti_parcellation_path = None


	image_data.populate_hdf5(hdf5_file_path,
							lh_gii_data_path=lh_gii_data_path,
							lh_inclusion_mask_path=lh_inclusion_mask_path,
							lh_parcellation_path=lh_parcellation_path,
							rh_gii_data_path=rh_gii_data_path,
							rh_inclusion_mask_path=rh_inclusion_mask_path,
							rh_parcellation_path=rh_parcellation_path,
							nifti_data_path=nifti_data_path,
							nifti_inclusion_mask_path=nifti_inclusion_mask_path,
							nifti_parcellation_path=nifti_parcellation_path,
							normalize_within_parcels = normalize_within_parcels,
							normalize_within_dataset = normalize_within_dataset)

	confounds_dict = _populate_confounds_dict(file_path_dictionary, aroma_used = aroma_used)
	general_info_dict = _populate_general_info_dict(confounds_dict, file_path_dictionary, TR)
	with h5py.File(hdf5_file_path, 'a') as f:

		metadata_obj = f.create_group('fmriprep_metadata')
		general._dict_to_hdf5_attrs(metadata_obj, general_info_dict)
		general._dict_to_hdf5_subdatasets(f, confounds_dict, '/fmriprep_metadata')

	return


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

	general_info_dict['run_id'] = temp_path[0:temp_path.find('_space')]

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
