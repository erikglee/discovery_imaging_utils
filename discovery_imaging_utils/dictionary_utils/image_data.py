import numpy as np
import os
import nibabel as nib
from discovery_imaging_utils import nifti_utils
from discovery_imaging_utils import gifti_utils
import json
import h5py



def populate(lh_gii_data_path=None,
			lh_inclusion_mask_path=None,
			lh_parcellation_path=None,
			rh_gii_data_path=None,
			rh_inclusion_mask_path=None,
			rh_parcellation_path=None,
			nifti_data_path=None,
			nifti_inclusion_mask_path=None,
			nifti_parcellation_path=None,
			normalize = True):
	"""Takes a file_path_dictionary and uses it to populate an image_data_dictionary

	Takes a file_path_dictionary generated by, generate_file_paths, and creates an
	image_data_dictionary.


	Parameters
	----------

	file_path_dictionary : dict
		dictionary created by generate_file_paths. Must have at a minimum entries
		with paths for

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


	file_path_dictionary = {}

	if type(lh_gii_data_path) != type(None):
		file_path_dictionary['lh_gii_data_path'] = lh_gii_data_path
	if type(lh_inclusion_mask_path) != type(None):
		file_path_dictionary['lh_inclusion_mask_path'] = lh_inclusion_mask_path
	if type(lh_parcellation_path) != type(None):
		file_path_dictionary['lh_parcellation_path'] = lh_parcellation_path
	if type(rh_gii_data_path) != type(None):
		file_path_dictionary['rh_gii_data_path'] = rh_gii_data_path
	if type(rh_inclusion_mask_path) != type(None):
		file_path_dictionary['rh_inclusion_mask_path'] = rh_inclusion_mask_path
	if type(rh_parcellation_path) != type(None):
		file_path_dictionary['rh_parcellation_path'] = rh_parcellation_path
	if type(nifti_data_path) != type(None):
		file_path_dictionary['nifti_data_path'] = nifti_data_path
	if type(nifti_inclusion_mask_path) != type(None):
		file_path_dictionary['nifti_inclusion_mask_path'] = nifti_inclusion_mask_path
	if type(nifti_parcellation_path) != type(None):
		file_path_dictionary['nifti_parcellation_path'] = nifti_parcellation_path



	image_data_dict = {}
	metadata_dict = {}
	metadata_dict['filepaths_dict'] = file_path_dictionary

	lh_data = None
	rh_data = None
	nifti_data = None

	has_lh_gifti = False
	has_rh_gifti = False
	has_lh_gifti_parcellation = False
	has_rh_gifti_parcellation = False
	has_nifti = False
	has_nifti_parcellation = False

	#If there is lh surface data
	if 'lh_gii_data_path' in file_path_dictionary.keys():

		has_lh_gifti = True
		lh_data = gifti_utils.load_gifti_func(file_path_dictionary['lh_gii_data_path'])
		image_data_dict['lh_gifti_shape'] = lh_data.shape
		lh_gifti_ids = np.arange(0, lh_data.shape[0], 1, dtype=int)


		#If inclusion mask is specified, set zero values in inclusion
		#mask to NaN.. the _parcellate_gifti function knows how to handle this
		if 'lh_inclusion_mask_path' in file_path_dictionary.keys():

			#Make function to set appropriate values to NaN.....
			lh_data, lh_inclusion_inds = gifti_utils.incorporate_gifti_inclusion_mask(lh_data, file_path_dictionary['lh_inclusion_mask_path'])
			lh_gifti_ids = lh_inclusion_inds



		if 'lh_parcellation_path' in file_path_dictionary.keys():

			has_lh_gifti_parcellation = True
			#Need to (1) parcellate data, (2) return parcel labels, (3) save info to recreate parcels
			lh_data, lh_labels, lh_parcels_dict = gifti_utils.parcellate_gifti(lh_data, file_path_dictionary['lh_parcellation_path'])
			lh_gifti_ids = lh_labels
			image_data_dict['lh_parcels_dict'] = lh_parcels_dict


	#If there is rh surface data
	if 'rh_gii_data_path' in file_path_dictionary.keys():

		has_rh_gifti = True
		rh_data = gifti_utils.load_gifti_func(file_path_dictionary['rh_gii_data_path'])
		image_data_dict['rh_gifti_shape'] = rh_data.shape
		rh_gifti_ids = np.arange(0, rh_data.shape[0], 1, dtype=int)


		#If inclusion mask is specified, set zero values in inclusion
		#mask to NaN.. the _parcellate_gifti function knows how to handle this
		if 'rh_inclusion_mask_path' in file_path_dictionary.keys():

			#Make function to set appropriate values to NaN.....
			rh_data, rh_inclusion_inds = gifti_utils.incorporate_gifti_inclusion_mask(rh_data, file_path_dictionary['rh_inclusion_mask_path'])
			rh_gifti_ids = rh_inclusion_inds



		if 'rh_parcellation_path' in file_path_dictionary.keys():

			has_rh_gifti_parcellation = True
			#Need to (1) parcellate data, (2) return parcel labels, (3) save info to recreate parcels
			rh_data, rh_labels, rh_parcels_dict = gifti_utils.parcellate_gifti(rh_data, file_path_dictionary['rh_parcellation_path'])
			rh_gifti_ids = rh_labels
			image_data_dict['rh_parcels_dict'] = rh_parcels_dict



	#If there is nifti data
	if 'nifti_data_path' in file_path_dictionary.keys():

		has_nifti = True
		nifti_img = nib.load(file_path_dictionary['nifti_data_path'])
		nifti_data = nifti_img.get_fdata()
		#Or instead nifti_data = nifti_img.dataobj
		if nifti_data.ndim > 3:
			nifti_3d = np.squeeze(nifti_data[:,:,:,0])
			nifti_ids = np.where(nifti_3d != None)
		else:
			nifti_ids = np.where(nifti_data != None)

		nifti_inclusion_inds = None
		image_data_dict['nifti_affine'] = nifti_img.affine
		image_data_dict['nifti_shape'] = nifti_data.shape

		#remove from memory after done using...
		del nifti_img


		if 'nifti_inclusion_mask_path' in file_path_dictionary.keys():

			nifti_data, nifti_inclusion_inds = nifti_utils.incorporate_nifti_inclusion_mask(nifti_data, file_path_dictionary['nifti_inclusion_mask_path'])
			nifti_ids = nifti_inclusion_inds

		if 'nifti_parcellation_path' in file_path_dictionary.keys():

			has_nifti_parcellation = True
			nifti_data, nifti_labels, nifti_parcels_dict = nifti_utils.parcellate_nifti(nifti_data, file_path_dictionary['nifti_parcellation_path'])
			nifti_ids = nifti_labels

			image_data_dict['nifti_parcels_dict'] = nifti_parcels_dict

		#If the data hasn't already been brought down to 2d, then do that now
		if nifti_data.ndim > 2:

			#Check what the final dimension should be
			if nifti_data.ndim == 3:
				depth = 1
			else:
				depth = nifti_data.shape[3]


			if type(nifti_inclusion_inds) == type(None):

				nifti_data = np.reshape(nifti_data, (nifti_data.shape[0]*nifti_data.shape[1]*nifti_data.shape[2], depth))

			else:

				nifti_data = np.reshape(nifti_data[nifti_inclusion_inds], (nifti_inclusion_inds[0].shape[0], depth))



	#FYI...
	#data inds will specify how to access lh/rh/nifti elements from the data
	#key, and the ids will alternatively specify what those inds mean in terms
	#of the lh/rh/nifti or parcellation schemes

	if (has_lh_gifti or has_rh_gifti or has_nifti) == False:
		raise NameError('Error: Must define path for at least one of the following - lh_gifti_data, rh_gifti_data, nifti_data')

	num_dimensions = []
	if type(lh_data) != type(None):
		num_dimensions.append(lh_data.shape[1])
	if type(rh_data) != type(None):
		num_dimensions.append(rh_data.shape[1])
	if type(nifti_data) != type(None):
		num_dimensions.append(nifti_data.shape[1])

	if np.unique(num_dimensions).shape[0] != 1:
		raise NameError('Error: LH, RH, and Nifti data must all have the same length.')

	data = None

	#Add lh gifti data
	if type(lh_data) != type(None):
		image_data_dict['lh_data_inds'] = np.arange(0, len(lh_gifti_ids), 1, dtype=int)
		data = lh_data

		image_data_dict['lh_ids'] = lh_gifti_ids


	#Add rh gifti data
	if type(rh_data) != type(None):
		if type(data) != type(None):
			rh_data_inds = np.arange(data.shape[0], data.shape[0] + len(rh_gifti_ids), 1, dtype=int)
			data = np.vstack((data, rh_data))
		else:
			rh_data_inds = np.arange(0, rh_data.shape[0], 1, dtype=int)
			data = rh_data

		image_data_dict['rh_data_inds'] = rh_data_inds
		image_data_dict['rh_ids'] = rh_gifti_ids




	#Add nifti data
	if type(nifti_data) != type(None):
		if type(data) != type(None):
			nifti_data_inds = np.arange(data.shape[0], data.shape[0] + nifti_data.shape[0], 1, dtype=int)
			data = np.vstack((data, nifti_data))
		else:
			nifti_data_inds = np.arange(0, nifti_data.shape[0], 1, dtype=int)
			data = nifti_data

		image_data_dict['nifti_data_inds'] = nifti_data_inds
		#image_data_dict['nifti_ids'] = ['nii_' + str(temp_label) for temp_label in nifti_data_inds]
		image_data_dict['nifti_ids'] = nifti_ids


	#Normaize data if necessary
	if normalize == True:
		if data.shape[1] > 1:

			image_data_dict['data_means'] = np.mean(data,axis=1)
			data = data/image_data_dict['data_means'][:,np.newaxis]*10000



	image_data_dict['data'] = data


	return image_data_dict




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

		os.makedirs(output_folder)


	#Make LH gifti if necessary
	if 'lh_data_inds' in image_data_dict.keys():

		lh_data = image_data_dict['data'][image_data_dict['lh_data_inds']]
		lh_gifti_data = np.zeros(image_data_dict['lh_gifti_shape'])


		#Unparcellate the data (this only works because dictionaries
		#are now ordered in python....)
		if 'lh_parcels_dict' in image_data_dict.keys():

			i = 0
			for parcel, inds in image_data_dict['lh_parcels_dict'].items():
				lh_gifti_data[inds] = lh_data[i]
				i += 1

		else:

			lh_gifti_data[image_data_dict['lh_ids']] = lh_data

		lh_gifti_path = os.path.join(output_folder, 'lh.data.func.gii')
		gifti_utils.arr2gifti(lh_gifti_data, lh_gifti_path)


	#Make RH gifti if necessary
	if 'rh_data_inds' in image_data_dict.keys():

		rh_data = image_data_dict['data'][image_data_dict['rh_data_inds']]
		rh_gifti_data = np.zeros(image_data_dict['rh_gifti_shape'])

		#Unparcellate the data (this only works because dictionaries
		#are now ordered in python....)
		if 'rh_parcels_dict' in image_data_dict.keys():

			i = 0
			for parcel, inds in image_data_dict['rh_parcels_dict'].items():
				rh_gifti_data[inds] = rh_data[i]
				i += 1

		else:

			rh_gifti_data[image_data_dict['rh_ids']] = rh_data

		rh_gifti_path = os.path.join(output_folder, 'rh.data.func.gii')
		gifti_utils.arr2gifti(rh_gifti_data, rh_gifti_path)


	if 'nifti_data_inds' in image_data_dict.keys():

		nifti_partial_data = image_data_dict['data'][image_data_dict['nifti_data_inds']]
		nifti_data = np.zeros(image_data_dict['nifti_shape'])

		#Unparcellate the data
		if 'nifti_parcels_dict' in image_data_dict.keys():

			i = 0
			for parcel, inds in image_data_dict['nifti_parcels_dict'].items():
				nifti_data[inds] = nifti_partial_data[i]
				i += 1

		else:

			if nifti_partial_data.shape[1] == 1:
				nifti_data[image_data_dict['nifti_ids']] = np.squeeze(nifti_partial_data)
			else:
				x = image_data_dict['nifti_ids'][0]
				y = image_data_dict['nifti_ids'][1]
				z = image_data_dict['nifti_ids'][2]
				nifti_data[x,y,z,:] = nifti_partial_data

		nifti_path = os.path.join(output_folder, 'data.nii.gz')
		nifti_utils.arr2nifti(nifti_data, image_data_dict['nifti_affine'], nifti_path)

	return


def populate_hdf5(hdf5_file_path,
			lh_gii_data_path=None,
			lh_inclusion_mask_path=None,
			lh_parcellation_path=None,
			rh_gii_data_path=None,
			rh_inclusion_mask_path=None,
			rh_parcellation_path=None,
			nifti_data_path=None,
			nifti_inclusion_mask_path=None,
			nifti_parcellation_path=None,
			normalize = True,
			overwrite = False):
	"""Takes a file_path_dictionary and uses it to populate an image_data_dictionary

	Takes a file_path_dictionary generated by, generate_file_paths, and creates an
	image_data_dictionary.


	Parameters
	----------

	file_path_dictionary : dict
		dictionary created by generate_file_paths. Must have at a minimum entries
		with paths for

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

	if overwrite == True:
		if os.path.exists(hdf5_file_path):
			os.remove(hdf5_file_path)

	with h5py.File(hdf5_file_path, 'w') as f:

		#Do stuff....


		file_path_dictionary = {}

		if type(lh_gii_data_path) != type(None):
			f['/metadata/file_paths/lh_gii_data_path'] = lh_gii_data_path
			file_path_dictionary['lh_gii_data_path'] = lh_gii_data_path
		if type(lh_inclusion_mask_path) != type(None):
			f['/metadata/file_paths/lh_inclusion_mask_path'] = lh_inclusion_mask_path
			file_path_dictionary['lh_inclusion_mask_path'] = lh_inclusion_mask_path
		if type(lh_parcellation_path) != type(None):
			f['/metadata/file_paths/lh_parcellation_path'] = lh_parcellation_path
			file_path_dictionary['lh_parcellation_path'] = lh_parcellation_path
		if type(rh_gii_data_path) != type(None):
			f['/metadata/file_paths/rh_gii_data_path'] = rh_gii_data_path
			file_path_dictionary['rh_gii_data_path'] = rh_gii_data_path
		if type(rh_inclusion_mask_path) != type(None):
			f['/metadata/file_paths/rh_inclusion_mask_path'] = rh_inclusion_mask_path
			file_path_dictionary['rh_inclusion_mask_path'] = rh_inclusion_mask_path
		if type(rh_parcellation_path) != type(None):
			f['/metadata/file_paths/rh_parcellation_path'] = rh_parcellation_path
			file_path_dictionary['rh_parcellation_path'] = rh_parcellation_path
		if type(nifti_data_path) != type(None):
			f['/metadata/file_paths/nifti_data_path'] = nifti_data_path
			file_path_dictionary['nifti_data_path'] = nifti_data_path
		if type(nifti_inclusion_mask_path) != type(None):
			f['/metadata/file_paths/nifti_inclusion_mask_path'] = nifti_inclusion_mask_path
			file_path_dictionary['nifti_inclusion_mask_path'] = nifti_inclusion_mask_path
		if type(nifti_parcellation_path) != type(None):
			f['/metadata/file_paths/nifti_parcellation_path'] = nifti_parcellation_path
			file_path_dictionary['nifti_parcellation_path'] = nifti_parcellation_path



		image_data_dict = {}
		metadata_dict = {}
		#metadata_dict['filepaths_dict'] = file_path_dictionary

		lh_data = None
		rh_data = None
		nifti_data = None

		has_lh_gifti = False
		has_rh_gifti = False
		has_lh_gifti_parcellation = False
		has_rh_gifti_parcellation = False
		has_nifti = False
		has_nifti_parcellation = False

		#If there is lh surface data
		if 'lh_gii_data_path' in file_path_dictionary.keys():

			has_lh_gifti = True
			f['lh_data'] = gifti_utils.load_gifti_func(lh_gii_data_path)
			f['/metadata/input_attributes/lh_gifti_shape'] = f['lh_data'].shape #or could put this as an attribute?
			lh_gifti_ids = np.arange(0, f['lh_data'].shape[0], 1, dtype=int)


			#If inclusion mask is specified, set zero values in inclusion
			#mask to NaN.. the _parcellate_gifti function knows how to handle this
			if 'lh_inclusion_mask_path' in file_path_dictionary.keys():

				#Make function to set appropriate values to NaN.....
				f['lh_data_masked'], lh_inclusion_inds = gifti_utils.incorporate_gifti_inclusion_mask(f['lh_data'], f['/metadata/file_paths/lh_inclusion_mask_path'])
				lh_gifti_ids = lh_inclusion_inds

				del f['lh_data']
				f['lh_data'] = f['lh_data_masked']
				del f['lh_data_masked']



			if 'lh_parcellation_path' in file_path_dictionary.keys():

				has_lh_gifti_parcellation = True
				#Need to (1) parcellate data, (2) return parcel labels, (3) save info to recreate parcels
				f['lh_data_masked'], lh_labels, lh_parcels_dict = gifti_utils.parcellate_gifti(f['lh_data'], f['/metadata/file_paths/lh_parcellation_path'])
				lh_gifti_ids = lh_labels
				_dict_to_hdf5_attrs(f, lh_parcels_dict, base_path = '/metadata/parcel_dicts/lh/')
				#image_data_dict['lh_parcels_dict'] = lh_parcels_dict

				del f['lh_data']
				f['lh_data'] = f['lh_data_masked']
				del f['lh_data_masked']

			f['lh_ids'] = lh_gifti_ids


		#If there is rh surface data
		if 'rh_gii_data_path' in file_path_dictionary.keys():

			has_rh_gifti = True
			f['rh_data'] = gifti_utils.load_gifti_func(rh_gii_data_path)
			image_data_dict['rh_gifti_shape'] = f['rh_data'].shape
			rh_gifti_ids = np.arange(0, f['rh_data'].shape[0], 1, dtype=int)


			#If inclusion mask is specified, set zero values in inclusion
			#mask to NaN.. the _parcellate_gifti function knows how to handle this
			if 'rh_inclusion_mask_path' in file_path_dictionary.keys():

				#Make function to set appropriate values to NaN.....
				f['rh_data_masked'], rh_inclusion_inds = gifti_utils.incorporate_gifti_inclusion_mask(f['rh_data'], f['/metadata/file_paths/rh_inclusion_mask_path'])
				rh_gifti_ids = rh_inclusion_inds

				del f['rh_data']
				f['rh_data'] = f['rh_data_masked']
				del f['rh_data_masked']



			if 'rh_parcellation_path' in file_path_dictionary.keys():

				has_rh_gifti_parcellation = True
				#Need to (1) parcellate data, (2) return parcel labels, (3) save info to recreate parcels
				f['rh_data_masked'], rh_labels, rh_parcels_dict = gifti_utils.parcellate_gifti(f['rh_data'], f['/metadata/file_paths/rh_parcellation_path'])
				rh_gifti_ids = rh_labels
				_dict_to_hdf5_attrs(f, rh_parcels_dict, base_path = '/metadata/parcel_dicts/rh/')
				#image_data_dict['rh_parcels_dict'] = rh_parcels_dict

				del f['rh_data']
				f['rh_data'] = f['rh_data_masked']
				del f['rh_data_masked']

			f['rh_ids'] = rh_gifti_ids

		#If there is nifti data
		if 'nifti_data_path' in file_path_dictionary.keys():

			has_nifti = True
			nifti_img = nib.load(f['/metadata/file_paths/nifti_data_path'])
			nifti_data = nifti_img.get_fdata() #This probably returns a dataset that acts as np array?
			if nifti_data.ndim > 3:
				nifti_3d = np.squeeze(nifti_data[:,:,:,0])
				nifti_ids = np.where(nifti_3d != None)
			else:
				nifti_ids = np.where(nifti_data != None)

			nifti_inclusion_inds = None
			f['/metadata/input_attributes/nifti_affine'] = nifti_img.affine
			f['/metadata/input_attributes/nifti_shape'] = nifti_data.shape

			#remove from memory after done using...
			del nifti_img


			if 'nifti_inclusion_mask_path' in file_path_dictionary.keys():

				nifti_data, nifti_inclusion_inds = nifti_utils.incorporate_nifti_inclusion_mask(nifti_data, f['/metadata/file_paths/nifti_inclusion_mask_path'])
				nifti_ids = nifti_inclusion_inds

			if 'nifti_parcellation_path' in file_path_dictionary.keys():

				has_nifti_parcellation = True
				nifti_data, nifti_labels, nifti_parcels_dict = nifti_utils.parcellate_nifti(nifti_data, f['/metadata/file_paths/nifti_parcellation_path'])
				nifti_ids = nifti_labels

				_dict_to_hdf5_attrs(f, nifti_parcels_dict, base_path = '/metadata/parcel_dicts/nifti/')

			f['nifti_ids'] = nifti_ids



			#If the data hasn't already been brought down to 2d, then do that now
			if nifti_data.ndim > 2:

				#Check what the final dimension should be
				if nifti_data.ndim == 3:
					depth = 1
				else:
					depth = nifti_data.shape[3]


				if type(nifti_inclusion_inds) == type(None):

					nifti_data = np.reshape(nifti_data, (nifti_data.shape[0]*nifti_data.shape[1]*nifti_data.shape[2], depth))

				else:

					nifti_data = np.reshape(nifti_data[nifti_inclusion_inds], (nifti_inclusion_inds[0].shape[0], depth))



		#FYI...
		#data inds will specify how to access lh/rh/nifti elements from the data
		#key, and the ids will alternatively specify what those inds mean in terms
		#of the lh/rh/nifti or parcellation schemes

		if (has_lh_gifti or has_rh_gifti or has_nifti) == False:
			raise NameError('Error: Must define path for at least one of the following - lh_gifti_data, rh_gifti_data, nifti_data')

		num_dimensions = []
		num_locations = 0
		if has_lh_gifti:
			num_dimensions.append(f['lh_data'].shape[1])
			num_locations += f['lh_data'].shape[0]
		if has_rh_gifti:
			num_dimensions.append(f['rh_data'].shape[1])
			num_locations += f['rh_data'].shape[0]
		if has_nifti:
			num_dimensions.append(f['nifti_data'].shape[1])
			num_locations += f['nifti_data'].shape[0]

		if np.unique(num_dimensions).shape[0] != 1:
			print(f['lh_data'].shape)
			raise NameError('Error: LH, RH, and Nifti data must all have the same length.')

		data = f.create_dataset('data', (int(num_locations), int(num_dimensions[0])), dtype=np.float32)

		#Add lh gifti data
		inds_counted = 0
		if has_lh_gifti:
			f['lh_data_inds'] = np.arange(0, len(lh_gifti_ids), 1, dtype=int)
			data[inds_counted:(inds_counted + f['lh_data'].shape[0]),:] = lh_data
			inds_counted = int(inds_counted + f['lh_data'].shape[0])

			image_data_dict['lh_ids'] = lh_gifti_ids


		#Add rh gifti data
		if has_rh_gifti:
			if inds_counted > 0:
				image_data_dict['rh_data_inds'] = np.arange(inds_counted, inds_counted + len(rh_gifti_ids), 1, dtype=int)
				data[inds_counted:(inds_counted + f['rh_data'].shape[0]),:] = rh_data
			else:
				image_data_dict['rh_data_inds'] = np.arange(0, rh_data.shape[0], 1, dtype=int)
				data[0:f['rh_data'].shape[0],:] = rh_data

			image_data_dict['rh_ids'] = rh_gifti_ids
			inds_counted = int(inds_counted + f['rh_data'].shape[0])




		#Add nifti data
		if has_nifti:
			if inds_counted > 0:
				nifti_data_inds = np.arange(inds_counted, inds_counted + nifti_data.shape[0], 1, dtype=int)
				data[inds_counted:(inds_counted + f['nifti_data'].shape[0]),:] = nifti_data
			else:
				nifti_data_inds = np.arange(0, nifti_data.shape[0], 1, dtype=int)
				data = nifti_data

			inds_counted = int(inds_counted + f['nifti_data'].shape[0])
			f['nifti_data_inds'] = nifti_data_inds
			image_data_dict['nifti_ids'] = nifti_ids


		#Normaize data if necessary
		if normalize == True:
			if data.shape[1] > 1:

				f['data_means'] = np.mean(data,axis=1)
				data = data/f['data_means'][:,np.newaxis]*10000




	return


def _dict_to_hdf5_attrs(hdf5_file_object, dictionary, base_path = ''):
	"""Function adds dictionary to hdf5 file

	Function takes a loaded HDF5 file, and adds dictionary
	key/value pairs as new datasets under the desired base
	path (so you can have dictionary be put into sub groups
	instead of main group, otherwise leave base_path empty).

	Parameters
	----------

	hdf5_file_object : hdf5
		hdf5 file loaded from h5py
	dictionary : dict
		dictionary to be added to hdf5 file as attributes
	base_path : str, optional
		optional if you want to have dictionary nested within
		an hdf5 group

	"""

	for key, value in dictionary.items():

		hdf5_file_object[os.path.join(base_path,key)] = value

	return
