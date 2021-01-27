import numpy as np
import os
import nibabel as nib
from discovery_imaging_utils import nifti_utils
from discovery_imaging_utils import gifti_utils
from discovery_imaging_utils.dictionary_utils import general
import json
import h5py
from subprocess import run
import shutil



def convert_hdf5_to_images(hdf5_file_path, output_folder, overwrite = False):

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


	with h5py.File(hdf5_file_path, 'r') as f:

		#Make LH gifti if necessary
		if 'lh_data_inds' in f.keys():

			lh_data = f['data'][f['lh_data_inds'][:]]

			if lh_data.ndim == 1:
				dim = 1
			else:
				dim = lh_data.shape[1]

			lh_gifti_data = np.zeros((f['ids/lh_ids'].attrs['lh_gifti_shape'][0], dim))


			if 'parcel_names' in f['ids/lh_ids'].attrs.keys():

				i = 0
				for parcel, inds in f['ids/lh_ids'].items():

					lh_gifti_data[tuple(inds)] = lh_data[i]
					i += 1

			else:

				lh_gifti_data[f['ids/lh_ids'],...] = lh_data

			lh_gifti_path = os.path.join(output_folder, 'lh.data.func.gii')
			gifti_utils.arr2gifti(lh_gifti_data, lh_gifti_path)


		#Make RH gifti if necessary
		if 'rh_data_inds' in f.keys():

			rh_data = f['data'][f['rh_data_inds'][:]]
			rh_gifti_data = np.zeros((f['ids/rh_ids'].attrs['rh_gifti_shape'][0], dim))

			if rh_data.ndim == 1:
				dim = 1
			else:
				dim = rh_data.shape[1]


			if 'parcel_names' in f['ids/rh_ids'].attrs.keys():

				i = 0
				for parcel, inds in f['ids/rh_ids'].items():
					rh_gifti_data[tuple(inds)] = rh_data[i]
					i += 1

			else:

				rh_gifti_data[f['ids/rh_ids'],...] = rh_data

			rh_gifti_path = os.path.join(output_folder, 'rh.data.func.gii')
			gifti_utils.arr2gifti(rh_gifti_data, rh_gifti_path)


		if 'nifti_data_inds' in f.keys():

			print('Warning Nifti part of this function needs adjustment...')
			nifti_partial_data = f['data'][f['nifti_data_inds'][:]]
			nifti_data = np.zeros((f['ids/nifti_ids'].attrs['nifti_shape']))

			#Unparcellate the data
			if 'nifti_parcels_dict' in f['ids/nifti_ids'].attrs.keys():

				i = 0
				for parcel, inds in f['ids/nifti_ids'].items():
					nifti_data[inds] = nifti_partial_data[i]
					i += 1

			else:

				if nifti_partial_data.shape[1] == 1:
					nifti_data[f['ids/nifti_ids']] = np.squeeze(nifti_partial_data)
				else:
					x = f['ids/nifti_ids'][0]
					y = f['ids/nifti_ids'][1]
					z = f['ids/nifti_ids'][2]
					nifti_data[x,y,z,:] = nifti_partial_data

			nifti_path = os.path.join(output_folder, 'data.nii.gz')
			nifti_utils.arr2nifti(nifti_data, f['ids/nifti_ids'].attrs['nifti_affine'], nifti_path)

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
			normalize_within_parcels = False,
			normalize_within_dataset = True,
			overwrite = False,
			repack = True,
			repack_path = None,
			max_verts_per_chunk = 1000):
	"""Function that creates an HDF5 file to store neuroimaging data

	The purpose of this function is to create an hdf5 file referred to as an
	'image_data_dictionary' that can be used to facilitate easy use of neuroimaging
	data for different analyses. With details outlined below, this structure
	supports scalar (i.e. anatomical), or vector (i.e. functional) data, in
	both volumetric and surface spaces, allowing for sparse representation (i.e.
	excluding a brain mask), and mixed parcellation schemes (i.e. you could
	parcellate one hemisphere but not the other if desired). The main element
	of the created HDF5 file is its dataset 'data' that contains combined data
	from all input sources. An important feature of this data structure, is that
	all information required to convert the data back into original form is
	contained within the HDF5, such that after future data manipulations the
	resulting data structure can easily be converted back to nifti/gifti files
	so that the results can be viewed with standard neuroimaging visualization
	packages.

	This function can take 1-3 sources, constituting nifti, lh gifti, and rh
	gifti data. In simplest form, if paths to all three possible source types
	are given, then this function will create a hdf5 file with a 'data' dataset
	that contains the combined data from all sources, with shape
	<n_regions, n_timepoints> (in vector case)

	'data' will be a dataset that containing data from each data source. The
	elements from each data source can be identified by using 'lh_ids', 'rh_ids',
	and 'nifti_ids'. Such that idd['data'][idd['lh_ids'],:] would return all
	the data elements corresponding to the left hemisphere.

	Each source will also have its own data_inds dataset such as 'lh_data_inds'.
	The data_inds datasets show what each element of 'data' is in the original
	source format (i.e. which voxel, vertex, or parcel depending on usage). In
	the most basic case this will give one index per location in the data source.
	If a inclusion mask is incorporated, this will only index desired voxels/
	vertices. If a parcellation is included, this will represent the names of
	the different parcels/ROIs.

	Metadata that is associated with a specific source (i.e. lh parcellation path)
	can be found as attributes under the sources *_ids dataset. Source agnostic
	metadata, such as whether or not normalization was used, can be found as
	attrbutes under the dummy dataset 'general_idd_metadata'.



	Parameters
	----------

	hdf5_file_path : dict
		dictionary created by generate_file_paths. Must have at a minimum entries
		with paths for

	TR : float
		the repitition time in seconds

	normalize_within_parcels : bool, defaults to False
		if true, and if parcels are provided, and if data has temporal dimension,
		then voxels/vertices contributing to parcels/rois will be temporally
		demeaned before averaging, AND then the average voxel/vertex intensity
		will be inserted back into the time course.
	normalize_within_dataset : bool, defaults to True
		whether data should be normalized to have a global mean of 1k. Currently
		only supported for data with temporal dimension.



	"""

	#HOW TO SET ATTRIBUTES REMINDER:
	#dataset.attrs['name_of_desired_attribute'] = attribute

	if overwrite == True:
		if os.path.exists(hdf5_file_path):
			os.remove(hdf5_file_path)

	with h5py.File(hdf5_file_path, 'w') as f:


		file_path_dictionary = {}
		lh_metadata_dict = {}
		rh_metadata_dict = {}
		nifti_metadata_dict = {}
		image_data_dict = {}

		if type(lh_gii_data_path) != type(None):
			print('Found LH Gifti Input')
			file_path_dictionary['lh_gii_data_path'] = lh_gii_data_path
			lh_metadata_dict['lh_gii_data_path'] = lh_gii_data_path
			if type(lh_inclusion_mask_path) != type(None):
				file_path_dictionary['lh_inclusion_mask_path'] = lh_inclusion_mask_path
				lh_metadata_dict['lh_inclusion_mask_path'] = lh_inclusion_mask_path
			if type(lh_parcellation_path) != type(None):
				file_path_dictionary['lh_parcellation_path'] = lh_parcellation_path
				lh_metadata_dict['lh_parcellation_path'] = lh_parcellation_path
		if type(rh_gii_data_path) != type(None):
			print('Found RH Gifti Input')
			file_path_dictionary['rh_gii_data_path'] = rh_gii_data_path
			rh_metadata_dict['rh_gii_data_path'] = rh_gii_data_path
			if type(rh_inclusion_mask_path) != type(None):
				file_path_dictionary['rh_inclusion_mask_path'] = rh_inclusion_mask_path
				rh_metadata_dict['rh_inclusion_mask_path'] = rh_inclusion_mask_path
			if type(rh_parcellation_path) != type(None):
				file_path_dictionary['rh_parcellation_path'] = rh_parcellation_path
				rh_metadata_dict['rh_parcellation_path'] = rh_parcellation_path
		if type(nifti_data_path) != type(None):
			print('Found Nifti Input')
			file_path_dictionary['nifti_data_path'] = nifti_data_path
			nifti_metadata_dict['nifti_data_path'] = nifti_data_path
			if type(nifti_inclusion_mask_path) != type(None):
				file_path_dictionary['nifti_inclusion_mask_path'] = nifti_inclusion_mask_path
				nifti_metadata_dict['nifti_inclusion_mask_path'] = nifti_inclusion_mask_path
			if type(nifti_parcellation_path) != type(None):
				file_path_dictionary['nifti_parcellation_path'] = nifti_parcellation_path
				nifti_metadata_dict['nifti_parcellation_path'] = nifti_parcellation_path



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

			#UPDATE THESE ITEMS WITH READ_DIRECT FUNCTION TO
			#LOWER MEMORY USAGE!!!!!

			has_lh_gifti = True
			gifti_utils.load_gifti_func_to_hdf5(lh_gii_data_path, f, 'lh_data', num_verts_in_chunk = 1000, compression = None)
			#f['lh_data'] = gifti_utils.load_gifti_func(lh_gii_data_path)
			lh_metadata_dict['lh_gifti_shape'] = f['lh_data'].shape
			lh_gifti_ids = np.arange(0, f['lh_data'].shape[0], 1, dtype=int)


			#If inclusion mask is specified, set zero values in inclusion
			#mask to NaN.. the _parcellate_gifti function knows how to handle this
			if 'lh_inclusion_mask_path' in file_path_dictionary.keys():

				#Make function to set appropriate values to NaN.....
				f['lh_data_masked'], lh_inclusion_inds = gifti_utils.incorporate_gifti_inclusion_mask(f['lh_data'], lh_inclusion_mask_path)
				lh_gifti_ids = lh_inclusion_inds

				del f['lh_data']
				f.flush()
				f.create_dataset('lh_data', data = f['lh_data_masked'], compression = 'gzip')
				del f['lh_data_masked']



			if 'lh_parcellation_path' in file_path_dictionary.keys():

				has_lh_gifti_parcellation = True
				#Need to (1) parcellate data, (2) return parcel labels, (3) save info to recreate parcels
				f['lh_data_masked'], lh_labels, lh_parcels_dict = gifti_utils.parcellate_gifti(f['lh_data'], lh_parcellation_path, intensity_normalize_before_averaging = normalize_within_parcels)
				lh_gifti_ids = lh_labels
				#image_data_dict['lh_parcels_dict'] = lh_parcels_dict

				del f['lh_data']
				f.flush()
				f.create_dataset('lh_data', data = f['lh_data_masked'], compression = 'gzip')
				del f['lh_data_masked']

			#if has_lh_gifti_parcellation:
			#	_dict_to_hdf5_subdatasets(f, lh_parcels_dict, '/ids/lh')

			print('Finished Loading LH Data')


		#If there is rh surface data
		if 'rh_gii_data_path' in file_path_dictionary.keys():

			has_rh_gifti = True
			gifti_utils.load_gifti_func_to_hdf5(rh_gii_data_path, f, 'rh_data', num_verts_in_chunk = 1000, compression = None)
			#f['rh_data'] = gifti_utils.load_gifti_func(rh_gii_data_path)
			rh_metadata_dict['rh_gifti_shape'] = f['rh_data'].shape
			rh_gifti_ids = np.arange(0, f['rh_data'].shape[0], 1, dtype=int)


			#If inclusion mask is specified, set zero values in inclusion
			#mask to NaN.. the _parcellate_gifti function knows how to handle this
			if 'rh_inclusion_mask_path' in file_path_dictionary.keys():

				#Make function to set appropriate values to NaN.....
				f['rh_data_masked'], rh_inclusion_inds = gifti_utils.incorporate_gifti_inclusion_mask(f['rh_data'], rh_inclusion_mask_path)
				rh_gifti_ids = rh_inclusion_inds

				del f['rh_data']
				f.flush()
				f.create_dataset('rh_data', data = f['rh_data_masked'], compression = 'gzip')
				del f['rh_data_masked']



			if 'rh_parcellation_path' in file_path_dictionary.keys():

				has_rh_gifti_parcellation = True
				#Need to (1) parcellate data, (2) return parcel labels, (3) save info to recreate parcels
				f['rh_data_masked'], rh_labels, rh_parcels_dict = gifti_utils.parcellate_gifti(f['rh_data'], rh_parcellation_path, intensity_normalize_before_averaging = normalize_within_parcels)
				rh_gifti_ids = rh_labels
				#image_data_dict['rh_parcels_dict'] = rh_parcels_dict

				del f['rh_data']
				f.flush()
				f.create_dataset('rh_data', data = f['rh_data_masked'], compression = 'gzip')
				del f['rh_data_masked']


			#if has_rh_gifti_parcellation:
			#	_dict_to_hdf5_subdatasets(f, rh_parcels_dict, '/ids/rh')
			print('Finished Loading RH Data')

		#If there is nifti data
		if 'nifti_data_path' in file_path_dictionary.keys():

			#STILL NEED TO ADD COMPRESSION FOR IF PARCELLATION
			#ISNT USED (OR BRAIN MASK) (ALSO FOR HEMIS)

			has_nifti = True
			nifti_img = nib.load(nifti_data_path)
			nifti_data = f.create_dataset("nifti_data", nifti_img.dataobj.shape, chunks = True)
			nifti_data[...] = nifti_img.dataobj[...] #Could also do this through dataobj but would be slower
			nifti_metadata_dict['nifti_affine'] = nifti_img.affine
			nifti_metadata_dict['nifti_shape'] = f['nifti_data'].shape

			#Find indices to map back to nifti image
			nifti_3d = np.zeros(nifti_metadata_dict['nifti_shape'][0:3])
			nifti_ids = np.where(nifti_3d != None)

			nifti_inclusion_inds = None

			#remove from memory after done using...
			del nifti_img
			del nifti_3d


			if 'nifti_inclusion_mask_path' in file_path_dictionary.keys():

				f['nifti_data_masked'], nifti_inclusion_inds = nifti_utils.incorporate_nifti_inclusion_mask(f['nifti_data'], nifti_inclusion_mask_path)
				nifti_ids = nifti_inclusion_inds

				del f['nifti_data']
				f.create_dataset('nifti_data', data = f['nifti_data_masked'], compression = 'gzip')
				del f['nifti_data_masked']


			if 'nifti_parcellation_path' in file_path_dictionary.keys():

				has_nifti_parcellation = True
				f['nifti_data_masked'], nifti_labels, nifti_parcels_dict = nifti_utils.parcellate_nifti(f['nifti_data'], nifti_parcellation_path, demedian_before_averaging = normalize_within_parcels)
				nifti_ids = nifti_labels

				del f['nifti_data']
				f.flush()
				f.create_dataset('nifti_data', data = f['nifti_data_masked'], compression = 'gzip')
				del f['nifti_data_masked']

			#if has_nifti_parcellation:
			#	_dict_to_hdf5_subdatasets(f, nifti_parcels_dict, '/ids/lh_ids')
			print('Finished Loading Nifti Data')



			#MOST TIME DEMAND HAPPENS HERE ONWARD!!!!

			#If the data hasn't already been brought down to 2d, then do that now
			if f['nifti_data'].ndim > 2:

				#Check what the final dimension should be
				if f['nifti_data'].ndim == 3:
					depth = 1
				else:
					depth = f['nifti_data'].shape[3]


					#FIX BELOW FOR HDF5!!!!!!!!
				if type(nifti_inclusion_inds) == type(None):

					reshaped_nifti_data = np.reshape(f['nifti_data'], (f['nifti_data'].shape[0]*f['nifti_data'].shape[1]*f['nifti_data'].shape[2], depth))

				else:

					reshaped_nifti_data = np.reshape(f['nifti_data'][nifti_inclusion_inds], (nifti_inclusion_inds[0].shape[0], depth))

				del f['nifti_data']
				f['nifti_data'] = reshaped_nifti_data

			print('Finished Reshaping Nifti Data')



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

		if max_verts_per_chunk > num_locations:
			max_verts_per_chunk = num_locations

		data_dataset = f.create_dataset('data', (num_locations, num_dimensions[0]), compression = 'gzip', chunks = (max_verts_per_chunk, num_dimensions[0]))

		#Add lh gifti data
		inds_counted = 0
		if has_lh_gifti:
			f['lh_data_inds'] = np.arange(0, len(lh_gifti_ids), 1, dtype=int)
			inds_counted = int(inds_counted + f['lh_data'].shape[0])

			#Update virtual source/layout
			data_dataset[0:inds_counted,:] = f['lh_data']
			del f['lh_data']

			print('Added LH data to HDF5')


		#Add rh gifti data
		if has_rh_gifti:
			if inds_counted > 0:
				f['rh_data_inds'] = np.arange(inds_counted, inds_counted + len(rh_gifti_ids), 1, dtype=int)
			else:
				f['rh_data_inds'] = np.arange(0, f['rh_data'].shape[0], 1, dtype=int)

			inds_counted = int(inds_counted + f['rh_data'].shape[0])

			#Update virtual source/layout
			data_dataset[f['rh_data_inds'][:],:] = f['rh_data']
			del f['rh_data']


			print('Added RH data to HDF5')



		#Add nifti data
		if has_nifti:
			f['nifti_data_inds'] = np.arange(inds_counted, inds_counted + f['nifti_data'].shape[0], 1, dtype=int)

			#THIS LINE OF CODE IS VERY SLOW AND
			#PROBABLY > 90% OF RUNTIME!!!!!!
			data_dataset[f['nifti_data_inds'][:],:] = f['nifti_data']

			print('Added Nifti data to HDF5')

		#Add all the different datasets to a new
		#virtual dataset
		ids_group = f.create_group('ids')
		if has_lh_gifti:
			if has_lh_gifti_parcellation:
				lh_id_group = f.create_group('/ids/lh_ids')
				lh_id_group.attrs['parcel_names'] = lh_gifti_ids
				general._dict_to_hdf5_subdatasets(f, lh_parcels_dict, '/ids/lh_ids')
			else:
				f['/ids/lh_ids'] = lh_gifti_ids

			#Add metadata
			lh_id_obj = f['/ids/lh_ids']
			general._dict_to_hdf5_attrs(lh_id_obj, lh_metadata_dict)

		if has_rh_gifti:
			if has_rh_gifti_parcellation:
				rh_id_group = f.create_group('/ids/rh_ids')
				rh_id_group.attrs['parcel_names'] = rh_gifti_ids
				general._dict_to_hdf5_subdatasets(f, rh_parcels_dict, '/ids/rh_ids')
			else:
				f['/ids/rh_ids'] = rh_gifti_ids

			#Add metadata
			rh_id_obj = f['/ids/rh_ids']
			general._dict_to_hdf5_attrs(rh_id_obj, rh_metadata_dict)

		if has_nifti:
			if has_nifti_parcellation:
				nifti_id_group = f.create_group('/ids/nifti_ids')
				nifti_id_group.attrs['parcel_names'] = nifti_ids
				general._dict_to_hdf5_subdatasets(f, nifti_parcels_dict, '/ids/nifti_ids')
			else:
				f['/ids/nifti_ids'] = np.asarray(nifti_ids)

			#Add metadata
			nifti_id_obj = f['/ids/nifti_ids']
			general._dict_to_hdf5_attrs(nifti_id_obj, nifti_metadata_dict)

		data = f['data']


		#Normalize data if necessary
		if data.shape[1] > 1:
			data_means = np.mean(data,axis=1)
			f['data_means'] = data_means

			if normalize_within_dataset == True:
				data = (data/np.nanmean(data_means))*1000


				print('Finished Normalizing Data Data')



		general._dict_to_hdf5_attrs(f['data'], file_path_dictionary)

		gim = f.create_dataset('general_idd_metadata', data = 1)
		gim.attrs['normalize_within_parcels'] = normalize_within_parcels
		gim.attrs['normalize_within_dataset'] = normalize_within_dataset
		gim.attrs['overwrite'] = overwrite
		gim.attrs['repack'] = repack,
		gim.attrs['max_verts_per_chunk'] = max_verts_per_chunk

		f.flush()


	#Optional - repack the data to make big savings on storage
	#since otherwise HDF5 will assume some intermediate attributes
	#that may be large are still in the HDF5 file....

	if type(repack_path) != type(None):

		completed_proc = run([repack_path, hdf5_file_path, hdf5_file_path + '_repack'])

		if completed_proc.returncode != 0:

			os.remove(hdf5_file_path + '_repack')
			raise NameError('Error: repacking the HDF5 file to save on space was not successful. Rerun with keyword argument repack set to False or add h5repack to your subprocess path. HDF5s repack tool can also be ran outside of the context of this script later to save on space.')
		else:
			shutil.move(hdf5_file_path + '_repack', hdf5_file_path)

	elif repack == True:

		try:
			completed_proc = run(['h5repack', hdf5_file_path, hdf5_file_path + '_repack'])
		except:
			raise NameError('Repacking of hdf5 file (to reduce hdf5 file size) failed. This presumably means h5repack is not on your shell path. To continue either (1) add h5repack to shell path, or (2) set repack = False')

		if completed_proc.returncode != 0:

			os.remove(hdf5_file_path + '_repack')
			raise NameError('Error: repacking the HDF5 file to save on space was not successful. Rerun with keyword argument repack set to False or add h5repack to your subprocess path. HDF5s repack tool can also be ran outside of the context of this script later to save on space.')
		else:
			shutil.move(hdf5_file_path + '_repack', hdf5_file_path)


	return


def hdf5_from_template(data_arr, existing_hdf5_path, new_hdf5_path, overwrite = False):
	'''Takes an hdf5 and makes a new one with new data array

	Currently does not repack the hdf5 file (which can help with storage
	efficiency)

	Parameters
	----------

	data_arr: numpy.ndarray
		Data array to insert into new hdf5

	existing_hdf5_path : str
		String to existing hdf5

	new_hdf5_path : str
		Path to hdf5 file to be created

	overwrite : bool, default False
		Whether to overwrite if new_hdf5_path already exists


	'''

	if os.path.exists(new_hdf5_path) and (overwrite == False):
		raise NameError('Error: HDF5 already exists at new_hdf5_path. Delete file or set overwrite to True')
	else:
		shutil.copy(existing_hdf5_path, new_hdf5_path)

	with h5py.File(new_hdf5_path, 'r+') as f:

		if f['data'].shape[0] != data_arr.shape[0]:
			raise NameError('Error: the first dimension of the data_arr must match with existing hdf5')
		del f['data']

		f['data'] = data_arr
		f.flush()

		return
