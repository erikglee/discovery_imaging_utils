import pandas as pd
import numpy as np
import os
import glob
from discovery_imaging_utils import imaging_utils
from discovery_imaging_utils import nifti_utils
import json

#Run in this order:
#(1) file_paths_dict = generate_file_paths(....)
#(2) check if files present with all_file_paths_exist(...)
#(3) if files present, parc_ts_dict = populate_parc_dictionary(....)
#(4) then use save/load functions to store in directory structure


def generate_file_paths(lh_gii_path=None,
			lh_inclusion_mask_path=None, 
			lh_parcellation_path=None,
			nifti_path=None,
			nifti_inclusion_mask_path=None,
			nifti_parcellation_path,
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


    if type(lh_gii_data_path) == type(None):
	if type(nifti_data_path) == type(None):
	    raise NameError('Error: At minimum must specify lh_gii_data_path or nifti_data_path')

    
    path_dictionary = {}
    prefix = ''
    
    #Gifti related paths
    if type(lh_gii_data_path) != type(None):
        
	path_dictionary['lh_data_path'] = lh_gii_data_path
	path_dictionary['rh_data_path'] = lh_gii_data_path[:-10] + 'R.func.gii'

	#For finding aroma/nuisance files
	prefix = '_'.join(lh_gii_data_path.split('_')[:-2])

    #Paths for potential surface inclusion masks
    if type(lh_inclusion_mask_path) != type(None):

	path_dictionary['lh_inclusion_mask_path'] = lh_inclusion_mask_path
	path_dictionary['rh_inclusion_mask_path'] = 'rh' + lh_inclusion_mask_path[2:]

    #Paths for potential surface inclusion masks
    if type(lh_parcellation_path) != type(None):

	path_dictionary['lh_parcellation_path'] = lh_parcellation_path
	path_dictionary['rh_parcellation_path'] = 'rh' + lh_parcellation_path[2:]



    
    #Nifti related paths
    if type(nifti_data_path) != type(None):
            
        path_dictionary['nifti_data_path'] = nifti_data_path
        
        if prefix != '':
            if '_'.join(nifti_data_path.split('_')[:-3]) != prefix:
                raise NameError('Error: It doesnt look like the nifti and gifti timeseries point to the same run')
        
        prefix = '_'.join(nifti_data_path.split('_')[:-3])

    if type(nifti_inclusion_mask_path) != type(None):

	path_dictionary['nifti_inclusion_mask_path'] = nifti_inclusion_mask_path
     
    
    #Aroma related paths
    if aroma_included:
        path_dictionary['melodic_mixing_path'] = prefix + '_desc-MELODIC_mixing.tsv'
        path_dictionary['aroma_noise_ics_path'] = prefix + '_AROMAnoiseICs.csv'

    #Confounds path  
    path_dictionary['confounds_regressors_path'] = prefix + '_desc-confounds_regressors.tsv'

    
    return path_dictionary
        

def populate_image_data_dictionary(file_path_dictionary, TR):
    """Takes a file_path_dictionary and uses it to populate an image_data_dictionary

    Takes a file_path_dictionary generated by, generate_file_paths, and creates an
    image_data_dictionary.


    Parameters
    ----------

    file_path_dictionary : dict
	dictionary created by generate_file_paths

    TR : float
	the repitition time in seconds


    Returns
    -------

    image_data_dictionary : dict
	.....input details






    """
    
    image_data_dictionary = {}
    has_gifti = False
    has_nifti = False
    
    ##############################################################################
    #Load the timeseries data and apply parcellation, saving also the parcel labels
    
    if 'lh_func_path' in file_path_dictionary.keys():
    
        has_gifti = True
        lh_data = imaging_utils.load_gifti_func(file_path_dictionary['lh_func_path'])
        rh_data = imaging_utils.load_gifti_func(file_path_dictionary['rh_func_path'])

	lh_inds = []
	rh_inds = []


	#NEED TO UPDATE THIS
	#NEED TO UPDATE THIS
	#NEED TO UPDATE THIS
        parc_ts_dictionary['surface_time_series'] = time_series_cortex
        parc_ts_dictionary['surface_median_ts_intensities'] = parc_median_signal_intensities

        
        
        
    if 'nifti_func_path' in file_path_dictionary.keys():
            
        
        has_nifti = True
        time_series_nifti, parcel_labels_nifti, parc_median_signal_intensities_nifti = nifti_utils.nifti_rois_to_time_signals(file_path_dictionary['nifti_func_path'], file_path_dictionary['nifti_parcellation_path'], demedian_before_averaging = True)

            
	#NEED TO UPDATE THIS
	#NEED TO UPDATE THIS
	nifti_inds = []
        parc_ts_dictionary['nifti_time_series'] = time_series_nifti
        parc_ts_dictionary['nifti_median_ts_intensities'] = parc_median_signal_intensities_nifti
        
        
    if (has_nifti and has_gifti):
        
        parc_ts_dictionary['time_series'] = np.vstack((parc_ts_dictionary['surface_time_series'],parc_ts_dictionary['nifti_time_series']))
        parc_ts_dictionary['labels'] = parc_ts_dictionary['surface_labels'] + parc_ts_dictionary['nifti_labels']
        parc_ts_dictionary['median_ts_intensities'] = np.hstack((parc_ts_dictionary['surface_median_ts_intensities'],parc_ts_dictionary['nifti_median_ts_intensities']))
    
    elif has_nifti:
        
        parc_ts_dictionary['time_series'] = parc_ts_dictionary['nifti_time_series']
        parc_ts_dictionary['labels'] = parc_ts_dictionary['nifti_labels']
        parc_ts_dictionary['median_ts_intensities'] = parc_ts_dictionary['nifti_median_ts_intensities']
        
    elif has_gifti:
        
        parc_ts_dictionary['time_series'] = parc_ts_dictionary['surface_time_series']
        parc_ts_dictionary['labels'] = parc_ts_dictionary['surface_labels']
        parc_ts_dictionary['median_ts_intensities'] = parc_ts_dictionary['surface_median_ts_intensities']
        
    else:
        
        raise NameError('Error: File Path Dictionary must either have gifti or nifti specified')
    
    #Set the TR
    parc_ts_dictionary['TR'] = TR
    

    ####################################################
    #Get the variables from the confounds regressors file
    #confound_df = pd.read_csv(self.confounds_regressors_path, sep='\t')
    #for (columnName, columnData) in confound_df.iteritems():
    #    setattr(self, columnName, columnData.as_matrix())
    parc_ts_dictionary['confounds'] = populate_confounds_dict(file_path_dictionary)
    
    parc_ts_dictionary['file_path_dictionary.json'] = file_path_dictionary
    
    ###################################################
    #Calculate general info, such as number of high motion timepoints
    #number of volumes that need to be skipped at the beginning of the
    #scan, etc.
    
    parc_ts_dictionary['general_info.json'] = populate_general_info_dict(parc_ts_dictionary)
    
    
    
    return parc_ts_dictionary
    

def populate_general_info_dict(parc_ts_dictionary):
    
    general_info_dict = {}
    
    ###################################################
    #Calculate the number of timepoints to skip at the beginning for this person.
    #If equal to zero, we will actually call it one so that we don't run into any
    #issues during denoising with derivatives
    general_info_dict['n_skip_vols'] = len(np.where(np.absolute(parc_ts_dictionary['confounds']['a_comp_cor_00']) < 0.00001)[0])
    if general_info_dict['n_skip_vols'] == 0:
        general_info_dict['n_skip_vols'] = 1
                
    general_info_dict['mean_fd'] = np.mean(parc_ts_dictionary['confounds']['framewise_displacement'][general_info_dict['n_skip_vols']:])
    general_info_dict['mean_dvars'] = np.mean(parc_ts_dictionary['confounds']['dvars'][general_info_dict['n_skip_vols']:])
    general_info_dict['num_std_dvars_above_1p5'] = len(np.where(parc_ts_dictionary['confounds']['std_dvars'][general_info_dict['n_skip_vols']:] > 1.5)[0])
    general_info_dict['num_std_dvars_above_1p3'] = len(np.where(parc_ts_dictionary['confounds']['std_dvars'][general_info_dict['n_skip_vols']:] > 1.3)[0])
    general_info_dict['num_std_dvars_above_1p2'] = len(np.where(parc_ts_dictionary['confounds']['std_dvars'][general_info_dict['n_skip_vols']:] > 1.2)[0])

    general_info_dict['num_fd_above_0p5_mm'] = len(np.where(parc_ts_dictionary['confounds']['framewise_displacement'][general_info_dict['n_skip_vols']:] > 0.5)[0])
    general_info_dict['num_fd_above_0p4_mm'] = len(np.where(parc_ts_dictionary['confounds']['framewise_displacement'][general_info_dict['n_skip_vols']:] > 0.4)[0])
    general_info_dict['num_fd_above_0p3_mm'] = len(np.where(parc_ts_dictionary['confounds']['framewise_displacement'][general_info_dict['n_skip_vols']:] > 0.3)[0])
    general_info_dict['num_fd_above_0p2_mm'] = len(np.where(parc_ts_dictionary['confounds']['framewise_displacement'][general_info_dict['n_skip_vols']:] > 0.2)[0])
    general_info_dict['labels'] = parc_ts_dictionary['labels']

    
    #Set TR
    general_info_dict['TR'] = parc_ts_dictionary['TR']
        
    #Find session/subject names
    if 'lh_func_path' in parc_ts_dictionary['file_path_dictionary.json'].keys():
    
        temp_path = parc_ts_dictionary['file_path_dictionary.json']['lh_func_path'].split('/')[-1]
        split_end_path = temp_path.split('_')
        
    else:
        
        temp_path = parc_ts_dictionary['file_path_dictionary.json']['nifti_func_path'].split('/')[-1]
        split_end_path = temp_path.split('_')
        
    general_info_dict['subject'] = split_end_path[0]

    if split_end_path[1][0:3] == 'ses':
        general_info_dict['session'] = split_end_path[1]
    else:
        general_info_dict['session'] = []
        
    return general_info_dict
    
    
    
def populate_confounds_dict(file_path_dictionary):
    
        confounds_dictionary = {}
    
        confounds_regressors_tsv_path = file_path_dictionary['confounds_regressors_path']
        confound_df = pd.read_csv(confounds_regressors_tsv_path, sep='\t')
        for (columnName, columnData) in confound_df.iteritems():
            confounds_dictionary[columnName] = columnData.as_matrix()
            
        
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
            print(file_path_dictionary[temp_field])
            files_present = False
            
    return files_present
