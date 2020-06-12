import pandas as pd
import numpy as np
import os
import glob
from eriks_packages import imaging_utils
from eriks_packages import nifti_utils
import json

#Run in this order:
#(1) file_paths_dict = generate_file_paths(....)
#(2) check if files present with all_file_paths_exist(...)
#(3) if files present, parc_ts_dict = populate_parc_dictionary(....)
#(4) then use save/load functions to store in directory structure




def all_file_paths_exist(file_path_dictionary):
    
    #Check if all files exist, and if they don't
    #return False
    files_present = True
    
    for temp_field in file_path_dictionary:
        
        if os.path.exists(file_path_dictionary[temp_field]) == False:
            print(file_path_dictionary[temp_field])
            files_present = False
            
    return files_present

def generate_file_paths(lh_gii_path=None, lh_parcellation_path=None, nifti_ts_path=None,
                         nifti_parcellation_path=None, aroma_included=True):
    
    #This function gathers paths useful for imaging analyses... either a gifti or nifti
    #or both must be specified along with accompanying parcellation paths, outputs a dictionary
    #with relevant paths, output of this function can be used with "all_file_paths_exist", then
    #with "populate_parc_dictionary", then can be saved with "save_dictionary"
    
    path_dictionary = {}
    prefix = ''
    
    #Gifti related paths
    if type(lh_gii_path) != type(None):
        
        if type(lh_parcellation_path) == type(None):
            
            raise NameError('Error: need to specify lh parcellation path if you specify a lh_gii_path')
        
        path_dictionary['lh_func_path'] = lh_gii_path
        path_dictionary['rh_func_path'] = lh_gii_path[:-10] + 'R.func.gii'
        path_dictionary['lh_parcellation_path'] = lh_parcellation_path
        
        lh_parcel_end = lh_parcellation_path.split('/')[-1]
    
        path_dictionary['rh_parcellation_path'] = lh_parcellation_path[:-len(lh_parcel_end)] + 'r' + lh_parcellation_path.split('/')[-1][1:]

        #For finding aroma/nuisance files
        prefix = '_'.join(lh_gii_path.split('_')[:-2])
    
    #Nifti related paths
    if type(nifti_ts_path) != type(None):
        
        if type(nifti_parcellation_path) == type(None):
            
            raise NameError('Error: need to specify nifti parcellation path if you specify a nifti_ts_path')
            
        path_dictionary['nifti_func_path'] = nifti_ts_path
        path_dictionary['nifti_parcellation_path'] = nifti_parcellation_path
        
        nifti_parcellation_json = nifti_parcellation_path.split('.')[0] + '.json'
        if os.path.exists(nifti_parcellation_json) == True:
            path_dictionary['nifti_parcellation_json_path'] = nifti_parcellation_json
        
        if prefix != '':
            if '_'.join(nifti_ts_path.split('_')[:-3]) != prefix:
                raise NameError('Error: It doesnt look like the nifti and gifti timeseries point to the same run')
        
        prefix = '_'.join(nifti_ts_path.split('_')[:-3])
     
    
    #Aroma related paths
    if aroma_included:
        path_dictionary['melodic_mixing_path'] = prefix + '_desc-MELODIC_mixing.tsv'
        path_dictionary['aroma_noise_ics_path'] = prefix + '_AROMAnoiseICs.csv'

    #Confounds path  
    path_dictionary['confounds_regressors_path'] = prefix + '_desc-confounds_regressors.tsv'
    
    return path_dictionary
        

def populate_parc_dictionary(file_path_dictionary, TR):
    
    parc_ts_dictionary = {}
    has_gifti = False
    has_nifti = False
    
    ##############################################################################
    #Load the timeseries data and apply parcellation, saving also the parcel labels
    
    if 'lh_func_path' in file_path_dictionary.keys():
    
        has_gifti = True
        lh_data = imaging_utils.load_gifti_func(file_path_dictionary['lh_func_path'])
        rh_data = imaging_utils.load_gifti_func(file_path_dictionary['rh_func_path'])
        time_series_cortex, parcel_labels_cortex, parc_median_signal_intensities = imaging_utils.demedian_parcellate_func_combine_hemis(lh_data, rh_data, file_path_dictionary['lh_parcellation_path'], file_path_dictionary['rh_parcellation_path'])

        parc_ts_dictionary['labels_surface'] = parcel_labels_cortex
        parc_ts_dictionary['time_series_surface'] = time_series_cortex
        parc_ts_dictionary['median_ts_intensities_surface'] = parc_median_signal_intensities
        
        
        
    if 'nifti_func_path' in file_path_dictionary.keys():
            
        
        has_nifti = True
        time_series_nifti, parcel_labels_nifti, parc_median_signal_intensities_nifti = nifti_utils.nifti_rois_to_time_signals(file_path_dictionary['nifti_func_path'], file_path_dictionary['nifti_parcellation_path'], demedian_before_averaging = True)

        
        if 'nifti_parcellation_json_path' in file_path_dictionary.keys():
            
            with open(file_path_dictionary['nifti_parcellation_json_path'], 'r') as temp_file:
                
                json_contents = temp_file.read()
                json_dict = json.loads(json_contents)
            
            
            parc_ts_dictionary['nifti_parcellation_info.json'] = json_dict
            
            output_labels = []
            for temp_label in parcel_labels_nifti:
            
                strint_label = str(int(temp_label))
                
                if 'unique_region_identifier' in json_dict[strint_label].keys():
                    
                    output_labels.append(json_dict[strint_label]['unique_region_identifier'])
                    
                else:
                    
                    output_labels.append(strint_label)
                    
            parcel_labels_nifti = output_labels
            
        parc_ts_dictionary['labels_nifti'] = parcel_labels_nifti
        parc_ts_dictionary['time_series_nifti'] = time_series_nifti
        parc_ts_dictionary['median_ts_intensities_nifti'] = parc_median_signal_intensities_nifti
        
        
    if (has_nifti and has_gifti):
        
        parc_ts_dictionary['time_series'] = np.vstack((parc_ts_dictionary['time_series_surface'],parc_ts_dictionary['time_series_nifti']))
        parc_ts_dictionary['labels'] = parc_ts_dictionary['labels_surface'] + parc_ts_dictionary['labels_nifti']
        parc_ts_dictionary['median_ts_intensities'] = np.hstack((parc_ts_dictionary['median_ts_intensities_surface'],parc_ts_dictionary['median_ts_intensities_nifti']))
    
    elif has_nifti:
        
        parc_ts_dictionary['time_series'] = parc_ts_dictionary['time_series_nifti']
        parc_ts_dictionary['labels'] = parc_ts_dictionary['labels_nifti']
        parc_ts_dictionary['median_ts_intensities'] = parc_ts_dictionary['median_ts_intensities_nifti']
        
    elif has_gifti:
        
        parc_ts_dictionary['time_series'] = parc_ts_dictionary['time_series_surface']
        parc_ts_dictionary['labels'] = parc_ts_dictionary['labels_surface']
        parc_ts_dictionary['median_ts_intensities'] = parc_ts_dictionary['median_ts_intensities_surface']
        
    else:
        
        raise NameError('Error: File Path Dictionary must either have gifti or nifti specified')
    
    
    parc_ts_dictionary['TR'] = TR
    
    
    
    ####################################################
    #Load the melodic IC time series
    melodic_df = pd.read_csv(file_path_dictionary['melodic_mixing_path'], sep="\t", header=None)
    parc_ts_dictionary['melodic_mixing_matrix'] = melodic_df.values 

    ####################################################
    #Load the indices of the aroma ics
    aroma_ics_df = pd.read_csv(file_path_dictionary['aroma_noise_ics_path'], header=None)
    parc_ts_dictionary['aroma_noise_ic_inds'] = (aroma_ics_df.values - 1).reshape(-1,1)
    

    ####################################################
    #Gather the ICs identified as noise/clean by AROMA
    noise_comps = parc_ts_dictionary['aroma_noise_ic_inds']
    all_ics = melodic_df.values 
    mask = np.zeros(all_ics.shape[1],dtype=bool)
    mask[noise_comps] = True
    parc_ts_dictionary['aroma_noise_ic_timeseries'] = all_ics[:,~mask]
    parc_ts_dictionary['aroma_clean_ic_timeseries'] = all_ics[:,mask]

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
    general_info_dict['n_skip_vols'] = len(np.where(np.sum(np.absolute(parc_ts_dictionary['melodic_mixing_matrix']), axis=1) < 0.1)[0])
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
        
        return confounds_dictionary
    
    
    
def save_dictionary(dictionary, path_for_dictionary_dir, overwrite = False):
    
        
    if path_for_dictionary_dir[-5:] == '.json':
        
        with open(os.path.join(path_for_dictionary_dir), 'w') as temp_file:
            json_dict = json.dumps(dictionary, indent = 4, sort_keys = True)
            temp_file.write(json_dict)        

    else:
        
        if os.path.exists(path_for_dictionary_dir) == True:
        
            if overwrite == False:

                raise NameError('Error: Parc Timeseries Dictionary Directory Already Exists at this path')

        else:
        
            os.makedirs(path_for_dictionary_dir)
        
        
        
        for temp_key in dictionary.keys():


            if type(dictionary[temp_key]) == dict:

                to_overwrite = overwrite
                save_dictionary(dictionary[temp_key], os.path.join(path_for_dictionary_dir, temp_key), overwrite = to_overwrite)

            else:


                if path_for_dictionary_dir[-3:] == 'txt':

                    with open(os.path.join(path_for_dictionary_dir, temp_key), 'w') as temp_file:
                        temp_file.write(str(dictionary[temp_key]))

                else:

                    np.save(os.path.join(path_for_dictionary_dir, temp_key), dictionary[temp_key])
                 
            
    return
                
                
    
    
    
def load_dictionary(dictionary_dir_path):
    
    dictionary = {}
    
    os.chdir(dictionary_dir_path)
    directory_contents = os.listdir()
    directory_final_name = dictionary_dir_path.split('/')[-1]
    for temp_file in directory_contents:
        
        if os.path.isdir(temp_file):
            
            dictionary[temp_file] = load_dictionary(os.path.join(dictionary_dir_path, temp_file))
            os.chdir(dictionary_dir_path)
            
        else:
            
            if temp_file[-5:] == '.json':
                                
                with open(temp_file, 'r') as temp_json:
                    json_contents = temp_json.read()
                    dictionary[temp_file] = json.loads(json_contents)
                    
            else:
            
                if dictionary_dir_path[-3:] == 'txt':

                    with open(temp_file,'r') as temp_reading:
                        file_contents = temp_reading.read()

                        try:
                            dictionary[temp_file.split('.')[0]] = float(file_contents)
                        except:

                            if file_contents[0] == '[' and file_contents[-1] == ']':
                                split_contents = file_contents[1:-1].split(',')
                                dictionary[temp_file.split('.')[0]] = []
                                for temp_item in split_contents:
                                    split_limited_1 = temp_item.replace(' ','')
                                    split_limited_2 = split_limited_1.replace("'","")
                                    dictionary[temp_file.split('.')[0]].append(split_limited_2)

                            else:
                                dictionary[temp_file.split('.')[0]] = file_contents                

                else:

                    dictionary[temp_file.split('.')[0]] = np.load(temp_file)
                
                
                
    return dictionary