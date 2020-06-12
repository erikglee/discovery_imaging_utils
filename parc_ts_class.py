import _pickle as pickle
import pandas as pd
import numpy as np
import os
from eriks_packages import imaging_utils


#This class is for storing parcellated, uncleaned resting state data. To use this class, you should have a parcellation you
#are interested in using in gifti space, and also functional data projected onto the standard fsaverage surface. Outside of #that, you should have the MELODIC mixing tsv file, AROMA noise ICs csv file, and confounds regressors tsv file in the same
#folder as the surface gifti data. The class should store all data needed for any subsequent denoising, and make denoising 
#much easier if you want to implement flexible routines.

#Example usage new_parc_timeseries = parc_timeseries(path_to_lh_gifti_func_file, path_to_lh_parcellation, TR_of_func_scan)

#The previous code just initializes some paths of interest. If you then want to load all the different data elements 
#(including parcellated time series, parcel labels, mean fd, number of skip volumes at beginning of scan, confounds, etc.)
#use the following (after executing the previously shown line of code)

# new_parc_timeseries.populate_all_fields()

#Also if you want to save/load this object for later use (which is one of the main points to save on computation)
# then call new_parc_timeseries.save_object(path_to_file_to_be_created)
# or to load loaded_timeseries = parc_timeseries.load_object(path_to_existing_object)

class parc_timeseries:
    
    
    def __init__(self, lh_gii_path, lh_parcellation_path, TR, aroma_included=True):
        
        #Collect file paths
        self.file_paths = _file_paths(lh_gii_path, lh_parcellation_path, aroma_included)
        
        #Set TR
        self.TR = TR
        
        #Find session/subject names
        temp_path = lh_gii_path.split('/')
        end_path = temp_path[-1:][0]
        split_end_path = end_path.split('_')
        self.subject = split_end_path[0]
        
        if split_end_path[1][0:3]:
            self.session = split_end_path[1]
        else:
            self.session = []
        
        #Fields to be populated when the function "populate_all_fields" is ran
        if aroma_included:
            self.melodic_mixing = [] #implemented
            self.aroma_noise_ic_inds = [] #implemented
            self.aroma_clean_ics = [] #implemented
            self.aroma_noise_ics = [] #implemented
                        
        #Fields to be populated when the function "populate_all_fields" is ran
        self.time_series = [] #implemented
        self.parc_labels = [] #implemented
        self.n_skip_vols = [] #implemented
        self.mean_fd = [] #implemented
        self.confounds = [] #implemented

        
            
            
        
    def all_files_present(self):
        
        
        #Check if all files exist, and if they don't
        #return False
        files_present = True

        for temp_field, temp_file in enumerate(self.file_paths.__dict__):

            if os.path.exists(self.file_paths.__dict__[temp_file]) == False:

                files_present = False
                print(temp_file)
                    
        return files_present
            
        
        
        
    def populate_all_fields(self):
        
        
        ##############################################################################
        #Load the timeseries data and apply parcellation, saving also the parcel labels
        lh_data = imaging_utils.load_gifti_func(self.file_paths.lh_func_path)
        rh_data = imaging_utils.load_gifti_func(self.file_paths.rh_func_path)
        self.time_series, self.parc_labels, self.parc_median_signal_intensities = imaging_utils.demedian_parcellate_func_combine_hemis(lh_data, rh_data, self.file_paths.lh_parcellation_path, self.file_paths.rh_parcellation_path)
        
        ####################################################
        #Load the melodic IC time series
        melodic_df = pd.read_csv(self.file_paths.melodic_mixing_path, sep="\t", header=None)
        self.melodic_mixing = melodic_df.values 
        
        ####################################################
        #Load the indices of the aroma ics
        aroma_ics_df = pd.read_csv(self.file_paths.aroma_noise_ics_path, header=None)
        self.aroma_noise_ic_inds = (aroma_ics_df.values - 1).reshape(-1,1)
        
        ####################################################
        #Gather the ICs identified as noise/clean by AROMA
        noise_comps = self.aroma_noise_ic_inds #do I need to convert to int?
        all_ics = melodic_df.values 

        mask = np.zeros(all_ics.shape[1],dtype=bool)
        mask[noise_comps] = True
        self.aroma_clean_ics = all_ics[:,~mask]
        self.aroma_noise_ics = all_ics[:,mask]
        
        ####################################################
        #Get the variables from the confounds regressors file
        #confound_df = pd.read_csv(self.confounds_regressors_path, sep='\t')
        #for (columnName, columnData) in confound_df.iteritems():
        #    setattr(self, columnName, columnData.as_matrix())
        self.confounds = confounds_class(self.file_paths.confounds_regressors_path)
        
        
        
        ###################################################
        #Calculate the number of timepoints to skip at the beginning for this person.
        #If equal to zero, we will actually call it one so that we don't run into any
        #issues during denoising with derivatives
        self.n_skip_vols = len(np.where(np.sum(np.absolute(self.melodic_mixing), axis=1) < 0.1)[0])
        if self.n_skip_vols == 0:
            self.n_skip_vols = 1
            
            
        ###################################################
        #Calculate the mean framewise displacement (not including the n_skip_vols)
        self.mean_fd = np.mean(self.confounds.framewise_displacement[self.n_skip_vols:])
        
        
        
        
        
        
            
    
    
    #usage parc_timeseries_you_want_to_save(name_of_file_to_be_made)
    def save_object(self, name_of_file):
                
        pickle.dump(self, open(name_of_file, "wb"))
        
    
    
    #usage new_object = parc_timeseries.load_object(name_of_file_to_be_loaded)
    def load_object(name_of_file):
                
        return pickle.load(open(name_of_file, "rb" ))
            


#This is an internal class for use with the parc_timeseries class. It is used
#to load the confounds from a confounds_regressors tsv file and put them into a data
#object, and also group together some commonly used nuisance regressors.
class confounds_class:
    
    
    def __init__(self, confounds_regressors_tsv_path):
        
        confound_df = pd.read_csv(confounds_regressors_tsv_path, sep='\t')
        for (columnName, columnData) in confound_df.iteritems():
            setattr(self, columnName, columnData.as_matrix())
            
        
        #For convenience, bunch together some commonly used nuisance components
        
        #Six motion realignment paramters
        self.six_motion_regs = np.vstack((self.trans_x, self.trans_y, self.trans_z,
                                         self.rot_x, self.rot_y, self.rot_z))
        
        #Six motion realignment parameters plus their temporal derivatives
        self.twelve_motion_regs = np.vstack((self.trans_x, self.trans_y, self.trans_z,
                                         self.rot_x, self.rot_y, self.rot_z,
                                         self.trans_x_derivative1, self.trans_y_derivative1,
                                         self.trans_z_derivative1, self.rot_x_derivative1,
                                         self.rot_y_derivative1, self.rot_z_derivative1))
        
        #Six motion realignment parameters, their temporal derivatives, and
        #the square of both
        self.twentyfour_motion_regs = np.vstack((self.trans_x, self.trans_y, self.trans_z,
                                         self.rot_x, self.rot_y, self.rot_z,
                                         self.trans_x_derivative1, self.trans_y_derivative1,
                                         self.trans_z_derivative1, self.rot_x_derivative1,
                                         self.rot_y_derivative1, self.rot_z_derivative1,
                                         self.trans_x_power2, self.trans_y_power2, self.trans_z_power2,
                                         self.rot_x_power2, self.rot_y_power2, self.rot_z_power2,
                                         self.trans_x_derivative1_power2, self.trans_y_derivative1_power2,
                                         self.trans_z_derivative1_power2, self.rot_x_derivative1_power2,
                                         self.rot_y_derivative1_power2, self.rot_z_derivative1_power2))
        
        #white matter, and csf
        self.wmcsf = np.vstack((self.white_matter, self.csf))
        
        #white matter, csf, and their temporal derivatives
        self.wmcsf_derivs = np.vstack((self.white_matter, self.csf, 
                                      self.white_matter_derivative1, self.csf_derivative1))
        
        #White matter, csf, and global signal
        self.wmcsfgsr = np.vstack((self.white_matter, self.csf, self.global_signal))
        
        #White matter, csf, and global signal plus their temporal derivatives
        self.wmcsfgsr_derivs = np.vstack((self.white_matter, self.csf, self.global_signal,
                                        self.white_matter_derivative1, self.csf_derivative1,
                                        self.global_signal_derivative1))
        
        #The first five anatomical comp cor components
        self.five_acompcors = np.vstack((self.a_comp_cor_00, self.a_comp_cor_01,
                                         self.a_comp_cor_02, self.a_comp_cor_03,
                                         self.a_comp_cor_04))
        
class _file_paths:
        
    def __init__(self, lh_gii_path, lh_parcellation_path, aroma_included=True):
        
        self.lh_func_path = lh_gii_path
        self.rh_func_path = lh_gii_path[:-10] + 'R.func.gii'
        self.lh_parcellation_path = lh_parcellation_path
        
        lh_parcel_end = lh_parcellation_path.split('/')[-1]
        self.rh_parcellation_path = lh_parcellation_path[:-len(lh_parcel_end)] + 'r' + lh_parcellation_path.split('/')[-1][1:]
        
        
        temp_path = lh_gii_path.split('/')
        end_path = temp_path[-1:][0]
        split_end_path = end_path.split('_')
                
        if split_end_path[1][0:3]:
            self.session = split_end_path[1]
        else:
            self.session = []
        
        
        
        self.confounds_regressors_path = lh_gii_path[:-len(end_path)] + end_path[:-31] + 'desc-confounds_regressors.tsv'

        if aroma_included:
            self.melodic_mixing_path = lh_gii_path[:-len(end_path)] + end_path[:-31] + 'desc-MELODIC_mixing.tsv'
            self.aroma_noise_ics_path = lh_gii_path[:-len(end_path)]+ end_path[:-31] + 'AROMAnoiseICs.csv'