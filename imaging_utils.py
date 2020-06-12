#!/usr/bin/env python

import sys
from nibabel import load as nib_load
import nibabel as nib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from scipy import signal
import os
from numpy import genfromtxt
from sklearn.decomposition import PCA




#Wrapper function to load functional data from
#a gifti file using nibabel. Returns data in shape
#<num_verts x num_timepoints>
def load_gifti_func(path_to_file):

    gifti_img = nib_load(path_to_file)
    gifti_list = [x.data for x in gifti_img.darrays]
    gifti_data = np.vstack(gifti_list).transpose()
                                
    return gifti_data

def load_cifti_func(path_to_file):
    
    cifti_img = nib_load(path_to_file)
    return np.asarray(cifti_img.dataobj).transpose()

#Function to make a netmat plot for one of the schaeffer 7 network parcellations.
#The input data can have dimensions consistant with any of the parcellations.
def imagesc_schaeffer(connectivity_matrix, parcel_labels, border_width, minmax):
        
    #The names of the different networks
    network_names = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    network_colors = [[121/255,3/255,136/255,1],[67/255,129/255,182/255,1],[0/255,150/255,0/255,1], \
                      [198/255,41/255,254/255,1],[219/255,249/255,160/255,1], \
                      [232/255,149/255,0/255,1], [207/255,60/255,74/255,1]]

    #[121,3,136,1]
    #Array to store network IDs (0-6, corresponding to order of network names)
    network_ids = np.zeros((len(parcel_labels),1))

    #Find which network each parcel belongs to
    for i in range(0,len(parcel_labels)):
        for j in range(0,len(network_names)):

            if network_names[j] in parcel_labels[i]:
                network_ids[i] = j


    #Create arrays for the sorted network ids and also store the inds to
    #obtain the sorted matrix
    sorted_ids = np.sort(network_ids, axis = 0, kind = 'mergesort')
    sorted_id_inds = np.argsort(network_ids, axis = 0, kind = 'mergesort')

    #Calculate where the center and edge of each network is for labeling
    #different networks on netmat figures
    network_edges = np.zeros((len(network_names),1))
    for i in range(0,len(network_names)):
        for j in range(0,len(parcel_labels)):

            if sorted_ids[j] == i:

                network_edges[i] = j

    network_centers = np.zeros((len(network_names),1))
    network_centers[0] = network_edges[0]/2.0
    for i in range(1,len(network_edges)):
        network_centers[i] = (network_edges[i] + network_edges[i-1])/2.0            

    #Sort the connectivity matrix
    sorted_conn_matrix = np.zeros(connectivity_matrix.shape)
    sorted_conn_matrix = np.reshape(connectivity_matrix[sorted_id_inds,:], connectivity_matrix.shape)
    sorted_conn_matrix = np.reshape(sorted_conn_matrix[:,sorted_id_inds], connectivity_matrix.shape)
    sorted_conn_matrix = (sorted_conn_matrix - minmax[0])*(1/(minmax[1]*2))


    cmap = matplotlib.cm.jet
    norm = matplotlib.colors.Normalize(vmin=-0.8, vmax=0.8)

    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    jet_conn_matrix =  m.to_rgba(sorted_conn_matrix, norm=False)

    jet_conn_with_borders = np.zeros((jet_conn_matrix.shape[0] + border_width, jet_conn_matrix.shape[1] + border_width, \
                                      jet_conn_matrix.shape[2]))
    jet_conn_with_borders[0:(-1*border_width),border_width:,:] = jet_conn_matrix
    for i in range(0,sorted_ids.shape[0]):
        jet_conn_with_borders[i,0:border_width,:] = network_colors[int(sorted_ids[i])]
        jet_conn_with_borders[(-1*border_width):,i+border_width,:] = network_colors[int(sorted_ids[i])]


######################################################################################
######################################################################################
######################################################################################
    #Calculate where the center and edge of each network is for labeling
    #different networks on netmat figures
    network_edges = np.zeros((len(network_names),1))
    for i in range(0,len(network_names)):
        for j in range(0,len(parcel_labels)):

            if sorted_ids[j] == i:

                network_edges[i] = j

    network_centers = np.zeros((len(network_names),1))
    network_centers[0] = network_edges[0]/2.0
    for i in range(1,len(network_edges)):
        network_centers[i] = (network_edges[i] + network_edges[i-1])/2.0

    #network_names4fig = ['', 'Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default', '']
    #network_centers_plus_overall_edge = np.copy(network_centers)
    #network_centers_plus_overall_edge = np.insert(network_centers_plus_overall_edge, 0, 0)
    #network_centers_plus_overall_edge = np.append(network_centers_plus_overall_edge, len(parcel_labels))
    #plt.xticks(network_centers_plus_overall_edge, network_names4fig, rotation = 'vertical')
    #plt.yticks(network_centers_plus_overall_edge, network_names4fig)
    
    plot_obj = plt.imshow(jet_conn_with_borders)#,cmap='jet')
    
    for i in network_edges[:-1]:
        plt.axvline(x=i + border_width + 0.7,color='black', lw=1)
    plt.axvline(x=border_width - 0.7, color='black', lw=1)    
    
    for i in network_edges:
        plt.axhline(y=i,color='black', lw=1)

######################################################################################
######################################################################################
######################################################################################

    #jet_conn_matrix.shape
    #plot_obj = plt.imshow(jet_conn_with_borders)#,cmap='jet')
    #plot_obj.colorbar(m)
    #plt.colorbar(jet)
    plt.xticks([])
    plt.yticks([])

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10.5, 10.5)
                                     
    return fig


#Function to make a netmat plot for one of the schaeffer 7 network parcellations.
#The input data can have dimensions consistant with any of the parcellations.
def simple_imagesc_schaeffer(connectivity_matrix, parcel_labels, minmax):
        
    #The names of the different networks
    network_names = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    network_colors = [[121/255,3/255,136/255,1],[67/255,129/255,182/255,1],[0/255,150/255,0/255,1], \
                      [198/255,41/255,254/255,1],[219/255,249/255,160/255,1], \
                      [232/255,149/255,0/255,1], [207/255,60/255,74/255,1]]

    #Array to store network IDs (0-6, corresponding to order of network names)
    network_ids = np.zeros((len(parcel_labels),1))

    #Find which network each parcel belongs to
    for i in range(0,len(parcel_labels)):
        for j in range(0,len(network_names)):

            if network_names[j] in parcel_labels[i]:
                network_ids[i] = j


    #Create arrays for the sorted network ids and also store the inds to
    #obtain the sorted matrix
    sorted_ids = np.sort(network_ids, axis = 0, kind = 'mergesort')
    sorted_id_inds = np.argsort(network_ids, axis = 0, kind = 'mergesort')

    #Calculate where the center and edge of each network is for labeling
    #different networks on netmat figures
    network_edges = np.zeros((len(network_names),1))
    for i in range(0,len(network_names)):
        for j in range(0,len(parcel_labels)):

            if sorted_ids[j] == i:

                network_edges[i] = j

    network_centers = np.zeros((len(network_names),1))
    network_centers[0] = network_edges[0]/2.0
    for i in range(1,len(network_edges)):
        network_centers[i] = (network_edges[i] + network_edges[i-1])/2.0            

    #Sort the connectivity matrix
    sorted_conn_matrix = np.zeros(connectivity_matrix.shape)
    sorted_conn_matrix = np.reshape(connectivity_matrix[sorted_id_inds,:], connectivity_matrix.shape)
    sorted_conn_matrix = np.reshape(sorted_conn_matrix[:,sorted_id_inds], connectivity_matrix.shape)

######################################################################################
######################################################################################
######################################################################################
    #Calculate where the center and edge of each network is for labeling
    #different networks on netmat figures
    network_edges = np.zeros((len(network_names),1))
    for i in range(0,len(network_names)):
        for j in range(0,len(parcel_labels)):

            if sorted_ids[j] == i:

                network_edges[i] = j

    network_centers = np.zeros((len(network_names),1))
    network_centers[0] = network_edges[0]/2.0
    for i in range(1,len(network_edges)):
        network_centers[i] = (network_edges[i] + network_edges[i-1])/2.0
    
    plt.subplots(dpi = 100)
    plot_obj = plt.imshow(sorted_conn_matrix, vmin=minmax[0], vmax=minmax[1])
    
    for i in network_edges[:-1]:
        plt.axvline(x=i + 0.7,color='red', lw=1)
    #plt.axvline(x= -0.7, color='red', lw=1)    
    
    for i in network_edges[:-1]:
        plt.axhline(y=i,color='red', lw=1)
        
######################################################################################
######################################################################################
######################################################################################
    plt.xticks(network_centers, network_names, rotation=45)
    plt.yticks(network_centers, network_names, rotation=45)
    
    plt.xlim((0,len(parcel_labels)))
    plt.ylim((0,len(parcel_labels)))

    fig = matplotlib.pyplot.gcf()
    ax = matplotlib.pyplot.gca()
    
    plt.colorbar()
                                     
    return (fig, ax)

def calc_fishers_icc(tp1, tp2):
    
    #Calculate intraclass correlation coefficient
    #from the equation on wikipedia describing
    #fisher's formulation. tp1 and tp2 should
    # be of shape (n,1) or (n,) where n is the
    #number of samples

    xhat = np.mean(np.vstack((tp1, tp2)))
    sq_dif1 = np.power((tp1 - xhat),2)
    sq_dif2 = np.power((tp2 - xhat),2)
    s2 = np.mean(np.vstack((sq_dif1, sq_dif2)))
    r = 1/(tp1.shape[0]*s2)*np.sum(np.multiply(tp1 - xhat, tp2 - xhat))
    
    return r

def calc_icc(tp1, tp2):
    
    #Calculate intraclass correlation coefficient
    #from the equation in "A Guideline of Selecting and 
    #Reporting Intraclass Correlation Coefficients for
    #Reliability Research" for the case "two-way mixed
    #effects, absolute agreement, and single measurement".
    #This should be the ideal way to measure test-retest
    #reliability. tp1 and tp2 should be of shape (n,1) or
    #(n,) where n is the number of samples
    

    #STILL IMPLEMENTING, NOT FINISHED YET!!!!!!!!!!
    
    #MSR = mean squared for rows
    #MSE = mean square for error
    #MSC = mean squared for columns
    #k = 2 - number of measurements
    #n = number of subjects
    
    #yhat = 
    
    #SSE = 

    xhat = np.mean(np.vstack((tp1, tp2)))
    sq_dif1 = np.power((tp1 - xhat),2)
    sq_dif2 = np.power((tp2 - xhat),2)
    s2 = np.mean(np.vstack((sq_dif1, sq_dif2)))
    r = 1/(tp1.shape[0]*s2)*np.sum(np.multiply(tp1 - xhat, tp2 - xhat))
    
    return r

def convert_to_upper_arr(np_square_matrix):
    
    #Function that takes a square matrix,
    #and outputs its upper triangle without
    #the diagonal as an array
    

    inds = np.triu_indices(np_square_matrix.shape[0], k = 1)
    return np_square_matrix[inds]


def pre_post_carpet_plot(noisy_time_series, cleaned_time_series):
    #This function is for calculating a carpet plot figure, that
    #will allow for comparison of the BOLD time series before and
    #after denoising takes place. The two input matrices should have
    #shape <num_parcels, num_timepoints>, and will ideally be from a
    #parcellated time series and not whole hemisphere data (lots of points).
    
    #The script will demean and then normalize all regions' time signals,
    #and then will display them side by side on grey-scale plots
    
    
    #Copy the data
    noisy_data = np.copy(noisy_time_series)
    clean_data = np.copy(cleaned_time_series)

    #Calculate means and standard deviations for all parcels
    noisy_means = np.mean(noisy_data, axis = 1)
    noisy_stds = np.std(noisy_data, axis = 1)
    clean_means = np.mean(clean_data, axis = 1)
    clean_stds = np.std(clean_data, axis = 1)
    
    #Empty matrices for demeaned and normalized data
    dn_noisy_data = np.zeros(noisy_data.shape)
    dn_clean_data = np.zeros(clean_data.shape)

    #Use the means and stds to mean and normalize all parcels' time signals
    for i in range(0, clean_data.shape[0]):
        dn_noisy_data[i,:] = (noisy_data[i,:] - noisy_means[i])/noisy_stds[i]
        dn_clean_data[i,:] = (clean_data[i,:] - clean_means[i])/clean_stds[i]
 
    #Create a subplot
    plot_obj = plt.subplot(1,2,1)
    
    #Plot the noisy data              
    img_plot = plt.imshow(dn_noisy_data, aspect = 'auto', cmap = 'binary')
    plt.title('Noisy BOLD Data')
    plt.xlabel('Timepoint #')
    plt.ylabel('Region # (Arbritrary)')
    plt.colorbar()
                             
    #Plot the clean data
    plt.subplot(1,2,2)
    img_plot2 = plt.imshow(dn_clean_data, aspect = 'auto', cmap = 'binary')
    plt.title('Clean BOLD Data')
    plt.xlabel('Timepoint #')
    plt.colorbar()
    fig = plt.gcf()
    fig.set_size_inches(15, 5)
                             
    return plot_obj


                             
                             
def denoise_func_data(input_data, confound_path, ica_ts_path, ica_noise_path, list_of_confounds_to_use, fd_thresh):
    #This function is used to denoise data output from fmriprep. 
    #functional data stored in input_data must already be loaded
    #and stored in a matrix with shape <num_regions, num_timepoints>
    
    #The function can only be ran if ICA-AROMA was requested during fmriprep.
    
    #What this function does, is takes all of a subject's IC's (both noise and
    #network relevant) and any desired confounds (i.e. csf, wm, movement), and
    #uses ordinary least squares on each region in input_data. Then, the script
    #takes the noise relevant coefficients from the nuisance IC's and their
    #corresponding time-courses, and combines them to generate an overall nuisance
    #signal. This nuisance signal is then subtracted from the original signal to create 
    #a cleaned time signal. Then a high-pass filter is applied to cut off any content
    #below 0.015Hz. The final signal that gets returned has the volumes before intensity
    #stabilizes removed.
                             
    #This procedure is comparable to what would be thought of as a soft ICA-AROMA implementation
    
    #Note: on top of the confounds included in list_of_counfounds_to_use, this script
    #will also make use of an additional two confounds to do linear detrending (intercept,
    #and slope)
                             
    #Secondary note: this function should run fine on any type of data, but will probably
    #take quite a long time ~20+ minutes on whole surface data with ~100k regions..
         
    func_data = np.copy(input_data)  
            
    
    #Create a dataframe for nuisance variables in confounds
    confound_df = pd.read_csv(confound_path, sep='\t')   
                             
    #Create a dataframe for motion IC inds, and grab inds
    motion_df = pd.read_csv(ica_noise_path, header=None)
    motion_ic_inds = (motion_df.values - 1).reshape(-1,1)
                             
    #Create a dataframe for melodic IC timeseries and extract values
    melodic_df = pd.read_csv(ica_ts_path, sep="\t", header=None)
    melodic_ics_time_series = melodic_df.values                      
                             
    #Find the number of zero inds at the beginning of scan that
    #fmriprep thinks we should exclude
    temp_sum = 0
    for i in range(0, melodic_ics_time_series.shape[0]):
        temp_sum = np.sum(melodic_ics_time_series[i])
        if temp_sum > 0.000001:
            num_zero_inds = i
            break
                             
    fd = confound_df.framewise_displacement.values
    fd = np.nan_to_num(fd)
    fd_exclusion = np.zeros((func_data.shape[1]))
    fd_exclusion[:num_zero_inds] = 1 #Start by auto removing the points before equilibrium
    #iterate through all timepoints
    for i in range(num_zero_inds, func_data.shape[1]):
        #move forward if current fd is greater than
        #the fd exclusion threshold
        if fd[i] >= fd_thresh:
                  
            #If it is the last timepoint, then exclude
            #the current timepoint and previous timepoint
            if i == func_data.shape[1]:
                fd_exclusion[i] = 1
                fd_exclusion[i-1] = 1
                             
            #Otherwise, exclude the previous, current, and
            #next timepoint. Note - the first timepoint will
            #always have a framewise displacement of 0, so we
            #don't need to put in an exception for this case
            if i != func_data.shape[1]:
                fd_exclusion[i+1] = 1
                fd_exclusion[i] = 1
                fd_exclusion[i-1] = 1
     
    #now we will go back over the fd_exclusion array, and get
    #rid of any snips where there aren't at least 3 consecutive
    #good timepoints
    bad_inds = np.where(fd_exclusion > 0.01)[0]
    for i in range(0, bad_inds.shape[0] - 1):
        #print('first: {first}, second: {second}, difference: {diff}'.format(first = bad_inds[i], second = bad_inds[i+1], diff = bad_inds[i+1] - bad_inds[i]))
        if (bad_inds[i+1] - bad_inds[i]) < 4:
            for j in range(bad_inds[i] + 1, bad_inds[i+1]):
                fd_exclusion[j] = 1000
    
    #Now we have the good inds to reference during other denoising steps
    bad_inds = np.where(fd_exclusion > 0.01)[0]
    good_inds = np.where(fd_exclusion < 0.01)[0]
    good_inds_slice = slice(good_inds)
                             
        

    #Create an array to store the cleaned data and
    #subsequently filtered data
    denoised_time_series = np.zeros((func_data.shape[0], good_inds.shape[0]))
    filt_denoised_ts = np.zeros((func_data.shape[0], good_inds.shape[0]))
                             
    #Design a filter that will result in signal < 0.015 Hz
    #being cutoff with a TR = 0.8 (SHOULD MAKE MORE GENERAL!!)
    b, a = signal.butter(6, 0.024, btype='highpass')

    constant = np.ones((func_data.shape[1]))
    linear = np.linspace(0,func_data.shape[1],func_data.shape[1])
    #linear = np.copy(linear[good_inds])
    partial_confounds = [constant, linear]
                             
    for conf_name in list_of_confounds_to_use:
        temp = confound_df.loc[ : , conf_name ]
        partial_confounds.append(np.copy(temp.values)) 
                             
    non_ic_confounds = np.vstack(partial_confounds)
    full_conf = np.hstack((melodic_ics_time_series, non_ic_confounds.transpose()))

    for i in range(0,func_data.shape[0]):
        mdl = sm.regression.linear_model.OLS(func_data[i,good_inds].reshape(-1,1),full_conf[good_inds,:])
        results = mdl.fit()
        confound_signal = np.zeros(melodic_ics_time_series.shape[0])
        for j in range(0,motion_ic_inds.shape[1]):
            confound_signal = np.add(confound_signal, full_conf[:,j]*results.params[motion_ic_inds[j]])


        #Now add the motion and other non-IC time series to the confound signal
        for j in range(melodic_ics_time_series.shape[1], full_conf.shape[1]):
            confound_signal = np.add(confound_signal, full_conf[:,j]*results.params[j])

        denoised_time_series[i,:] = func_data[i,good_inds] - confound_signal[good_inds]
        filt_denoised_ts[i,:] = signal.filtfilt(b, a, denoised_time_series[i,:])
                             
    
    return filt_denoised_ts
                             
                             
                             
                             
def parcellate_func_combine_hemis(lh_func, rh_func, lh_parcel_path, rh_parcel_path):
    
    #Function that takes functional data in the form <num_verts, num_timepoints> for
    #both the left and right hemisphere, and averages the functional time series across
    #all vertices defined in a given parcel, for every parcel, with the parcels identified
    #by a annotation file specified at ?h_parcel_path. The function then returns a combined
    #matrix of size <num_parcels, num_timepoints> and <num_labels> for the time series and
    #parcel label names, respectively. The lh parcels will preceed the rh parcels in order.
    
    #NOTE: THIS ASSUMES THE FIRST PARCEL WILL BE MEDIAL WALL, AND DISREGARDS ANY VERTICES WITHIN
    #THAT PARCEL. IF THIS IS NOT THE CASE FOR YOUR PARCELLATION, DO NOT USE THIS FUNCTION.
    
    #Output will be tuple of format [labels, ctab, names]
    lh_parcels = nib.freesurfer.io.read_annot(lh_parcel_path)
    rh_parcels = nib.freesurfer.io.read_annot(rh_parcel_path)
                             
    #Make array to store parcellated data with shape <num_parcels, num_timepoints>
    lh_parcellated_data = np.zeros((len(lh_parcels[2]) - 1, lh_func.shape[1]))
    rh_parcellated_data = np.zeros((len(rh_parcels[2]) - 1, rh_func.shape[1]))

    #Start with left hemisphere
    for i in range(1,len(lh_parcels[2])):

        #Find the voxels for the current parcel
        vois = np.where(lh_parcels[0] == i)

        #Take the mean of all voxels of interest
        lh_parcellated_data[i-1, :] = np.mean(lh_func[vois[0],:], axis = 0)

    #Move to right hemisphere
    for i in range(1,len(rh_parcels[2])):

        vois = np.where(rh_parcels[0] == i)
        rh_parcellated_data[i-1, :] = np.mean(rh_func[vois[0],:], axis = 0)

    #Then concatenate parcel labels and parcel timeseries between the left and right hemisphere
    #and drop the medial wall from label list
    parcellated_data = np.vstack((lh_parcellated_data, rh_parcellated_data))
    parcel_labels = lh_parcels[2][1:] + rh_parcels[2][1:]

    #Try to convert the parcel labels from bytes to normal string
    for i in range(0, len(parcel_labels)):
        parcel_labels[i] = parcel_labels[i].decode("utf-8")   
        
    return parcellated_data, parcel_labels
            
            
        
                             
def net_mat_summary_stats(matrix_data, include_diagonals, parcel_labels):
    #Function that takes a network matrix of size <num_parcels x num_parcels>
    #and calculates summary statistics for each grouping of parcels within a 
    #given network combination (i.e. within DMN would be one grouping, between
    #DMN and Control would be another grouping). If you would like to include
    #the diagonals of the matrix set include_diagonals to true, otherwise,
    #as is the case in conventional functional connectivity matrices, exclude
    #the diagonal since it will most commonly be 1 or Inf.
    
    #This function only works on data formatted in the Schaeffer/Yeo 7 network
    #configuration.
    
    #Parcel labels should be a list of strings that has the names of the different
    #parcels in the parcellation. This is how the function knows what parcels
    #belong to what networks.
    
            
    #The names of the different networks
    network_names = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']

    #Array to store network IDs (0-6, corresponding to order of network names)
    network_ids = np.zeros((len(parcel_labels),1))

    #Find which network each parcel belongs to
    for i in range(0,len(parcel_labels)):
        for j in range(0,len(network_names)):

            if network_names[j] in parcel_labels[i]:
                network_ids[i] = j

    #Calculate the average stat for each network combination
    network_stats = np.zeros((7,7))
    for i in range(0,7):
        for j in range(0,7):
            temp_stat = 0
            temp_stat_count = 0
            rel_inds_i = np.where(network_ids == i)[0]
            rel_inds_j = np.where(network_ids == j)[0]
            for inds_i in rel_inds_i:
                for inds_j in rel_inds_j:
                    if inds_i == inds_j:
                        if include_diagonals == True:
                            temp_stat += matrix_data[inds_i, inds_j]
                            temp_stat_count += 1
                    else:
                        temp_stat += matrix_data[inds_i, inds_j]
                        temp_stat_count += 1
            
            network_stats[i,j] = temp_stat/temp_stat_count
            
    
    return network_stats
                        
            
    
def net_summary_stats(parcel_data, parcel_labels):
    #Function that takes a statistic defined at a parcel level, and 
    #resamples that statistic to the network level. This function is a copy of 
    #net_mat_summary_stats only now defined to work on 1D instead of 2D data.
    
    #This function only works on data formatted in the Schaeffer/Yeo 7 network
    #configuration.
    
    #Parcel labels should be a list of strings that has the names of the different
    #parcels in the parcellation. This is how the function knows what parcels
    #belong to what networks.
    
            
    #The names of the different networks
    network_names = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']

    #Array to store network IDs (0-6, corresponding to order of network names)
    network_ids = np.zeros((len(parcel_labels),1))

    #Find which network each parcel belongs to
    for i in range(0,len(parcel_labels)):
        for j in range(0,len(network_names)):

            if network_names[j] in parcel_labels[i]:
                network_ids[i] = j

    #Calculate the average stat for each network combination
    network_stats = np.zeros((7))
    for i in range(0,7):
        temp_stat = 0
        temp_stat_count = 0
        rel_inds_i = np.where(network_ids == i)[0]
        for inds_i in rel_inds_i:
            temp_stat += parcel_data[inds_i]
            temp_stat_count += 1

        network_stats[i] = temp_stat/temp_stat_count
            
    
    return network_stats


def plot_network_timeseries(parcel_data, parcel_labels):
    
    
    #The names of the different networks
    network_names = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    network_colors = [[121/255,3/255,136/255,1],[67/255,129/255,182/255,1],[0/255,150/255,0/255,1], \
                  [198/255,41/255,254/255,1],[219/255,249/255,160/255,1], \
                  [232/255,149/255,0/255,1], [207/255,60/255,74/255,1]]

    #Array to store network IDs (0-6, corresponding to order of network names)
    network_ids = np.zeros((len(parcel_labels),1))

    #Find which network each parcel belongs to
    for i in range(0,len(parcel_labels)):
        for j in range(0,len(network_names)):

            if network_names[j] in parcel_labels[i]:
                network_ids[i] = j
    
    
    
    fig, ax = plt.subplots(7,1)

    for i in range(0,7):
        in_network = np.where(network_ids == i)[0]
        plt.sca(ax[i])
        
        for j in range(0, in_network.shape[0]):
            
            plt.plot(parcel_data[in_network[j]], color=network_colors[i])   
            
        plt.ylabel('Signal Intensity')
        plt.title('Time-Course For All ' + network_names[i] + ' Parcels')
        
        if i != 6:
            plt.xticks([])
    
    
    plt.xlabel('Volume # (excluding high-motion volumes)')
    fig.set_size_inches(15, 20)
    return fig
    
    
    
    
    
    
def plot_network_average_power_spectrum(parcel_data, parcel_labels, TR):
    
    #Think about how we should be doing this in the presence of scrubbing...
    
    return
    
    
    
def calc_alff_falff(parcel_data, freq_range_of_interest):
    
                             
        
    return

def calc_norm_std(parcel_data, confound_path):
    #This script is used to calculate the normalized standard
    #deviation of a cleaned fmri time signal. This is a metric
    #representative of variability/amplitude in the BOLD signal.
    #This is a particularly good option if you are working with
    #scrubbed data such that the FFT for ALFF can no longer be
    #properly calculated.
    
    #parcel_data has size <num_regions, num_timepoints>. Confound
    #path is the path to the confound file for the run of interest.
    #The global signal will be taken from the confound file to calculate
    #the median BOLD signal in the brain before pre-processing. This will then
    #be used to normalize the standard deviation of the BOLD signal such that
    #the output measure will be std(BOLD_Time_Series)/median_global_signal_intensity.
    
    
    
    #Create a dataframe for nuisance variables in confounds
    confound_df = pd.read_csv(confound_path, sep='\t')  
    global_signal = confound_df.global_signal.values
    median_intensity = np.median(global_signal)
    
    parcel_std = np.zeros((parcel_data.shape[0]))
    for i in range(0, parcel_data.shape[0]):
        
        parcel_std[i] = np.std(parcel_data[i,:])/median_intensity
        
        
    return parcel_std

def network_bar_chart(network_vals, ylabel):
    
    #The names of the different networks
    network_names = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
    network_colors = [[121/255,3/255,136/255,1],[67/255,129/255,182/255,1],[0/255,150/255,0/255,1], \
                  [198/255,41/255,254/255,1],[219/255,249/255,160/255,1], \
                  [232/255,149/255,0/255,1], [207/255,60/255,74/255,1]]
    
    x = [1, 2, 3, 4, 5, 6, 7]
    fig = plt.bar(x, network_vals, color = network_colors, tick_label = network_names)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)

    
    return fig

def fs_anat_to_array(path_to_fs_subject, folder_for_output_files):
    #This function serves the function of collecting the aseg.stats file,
    #lh.aparc.stats file, and rh.aparc.stats files from a freesurfer subject
    #found at the path path_to_fs_subject, and grabs the volumes for all
    #subcortical structures, along with volumes, thicknesses, and surface
    #areas for all cortical structures, and saves them as .npy files under
    #folder_for_output_files. Also saves a text file with the names of the
    #regions (one for subcortical, and one for lh/rh)
    
    aseg_path = os.path.join(path_to_fs_subject, 'stats', 'aseg.stats')
    lh_path = os.path.join(path_to_fs_subject, 'stats', 'lh.aparc.stats')
    rh_path = os.path.join(path_to_fs_subject, 'stats', 'rh.aparc.stats')
    
    
    f = open(aseg_path, "r")
    lines = f.readlines()
    f.close()
    header = '# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange'
    subcort_names = ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter', 
                  'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen', 
                  'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'Left-Hippocampus', 
                  'Left-Amygdala', 'CSF' ,'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel', 
                  'Left-choroid-plexus', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent', 
                  'Right-Cerebellum-White-Matter','Right-Cerebellum-Cortex', 'Right-Thalamus-Proper', 
                  'Right-Caudate', 'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus',
                  'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC', 'Right-vessel', 
                  'Right-choroid-plexus', '5th-Ventricle', 'WM-hypointensities', 'Left-WM-hypointensities', 
                  'Right-WM-hypointensities', 'non-WM-hypointensities', 'Left-non-WM-hypointensities', 
                  'Right-non-WM-hypointensities', 'Optic-Chiasm', 'CC_Posterior', 'CC_Mid_Posterior', 
                  'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior']

    aseg_vol = []
    header_found = 0
    for i in range(0,len(lines)):

        if header_found == 1:
            split_line = lines[i].split()
            if split_line[4] != subcort_names[i-header_found_ind]:
                raise NameError('Error: anatomy names do not line up with expectation. Expected ' + 
                               subcort_names[i-header_found_ind] + ' but found ' + split_line[4])
            aseg_vol.append(float(split_line[3]))


        if header in lines[i]:
            header_found = 1
            header_found_ind = i + 1 #actually add one for formatting....
            #This indicates that (1) the column headings should
            #be correct, and that (2) this is where to start
            #looking for anatomical stats
            
            
    
    lh_f = open(lh_path, "r")
    lh_lines = lh_f.readlines()
    lh_f.close()
    
    header = '# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd'
    cort_names = ['bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus', 'entorhinal',
                  'fusiform', 'inferiorparietal', 'inferiortemporal', 'isthmuscingulate', 'lateraloccipital', 
                  'lateralorbitofrontal', 'lingual', 'medialorbitofrontal', 'middletemporal', 'parahippocampal', 
                  'paracentral', 'parsopercularis', 'parsorbitalis', 'parstriangularis', 'pericalcarine', 
                  'postcentral', 'posteriorcingulate', 'precentral', 'precuneus', 'rostralanteriorcingulate',
                  'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal', 'superiortemporal', 'supramarginal',
                  'frontalpole', 'temporalpole', 'transversetemporal', 'insula']

    lh_surface_area = []
    lh_volume = []
    lh_thickness = []
    header_found = 0
    for i in range(0,len(lh_lines)):

        if header_found == 1:
            split_line = lh_lines[i].split()
            if split_line[0] != cort_names[i-header_found_ind]:
                raise NameError('Error: anatomy names do not line up with expectation. Expected ' + 
                               cort_names[i-header_found_ind] + ' but found ' + split_line[4])
            #then insert text to actually grab/save the data.....

            lh_surface_area.append(float(split_line[2]))
            lh_volume.append(float(split_line[3]))
            lh_thickness.append(float(split_line[4]))

        if header in lh_lines[i]:
            header_found = 1
            header_found_ind = i + 1 #actually add one for formatting....
            #This indicates that (1) the column headings should
            #be correct, and that (2) this is where to start
            #looking for anatomical stats



    rh_f = open(rh_path, "r")
    rh_lines = rh_f.readlines()
    rh_f.close()

    rh_surface_area = []
    rh_volume = []
    rh_thickness = []
    header_found = 0
    for i in range(0,len(rh_lines)):

        if header_found == 1:
            split_line = rh_lines[i].split()
            if split_line[0] != cort_names[i-header_found_ind]:
                raise NameError('Error: anatomy names do not line up with expectation. Expected ' + 
                               cort_names[i-header_found_ind] + ' but found ' + split_line[4])
            #then insert text to actually grab/save the data.....

            rh_surface_area.append(float(split_line[2]))
            rh_volume.append(float(split_line[3]))
            rh_thickness.append(float(split_line[4]))

        if header in rh_lines[i]:
            header_found = 1
            header_found_ind = i + 1 #actually add one for formatting....
            #This indicates that (1) the column headings should
            #be correct, and that (2) this is where to start
            #looking for anatomical stats

    if os.path.exists(folder_for_output_files) == False:
        os.mkdir(folder_for_output_files)
    
    #Save the metrics as numpy files
    np.save(os.path.join(folder_for_output_files, 'aseg_vols.npy'), np.asarray(aseg_vol))
    np.save(os.path.join(folder_for_output_files, 'lh_aseg_surface_areas.npy'), np.asarray(lh_surface_area))
    np.save(os.path.join(folder_for_output_files, 'lh_aseg_volumes.npy'), np.asarray(lh_volume))
    np.save(os.path.join(folder_for_output_files, 'lh_aseg_thicknesses.npy'), np.asarray(lh_thickness))
    np.save(os.path.join(folder_for_output_files, 'rh_aseg_surface_areas.npy'), np.asarray(rh_surface_area))
    np.save(os.path.join(folder_for_output_files, 'rh_aseg_volumes.npy'), np.asarray(rh_volume))
    np.save(os.path.join(folder_for_output_files, 'rh_aseg_thicknesses.npy'), np.asarray(rh_thickness))
    
    #Calculate some bilateral metrics
    left_vent = 0
    right_vent = 18
    total_lateral_vent = aseg_vol[left_vent] + aseg_vol[right_vent]

    left_hipp = 11
    right_hipp = 26
    total_hipp_vol = aseg_vol[left_hipp] + aseg_vol[right_hipp]

    left_thal = 4
    right_thal = 22
    total_thal_vol = aseg_vol[left_thal] + aseg_vol[right_thal]

    left_amyg = 12
    right_amyg = 27
    total_amyg_vol = aseg_vol[left_amyg] + aseg_vol[right_amyg]
    
    #Also calculate global thickness
    numerator = np.sum(np.multiply(lh_surface_area,lh_thickness)) + np.sum(np.multiply(rh_surface_area,rh_thickness))
    denominator = np.sum(lh_surface_area) + np.sum(rh_surface_area)
    whole_brain_ave_thick = numerator/denominator
    
    discovery_metric_array = [total_hipp_vol, total_amyg_vol, total_thal_vol,
                             total_lateral_vent, whole_brain_ave_thick]
    
    np.save(os.path.join(folder_for_output_files, 'discovery_anat_metrics.npy'), np.asarray(discovery_metric_array))
    discovery_anat_ids = ['bilateral_hipp_volume', 'bilateral_amyg_vol', 'bilateral_thal_vol',
                          'bilateral_lateral_vent_vol', 'whole_brain_ave_thick']
    
    #Then save a file with the region names
    with open(os.path.join(folder_for_output_files, 'subcortical_region_names.txt'), 'w') as f:
        for item in subcort_names:
            f.write("%s\n" % item)
    
    with open(os.path.join(folder_for_output_files, 'cortical_region_names.txt'), 'w') as f:
        for item in cort_names:
            f.write("%s\n" % item)
            
    with open(os.path.join(folder_for_output_files, 'discovery_region_names.txt'), 'w') as f:
        for item in discovery_anat_ids:
            f.write("%s\n" % item)
    
    
    

#######################################################################################
#######################################################################################
 

def flexible_denoise_func(uncleaned_ts, noise_comps, clean_comps, pca_decomp_num, high_pass, low_pass, tr, scrub_thresh, fd_arr, skip_vols):
    
    #Function to do flexible denoising on time signals. Here the input, uncleaned_ts should
    #be a matrix of shape (n x t) where n is the number of regions and t is the number
    #of timepoints.
    
    #noise_comps should be shape (t x comps) with all the noise components you want.
    
    #clean_comps should be shape (t x comps) with all the clean IC components to include. Can
    #be [] if you don't want to include these in the model
    
    #pca_decomp_num should be False if you don't want to do a pca decomposition of noise_comps,
    #or otherwise should be equal to the maximum number of pca components for noise_comps you
    #want to be included in the final statistical model
    
    #high_pass is the Hz cutoff for any low frequency signal to be filtered out of the model.
    #This is currently manditory and if both high_pass and low_pass are given, a bandpass filter
    #will be constructed
    
    #low_pass should be False if you don't want to do a low pass filter after denoising, otherwise
    #put a value in Hz for the cutoff frequency
    
    #tr = repition time of the scan (used for filtering purposes)
    
    #scrub_thresh is the fd threshold used to determine whether or not to scrub. Set to False if you
    #don't want to include scrubbing. Importantly, scrubbing will be done as the last step in processing,
    #(i.e. after denoising and filtering) which may not be preferred
    
    #fd_arr can be set to False if scrub_thresh is set to False, otherwise fd_arr should be an array
    #of shpae (t x 1) with framewise displacement for the current run
    
    #skip_vols is the number of volumes you want to skip at the beginning of the scan.
    
    #Aside from what is specified in these parameters, a linear trend and constant is also 
    #automatically included in the denoising matrix, and after denoising there is automatically
    #a high-pass filter. Scrubbing is done after denoising.
    
    #Define filter charecteristics
    if high_pass != False:
        if low_pass == False:
            #If you only want a low-pass
            b, a = construct_filter('highpass', [high_pass], tr, 6)
        else:
            #If you want a band-pass
            b, a = construct_filter('bandpass', [high_pass, low_pass], tr, 6)
    
    new_time_signal = np.zeros((uncleaned_ts.shape[0], uncleaned_ts.shape[1] - skip_vols))
    
    #First do the PCA decomposition if necessary
    if pca_decomp_num != False:
        noise_comps = reduce_noise_ics(noise_comps[skip_vols:,:], pca_decomp_num)
    else:
        noise_comps = noise_comps[skip_vols:,:]
        
    if clean_comps != []:
        clean_comps = clean_comps[skip_vols:,:]
    
    
    #Calculate a constant and linear trend term 
    #for the linear model regressors
    constant = np.ones((noise_comps.shape[0],1))
    linear_trend = np.linspace(0,noise_comps.shape[0], 
                               noise_comps.shape[0]).reshape((noise_comps.shape[0],1)) 
    
    #Compile the bad regressors, and the full regressors (X)
    bad_regressors = np.hstack((noise_comps, constant, linear_trend))
    
    if clean_comps != []:
        X = np.hstack((noise_comps, constant, linear_trend, clean_comps))
    else:
        X = np.hstack((noise_comps, constant, linear_trend))
    
    #Calculate the term that will be multiplied with the
    #time signal to determine noise Beta weights
    XT_X_Neg1_XT = calculate_XT_X_Neg1_XT(X)
    
    
    #Apply nuisance regression/filtered
    for i in range(0, uncleaned_ts.shape[0]):
        #Have already skipped vols everything but uncleaned_ts
        temp_signal = partial_clean_fast(uncleaned_ts[i,skip_vols:], XT_X_Neg1_XT, bad_regressors)
        
        if high_pass != False:
            new_time_signal[i,:] = apply_filter(b, a, temp_signal)
        else:
            new_time_signal[i,:] = temp_signal
        
            
        
    #Implement scrubbing at this point, maybe earlier
    if scrub_thresh != False:
        
        fd = np.nan_to_num(fd_arr[skip_vols:])
        fd_exclusion = np.zeros((new_time_signal.shape[1]))
        #iterate through all timepoints
        for i in range(1, func_data.shape[1]):
            #move forward if current fd is greater than
            #the fd exclusion threshold
            if fd[i] >= scrub_thresh:

                #If it is the last timepoint, then exclude
                #the current timepoint and previous timepoint
                if i == new_time_signal.shape[1]:
                    fd_exclusion[i] = 1
                    fd_exclusion[i-1] = 1

                #Otherwise, exclude the previous, current, and
                #next timepoint. Note - the first timepoint will
                #always have a framewise displacement of 0, so we
                #don't need to put in an exception for this case
                if i != new_time_signal.shape[1]:
                    fd_exclusion[i+1] = 1
                    fd_exclusion[i] = 1
                    fd_exclusion[i-1] = 1

        #now we will go back over the fd_exclusion array, and get
        #rid of any snips where there aren't at least 3 consecutive
        #good timepoints
        bad_inds = np.where(fd_exclusion == 1)[0]
        for i in range(0, bad_inds.shape[0] - 1):
            if (bad_inds[i+1] - bad_inds[i]) < 4:
                for j in range(bad_inds[i] + 1, bad_inds[i+1]):
                    fd_exclusion[j] = 1000

        #Now we have the good inds to reference during other denoising steps
        bad_inds = np.where(fd_exclusion > 0.01)[0]
        good_inds = np.where(fd_exclusion < 0.01)[0]
        
        scrubbed_time_signal = new_time_signal[:,good_inds]
        new_time_signal = scrubbed_time_signal
        
    return new_time_signal

########################################################################################    
########################################################################################
########################################################################################


def calculate_XT_X_Neg1_XT(X):
    
    #Calculate term that can be multiplied with
    #Y to calculate the beta weights for least
    #squares regression. X should be of shape
    #(n x d) where n is the number of observations
    #and d is the number of dimensions/predictors
    #uses inverse transform
    
    XT = X.transpose()
    XT_X_Neg1 = np.linalg.pinv(np.matmul(XT,X))
    return np.matmul(XT_X_Neg1, XT)
    
########################################################################################    
########################################################################################
########################################################################################

def partial_clean_fast(Y, XT_X_Neg1_XT, bad_regressors):
    
    #Function to help in the denoising of time signal Y with shape
    #(n,1) or (n,) where n is the number of timepoints. 
    #XT_X_Neg1_XT is ((X^T)*X)^-1*(X^T), where ^T represents transpose
    #and ^-1 represents matrix inversions. X contains bad regressors including
    #noise ICs, a constant component, and a linear trend (etc.), and good regressors
    #containing non-motion related ICs. The Beta weights for the linear model
    #will be solved by multiplying XT_X_Neg1_XT with Y, and then the beta weights
    #determined for the bad regressors will be subtracted off from Y and the residuals
    #from this operation will be returned. For this reason, it is important to
    #put all bad regressors in front when doing matrix multiplication

    B = np.matmul(XT_X_Neg1_XT, Y)
    Y_noise = np.matmul(bad_regressors, B[:bad_regressors.shape[1]])
    return (Y - Y_noise)

    
########################################################################################    
########################################################################################
########################################################################################
########################################################################################
########################################################################################
    
    
from scipy.signal import butter, filtfilt
def construct_filter(btype, cutoff, TR, order):
    
    #btype should be 'lowpass', 'highpass', or 'bandpass' and
    #cutoff should be list (in Hz) with length 1 for low and high and
    #2 for band. Order is the order of the filter
    #which will be doubled since filtfilt will be used
    #to remove phase distortion from the filter. Recommended
    #order is 6. Will return filter coefficients b and a for
    #the desired butterworth filter.
    
    #Constructs filter coefficients. Use apply_filter to use
    #the coefficients to filter a signal.
    
    #Should have butter imported from scipy.signal
    
    
    nyq = 0.5 * (1/TR)
    
    if btype == 'lowpass':
        if len(cutoff) != 1:
            raise NameError('Error: lowpass type filter should have one cutoff values')
        low = cutoff[0]/nyq
        b, a = butter(order, low, btype='lowpass')
        
    elif btype == 'highpass':
        if len(cutoff) != 1:
            raise NameError('Error: highpass type filter should have one cutoff values')
        high = cutoff[0]/nyq
        b, a = butter(order, high, btype='highpass')
        
    elif btype == 'bandpass':
        if len(cutoff) != 2:
            raise NameError('Error: bandpass type filter should have two cutoff values')
        low = min(cutoff)/nyq
        high = max(cutoff)/nyq
        b, a = butter(order, [low, high], btype='bandpass')
        
    else: 
        raise NameError('Error: filter type should by low, high, or band')
        
        
    return b, a


########################################################################################
########################################################################################
########################################################################################

def apply_filter(b, a, signal):
    #Wrapper function to apply the filter coefficients from
    #construct_filter to a signal.
    
    #should have filtfilt imported from scipy.signal
    
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal
    
########################################################################################
########################################################################################
########################################################################################
    
def partial_clean(time_signal_of_interest, nuisance_vars, good_vars, skip_vols):
    
    #Function that solves for the time signal of interest as a function
    #of the variables stored in nuisance_vars and good_vars and then
    #subtracts the influence of nuisance_vars from the time signal of
    #interest and returns the residual time signal. The first few volumes
    #can be skipped with skip_vols. If you don't want to skip any volumes,
    #set skip_vols = 0. Also adds a constant and linear detrend to the model
    
        time_signal_of_interest = time_signal_of_interest.reshape((time_signal_of_interest.shape[0], 1))
        constant = np.ones(time_signal_of_interest.shape)
        linear_trend = np.linspace(0,time_signal_of_interest.shape[0], 
                                   time_signal_of_interest.shape[0]).reshape(time_signal_of_interest.shape) 
        full_conf = np.hstack((nuisance_vars, constant, linear_trend, good_vars))
        mdl = sm.regression.linear_model.OLS(time_signal_of_interest[skip_vols:],full_conf[skip_vols:,:])
        results = mdl.fit()
        confound_signal = np.zeros(nuisance_vars.shape[0] - skip_vols)
        for j in range(0,nuisance_vars.shape[1]):
            confound_signal = np.add(confound_signal, full_conf[skip_vols:,j]*results.params[j])
        confound_signal = np.add(confound_signal, full_conf[skip_vols:,nuisance_vars.shape[1]]*results.params[nuisance_vars.shape[1]])
        confound_signal = np.add(confound_signal, full_conf[skip_vols:,nuisance_vars.shape[1]+1]*results.params[nuisance_vars.shape[1]+1])

        return np.subtract(time_signal_of_interest[skip_vols:].reshape(time_signal_of_interest.shape[0] - skip_vols), confound_signal)
    

########################################################################################
########################################################################################
########################################################################################
    
def hard_clean(time_signal_of_interest, nuisance_vars, skip_vols):
    
        full_conf = nuisance_vars
        mdl = sm.regression.linear_model.OLS(time_signal_of_interest[skip_vols:],full_conf[skip_vols:,:])
        results = mdl.fit()
        confound_signal = np.zeros(nuisance_vars.shape[0] - skip_vols)
        for j in range(0,nuisance_vars.shape[1]):
            confound_signal = np.add(confound_signal, full_conf[skip_vols:,j]*results.params[j])
            
        return np.subtract(time_signal_of_interest[skip_vols:].reshape(time_signal_of_interest.shape[0] - skip_vols), confound_signal)
    
    
def reduce_noise_ics(matrix_of_noise_ics, num_dimensions):
    
    #Function to do a PCA decomposition of noise independent
    #components (or anything else) and return num_dimensions
    #many time signals. matrix_of_noise_ics should have shape
    #num_timepoints x num_components
    
    pca_temp = PCA()
    pca_temp.fit(matrix_of_noise_ics)
    
    pca_time_signal = np.zeros((matrix_of_noise_ics.shape[0], num_dimensions))

    for i in range(0, num_dimensions):
        for j in range(0, pca_temp.n_components_):
            pca_time_signal[:,i] += pca_temp.components_[i,j]*matrix_of_noise_ics[:,j]
            
    return pca_time_signal

########################################################################################
########################################################################################
########################################################################################

def output_stats_figures_pa_ap_compare(cleaned_ap, cleaned_pa):
    cleaned_ap_netmat = np.corrcoef(cleaned_ap)
    cleaned_pa_netmat = np.corrcoef(cleaned_pa)

    plt.figure()
    plt.imshow(cleaned_ap_netmat)
    plt.colorbar()
    plt.title('AP Conn Matrix')
    plt.figure()
    cleaned_ap.shape

    plt.imshow(cleaned_pa_netmat)
    plt.colorbar()
    plt.title('PA Conn Matrix')
    plt.figure()

    corr_dif = cleaned_ap_netmat - cleaned_pa_netmat
    plt.imshow(np.abs(corr_dif), vmin=0, vmax=0.1)
    plt.title('abs(AP - PA)')
    plt.colorbar()
    plt.figure()

    plt.hist(np.abs(np.reshape(corr_dif, corr_dif.shape[0]**2)), bins = 20)
    plt.title('abs(AP - PA) mean = ' + str(np.mean(np.abs(corr_dif))))

    ap_arr = cleaned_ap_netmat[np.triu_indices(cleaned_ap_netmat.shape[0], k = 1)]
    pa_arr = cleaned_pa_netmat[np.triu_indices(cleaned_pa_netmat.shape[0], k = 1)]
    plt.figure()
    plt.scatter(ap_arr, pa_arr)
    plt.title('AP-PA corr: ' + str(np.corrcoef(ap_arr, pa_arr)[0,1]))
    

def gather_cleaning_vars(path_to_func, list_of_confounds):
    
    #For a functional path (must be pointing to fsaverage),
    #and a list of confounds (from *desc-confounds_regressors.tsv).
    #This function will make two matrices of shape (t x n), where
    #t is the number of timepoints, and n the number of regressors.
    #The first matrix will contain 'nuisance_vars' which will be
    #a combination of the variables from list_of_confounds, and
    #independent components identified as noise by ICA-AROMA. 
    #The second will contain the indpendent components not identified
    #by ICA-AROMA, which are presumed to contain meaningful functional
    #data
    
    melodic_mixing_path = path_to_func[:-31] + 'desc-MELODIC_mixing.tsv'
    confound_path = path_to_func[:-31] + 'desc-confounds_regressors.tsv'
    noise_ics_path = path_to_func[:-31] + 'AROMAnoiseICs.csv'


    #Get the AP variables for cleaning
    confound_df = pd.read_csv(confound_path, sep='\t')
    partial_confounds = []
    for conf_name in list_of_confounds:
        temp = confound_df.loc[ : , conf_name ]
        partial_confounds.append(np.copy(temp.values)) 

    noise_comps = (genfromtxt(noise_ics_path, delimiter=',') - 1).astype(int)
    melodic_df = pd.read_csv(melodic_mixing_path, sep="\t", header=None)
    all_ics = melodic_df.values 

    mask = np.zeros(all_ics.shape[1],dtype=bool)
    mask[noise_comps] = True
    clean_ts = all_ics[:,~mask]  # array([0, 1, 2, 3])
    noise_ts = all_ics[:,mask] # array([4, 5, 6, 7, 8, 9])

    if len(list_of_confounds) > 0:
        non_ic_confounds = np.vstack(partial_confounds)
        full_noise_confounds = np.hstack((noise_ts, non_ic_confounds.transpose()))
    else:
        full_noise_confounds = noise_ts
    
    return full_noise_confounds, clean_ts



def gather_confounds(path_to_func, confounds):
    
    #Returns a numpy array with the confounds specified
    #in the list "confounds". path_to_func must be pointing to
    #an fsaverage file to work (not fsaverage5).
    
    confound_path = path_to_func[:-31] + 'desc-confounds_regressors.tsv'

    confound_df = pd.read_csv(confound_path, sep='\t')
    partial_confounds = []
    for conf_name in confounds:
        temp = confound_df.loc[ : , conf_name ]
        partial_confounds.append(np.copy(temp.values)) 
    
    return np.vstack(partial_confounds)


def find_mean_fd(path_to_func):
    
    #For a functional path (must be pointing to fsaverage),
    #and a list of confounds (from *desc-confounds_regressors.tsv).
    #This function will make two matrices of shape (t x n), where
    #t is the number of timepoints, and n the number of regressors.
    #The first matrix will contain 'nuisance_vars' which will be
    #a combination of the variables from list_of_confounds, and
    #independent components identified as noise by ICA-AROMA. 
    #The second will contain the indpendent components not identified
    #by ICA-AROMA, which are presumed to contain meaningful functional
    #data
    
    confound_path = path_to_func[:-31] + 'desc-confounds_regressors.tsv'

    confound_df = pd.read_csv(confound_path, sep='\t')
    partial_confounds = []
    temp = confound_df.loc[ : , 'framewise_displacement' ]
    fd_arr = np.copy(temp.values)
    
    return np.mean(fd_arr[1:])


def convert_to_upper_arr(np_square_matrix):
    
    #Function that takes a square matrix,
    #and outputs its upper triangle without
    #the diagonal as an array
    

    inds = np.triu_indices(np_square_matrix.shape[0], k = 1)
    return np_square_matrix[inds]


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
    
    
    def __init__(self, lh_gii_path, lh_parcellation_path, TR):
        
        self.lh_func_path = lh_gii_path
        self.rh_func_path = lh_gii_path[:-10] + 'R.func.gii'
        self.lh_parcellation_path = lh_parcellation_path
        
        lh_parcel_end = lh_parcellation_path.split('/')[-1]
        self.rh_parcellation_path = lh_parcellation_path[:-len(lh_parcel_end)] + 'r' + lh_parcellation_path.split('/')[-1][1:]

        self.TR = TR
        
        
        temp_path = lh_gii_path.split('/')
        end_path = temp_path[-1:][0]
        split_end_path = end_path.split('_')
        
        self.subject = split_end_path[0]
        
        if split_end_path[1][0:3]:
            self.session = split_end_path[1]
        else:
            self.session = []
                
        self.melodic_mixing_path = lh_gii_path[:-len(end_path)] + end_path[:-31] + 'desc-MELODIC_mixing.tsv'
        self.aroma_noise_ics_path = lh_gii_path[:-len(end_path)]+ end_path[:-31] + 'AROMAnoiseICs.csv'
        self.confounds_regressors_path = lh_gii_path[:-len(end_path)] + end_path[:-31] + 'desc-confounds_regressors.tsv'
        
        
        
        #Fields to be populated when the function "populate_all_fields" is ran
        self.time_series = [] #implemented
        self.parc_labels = [] #implemented
        self.n_skip_vols = [] #implemented
        self.mean_fd = [] #implemented
        self.melodic_mixing = [] #implemented
        self.aroma_noise_ic_inds = [] #implemented
        self.aroma_clean_ics = [] #implemented
        self.aroma_noise_ics = [] #implemented
        self.confounds = [] #implemented
        
        
        
        
        
    def populate_all_fields(self):
        
        ##############################################################################
        #Load the timeseries data and apply parcellation, saving also the parcel labels
        lh_data = imaging_utils.load_gifti_func(self.lh_func_path)
        rh_data = imaging_utils.load_gifti_func(self.rh_func_path)
        self.time_series, self.parc_labels = imaging_utils.parcellate_func_combine_hemis(lh_data, rh_data, self.lh_parcellation_path, self.rh_parcellation_path)
        
        ####################################################
        #Load the melodic IC time series
        melodic_df = pd.read_csv(self.melodic_mixing_path, sep="\t", header=None)
        self.melodic_mixing = melodic_df.values 
        
        ####################################################
        #Load the indices of the aroma ics
        aroma_ics_df = pd.read_csv(self.aroma_noise_ics_path, header=None)
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
        self.confounds = confounds_class(self.confounds_regressors_path)
        
        
        
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


        
import scipy.interpolate as interp
def interpolate(timepoint_defined, signal, interp_type):
    #defined_timepoints should be an array the length of the t with True at timepoints
    #that are defined and False at timepoints that are not defined. signal should also
    #be an array of length t. Timepoints at defined as False will be overwritten. This
    #script supports extrapolation at beginning/end of the time signal. As a quality control
    #for the spline interpolation, the most positive/negative values observed in the defined
    #portion of the signal are set as bounds for the interpolated signal
    
    #interpolation types supported:
    
        #(1) linear - takes closest point before/after undefined timepoint and interpolates.
        #    in end cases, uses the two points before/after
        #(2) cubic_spline - takes 5 closest time points before/after undefined timepoints
        #and applies cubic spline to undefined points. Uses defined signal to determine maximum/minimum
        #bounds for new interpolated points.
        #(3) spectral - yet to be implemented, will be based off of code from the 2014 Power
        #    paper

    timepoint_defined = np.array(timepoint_defined)
    
    true_inds = np.where(timepoint_defined == True)[0]
    false_inds = np.where(timepoint_defined == False)[0]
    

    signal_copy = np.array(signal)
    
    if interp_type == 'linear':
        
        #Still need to handle beginning/end cases
        
        for temp_timepoint in false_inds:
            
            
            #past_timepoint = true_inds[np.sort(np.where(true_inds < temp_timepoint)[0])[-1]]
            #future_timepoint = true_inds[np.sort(np.where(true_inds > temp_timepoint)[0])[0]]
            
            
            #Be sure there is at least one future timepoint and one past timepoint.
            #If there isn't, then grab either two past or two future timepoints and use those
            #for interpolation. If there aren't even two total past + future timepoints, then
            #just set the output to 0. Could also set the output to be unadjusted, but this
            #is a way to make the issue more obvious.
            temp_past_timepoint = np.sort(np.where(true_inds < temp_timepoint)[0])
            temp_future_timepoint = np.sort(np.where(true_inds > temp_timepoint)[0])
            
            #If we don't have enough data to interpolate/extrapolate
            if len(temp_past_timepoint) + len(temp_future_timepoint) < 2:
                
                signal_copy[temp_timepoint] = 0
               
            #If we do have enough data to interpolate/extrapolate
            else:
                
                if len(temp_past_timepoint) == 0:
                    past_timepoint = true_inds[temp_future_timepoint[1]]
                else:
                    past_timepoint = true_inds[temp_past_timepoint[-1]]

                if len(temp_future_timepoint) == 0:
                    future_timepoint = true_inds[temp_past_timepoint[-2]]
                else:
                    future_timepoint = true_inds[temp_future_timepoint[0]]

                #Find the appopriate past/future values
                past_value = signal_copy[int(past_timepoint)]
                future_value = signal_copy[int(future_timepoint)]

                #Use the interp1d function for interpolation
                interp_object = interp.interp1d([past_timepoint, future_timepoint], [past_value, future_value],
                                                bounds_error=False, fill_value='extrapolate')
                signal_copy[temp_timepoint] = interp_object(temp_timepoint).item(0)

        return signal_copy
            
    
    #For cubic spline interpolation, instead of taking the past/future timepoint
    #we will just take the closest 5 timepoints. If there aren't 5 timepoints, we will
    #set the output to 0
    if interp_type == 'cubic_spline':
        
        sorted_good = np.sort(signal_copy[true_inds])
        min_bound = sorted_good[0]
        max_bound = sorted_good[-1]
        
        #Continue if there are at least 5 good inds
        true_inds_needed = 5
        if len(true_inds) >= true_inds_needed:
        
            for temp_timepoint in false_inds:

                closest_inds = true_inds[np.argsort(np.absolute(true_inds - temp_timepoint))]
                closest_vals = signal_copy[closest_inds.astype(int)]
                interp_object = interp.interp1d(closest_inds, closest_vals, kind = 'cubic', bounds_error=False,
                                                fill_value='extrapolate')
                signal_copy[temp_timepoint.astype(int)] = interp_object(temp_timepoint).item(0)
                
            min_bound_exceded = np.where(signal_copy < min_bound)[0]
            if len(min_bound_exceded) > 0:
                
                signal_copy[min_bound_exceded] = min_bound
                
            max_bound_exceded = np.where(signal_copy > max_bound)[0]
            if len(max_bound_exceded) > 0:
                
                signal_copy[max_bound_exceded] = max_bound
             
        #If there aren't enough good timepoints, then set the bad timepoints = 0
        else:
        
            signal_copy[false_inds.astype(int)] = 0
                                               
                                               
        return signal_copy
    
    
def demedian_parcellate_func_combine_hemis(lh_func, rh_func, lh_parcel_path, rh_parcel_path):
    

    #Function that takes functional data in the form <num_verts, num_timepoints> for
    #both the left and right hemisphere, and averages the functional time series across
    #all vertices defined in a given parcel, for every parcel, with the parcels identified
    #by a annotation file specified at ?h_parcel_path. The function then returns a combined
    #matrix of size <num_parcels, num_timepoints> and <num_labels> for the time series and
    #parcel label names, respectively. The lh parcels will preceed the rh parcels in order.

    #Prior to taking the average of all vertices, all vertices time signals are divided by their
    #median signal intensity. The mean of all these medians within a given parcel is then 
    #exported with this function as the third argument

    #NOTE: THIS ASSUMES THE FIRST PARCEL WILL BE MEDIAL WALL, AND DISREGARDS ANY VERTICES WITHIN
    #THAT PARCEL. IF THIS IS NOT THE CASE FOR YOUR PARCELLATION, DO NOT USE THIS FUNCTION.

    #Output will be tuple of format [labels, ctab, names]
    lh_parcels = nib.freesurfer.io.read_annot(lh_parcel_path)
    rh_parcels = nib.freesurfer.io.read_annot(rh_parcel_path)

    #Make array to store parcellated data with shape <num_parcels, num_timepoints>
    lh_parcellated_data = np.zeros((len(lh_parcels[2]) - 1, lh_func.shape[1]))
    rh_parcellated_data = np.zeros((len(rh_parcels[2]) - 1, rh_func.shape[1]))
    lh_parcel_medians = np.zeros(len(lh_parcels[2]) - 1)
    rh_parcel_medians = np.zeros(len(rh_parcels[2]) - 1)


    lh_vertex_medians = np.nanmedian(lh_func, axis=1)
    rh_vertex_medians = np.nanmedian(rh_func, axis=1)

    lh_vertex_medians[np.where(lh_vertex_medians < 0.001)] = np.nan
    rh_vertex_medians[np.where(rh_vertex_medians < 0.001)] = np.nan

    lh_adjusted_func = lh_func/lh_vertex_medians[:,None]
    rh_adjusted_func = rh_func/rh_vertex_medians[:,None]




    #Start with left hemisphere
    for i in range(1,len(lh_parcels[2])):

        #Find the voxels for the current parcel
        vois = np.where(lh_parcels[0] == i)

        #Take the mean of all voxels of interest
        lh_parcellated_data[i-1, :] = np.nanmean(lh_adjusted_func[vois[0],:], axis = 0)
        lh_parcel_medians[i-1] = np.nanmean(lh_vertex_medians[vois[0]])

    #Move to right hemisphere
    for i in range(1,len(rh_parcels[2])):

        vois = np.where(rh_parcels[0] == i)
        rh_parcellated_data[i-1, :] = np.nanmean(rh_adjusted_func[vois[0],:], axis = 0)
        rh_parcel_medians[i-1] = np.nanmean(rh_vertex_medians[vois[0]])

    #Then concatenate parcel labels and parcel timeseries between the left and right hemisphere
    #and drop the medial wall from label list
    parcellated_data = np.vstack((lh_parcellated_data, rh_parcellated_data))
    parcel_labels = lh_parcels[2][1:] + rh_parcels[2][1:]
    parcel_medians = np.hstack((lh_parcel_medians, rh_parcel_medians))

    #Try to convert the parcel labels from bytes to normal string
    for i in range(0, len(parcel_labels)):
        parcel_labels[i] = parcel_labels[i].decode("utf-8")    

    return parcellated_data, parcel_labels, parcel_medians
