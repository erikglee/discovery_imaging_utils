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



def load_cifti_func(path_to_file):

    cifti_img = nib_load(path_to_file)
    return np.asarray(cifti_img.dataobj).transpose()


def calc_fishers_icc(tp1, tp2):

    """
    #Calculate intraclass correlation coefficient
    #from the equation on wikipedia describing
    #fisher's formulation. tp1 and tp2 should
    # be of shape (n,1) or (n,) where n is the
    #number of samples
    """

    xhat = np.mean(np.vstack((tp1, tp2)))
    sq_dif1 = np.power((tp1 - xhat),2)
    sq_dif2 = np.power((tp2 - xhat),2)
    s2 = np.mean(np.vstack((sq_dif1, sq_dif2)))
    r = 1/(tp1.shape[0]*s2)*np.sum(np.multiply(tp1 - xhat, tp2 - xhat))

    return r




def pre_post_carpet_plot(noisy_time_series, cleaned_time_series):
    """
    #This function is for calculating a carpet plot figure, that
    #will allow for comparison of the BOLD time series before and
    #after denoising takes place. The two input matrices should have
    #shape <num_parcels, num_timepoints>, and will ideally be from a
    #parcellated time series and not whole hemisphere data (lots of points).

    #The script will demean and then normalize all regions' time signals,
    #and then will display them side by side on grey-scale plots
    """


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



def parcellate_func_combine_hemis(lh_func, rh_func, lh_parcel_path, rh_parcel_path):

    """
    #Function that takes functional data in the form <num_verts, num_timepoints> for
    #both the left and right hemisphere, and averages the functional time series across
    #all vertices defined in a given parcel, for every parcel, with the parcels identified
    #by a annotation file specified at ?h_parcel_path. The function then returns a combined
    #matrix of size <num_parcels, num_timepoints> and <num_labels> for the time series and
    #parcel label names, respectively. The lh parcels will preceed the rh parcels in order.

    #NOTE: THIS ASSUMES THE FIRST PARCEL WILL BE MEDIAL WALL, AND DISREGARDS ANY VERTICES WITHIN
    #THAT PARCEL. IF THIS IS NOT THE CASE FOR YOUR PARCELLATION, DO NOT USE THIS FUNCTION.
    """

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
    """
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
    """


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
    """
    #Function that takes a statistic defined at a parcel level, and
    #resamples that statistic to the network level. This function is a copy of
    #net_mat_summary_stats only now defined to work on 1D instead of 2D data.

    #This function only works on data formatted in the Schaeffer/Yeo 7 network
    #configuration.

    #Parcel labels should be a list of strings that has the names of the different
    #parcels in the parcellation. This is how the function knows what parcels
    #belong to what networks.
    """


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


def calc_norm_std(parcel_data, confound_path):
    """
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
    """



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
    """
    #This function serves the function of collecting the aseg.stats file,
    #lh.aparc.stats file, and rh.aparc.stats files from a freesurfer subject
    #found at the path path_to_fs_subject, and grabs the volumes for all
    #subcortical structures, along with volumes, thicknesses, and surface
    #areas for all cortical structures, and saves them as .npy files under
    #folder_for_output_files. Also saves a text file with the names of the
    #regions (one for subcortical, and one for lh/rh)
    """

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

    return


def calculate_pinv(X):

    """
    Here for QC....
    """

    return np.linalg.pinv(X)



def partial_clean_fast(Y, XT_X_Neg1_XT, bad_regressors):

    """
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
    """

    B = np.matmul(XT_X_Neg1_XT, Y)
    Y_noise = np.matmul(bad_regressors, B[:bad_regressors.shape[1]])
    return (Y - Y_noise)


from scipy.signal import butter, filtfilt
def construct_filter(btype, cutoff, TR, order):

    """
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
    """


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

    """
    #Wrapper function to apply the filter coefficients from
    #construct_filter to a signal.

    #should have filtfilt imported from scipy.signal
    """

    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


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
    """
    #Function that takes a square matrix,
    #and outputs its upper triangle without
    #the diagonal as an array
    """


    inds = np.triu_indices(np_square_matrix.shape[0], k = 1)
    return np_square_matrix[inds]
