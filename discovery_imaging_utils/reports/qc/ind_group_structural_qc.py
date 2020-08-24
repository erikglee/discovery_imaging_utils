import glob
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import decomposition
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from pandas.plotting import table

from discovery_imaging_utils.freesurfer import fs_anat_to_dict
from discovery_imaging_utils.freesurfer import anat_dictionaries_to_csv


def make_reference_csv(fs_subjects_dir, output_csv_name):
    """Function to make reference table FreeSurfer stats

    Function takes path to FreeSurfer subjects dir and
    makes a csv file out of subject segmantation statistics

    TODO : automatically exclude subjects that have any NaNs

    Parameters
    ----------

    fs_subjects_dir : str
        path to location of FreeSurfer subjects dir
    output_csv_name : str
        path to csv file that will be created with output

    """

    os.chdir(fs_subjects_dir)
    subjects = glob.glob('*/stats')
    subj_anat_dicts = []

    for temp_subj in subjects:

        temp_subj = temp_subj.split('/')[0]
        subj_anat_dicts.append(fs_anat_to_dict(temp_subj, flatten=True))


    anat_dictionaries_to_csv(subj_anat_dicts, output_csv_name)

    return


def construct_report(subject_path, report_path, reference_csv_path, num_pcs=1, overwrite=False):
    """Function to make quality control tables from FreeSurfer output


    Function makes a quality control report for a single subject's
    FreeSurfer statistics using their output and a reference table
    from a given reference cohort. Statistics are from default aparc
    and aseg files. Computed statistics include vol_zscore,
    eTIV_adjusted_vol_zscore, prediction_error_zscore (where first
    PC across regions in reference cohort is fit to predict volume
    of each anatomical region), thickness_mean_zscore,
    thickness_std_zscore, and snr_zscore. Thickness only calculated
    for cortical structures, SNR only calculated for subcortical
    structures. Outputs the statistics as a csv file and as html
    file.

    Parameters
    ----------

    subject_path : str
        path to subject's FreeSurfer folder
    reference_csv_path : str
        path to file containing FreeSurfer data to use as
        reference, should be made from make_fs_reference_table
        function
    report_path : str
        path to folder that will be made for this subject's
        quality control output (tables, etc.)
    num_pcs : int, optional
        the number of principal components to be fit when
        predicting individual ROI statistics
    overwrite : bool, optional
        set to true if you want the function to continue if
        output_folder already exists

    """


    reference_df = pd.read_csv(reference_csv_path)
    subj_fs_dict = fs_anat_to_dict(subject_path, flatten=True)

    #remove any nans from reference_df
    reference_df = reference_df[reference_df.isna().any(axis=1) == False]

    #def make_fs_qc_stats(subject_fs_path, reference_csv_path, num_pcs = 1):

    region_ids = ['Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
            'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate',
            'Left-Putamen', 'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle',
            'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area',
            'Left-VentralDC', 'Right-Lateral-Ventricle',
            'Right-Inf-Lat-Vent', 'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
            'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum',
            'Right-Hippocampus', 'Right-Amygdala', 'Right-Accumbens-area', 'Right-VentralDC']

    parcel_ids = ['lh_bankssts', 'lh_caudalanteriorcingulate', 'lh_caudalmiddlefrontal',
                  'lh_cuneus', 'lh_entorhinal', 'lh_fusiform',
                  'lh_inferiorparietal', 'lh_inferiortemporal', 'lh_isthmuscingulate',
                  'lh_lateraloccipital', 'lh_lateralorbitofrontal', 'lh_lingual',
                  'lh_medialorbitofrontal', 'lh_middletemporal', 'lh_parahippocampal',
                  'lh_paracentral', 'lh_parsopercularis', 'lh_parsorbitalis',
                  'lh_parstriangularis', 'lh_pericalcarine', 'lh_postcentral',
                  'lh_posteriorcingulate', 'lh_precentral', 'lh_precuneus',
                  'lh_rostralanteriorcingulate', 'lh_rostralmiddlefrontal',
                  'lh_superiorfrontal', 'lh_superiorparietal',
                  'lh_superiortemporal', 'lh_supramarginal',
                  'lh_frontalpole', 'lh_temporalpole', 'lh_transversetemporal',
                  'lh_insula', 'rh_bankssts', 'rh_caudalanteriorcingulate',
                  'rh_caudalmiddlefrontal', 'rh_cuneus', 'rh_entorhinal',
                  'rh_fusiform', 'rh_inferiorparietal', 'rh_inferiortemporal',
                  'rh_isthmuscingulate', 'rh_lateraloccipital', 'rh_lateralorbitofrontal',
                  'rh_lingual', 'rh_medialorbitofrontal', 'rh_middletemporal',
                  'rh_parahippocampal', 'rh_paracentral', 'rh_parsopercularis',
                  'rh_parsorbitalis', 'rh_parstriangularis', 'rh_pericalcarine',
                  'rh_postcentral', 'rh_posteriorcingulate', 'rh_precentral',
                  'rh_precuneus', 'rh_rostralanteriorcingulate', 'rh_rostralmiddlefrontal',
                  'rh_superiorfrontal', 'rh_superiorparietal', 'rh_superiortemporal',
                  'rh_supramarginal', 'rh_frontalpole', 'rh_temporalpole',
                  'rh_transversetemporal', 'rh_insula']

    all_ids = region_ids + parcel_ids



    reference_vols = np.zeros((reference_df.shape[0], len(all_ids)))
    norm_reference_vols = np.zeros((reference_df.shape[0], len(all_ids)))
    reference_snr = np.zeros((reference_df.shape[0], len(all_ids)))


    reference_mean_thickness = np.zeros((reference_df.shape[0], len(all_ids)))
    reference_std_thickness = np.zeros((reference_df.shape[0], len(all_ids)))


    eTIV_id = 'extra_elements_eTIV_mm3'

    #Grab volumes and volumes normalized to eTIV
    vols = np.zeros(len(all_ids))
    norm_vols = np.zeros(len(all_ids))
    snr = np.zeros(len(all_ids))

    mean_thickness = np.zeros(len(all_ids))
    std_thickness = np.zeros(len(all_ids))
    for i, temp_id in enumerate(all_ids):

        if temp_id in region_ids:

            vols[i] = float(subj_fs_dict[temp_id + '_Volume_mm3'])
            norm_vols[i] = float(subj_fs_dict[temp_id + '_Volume_mm3'])/float(subj_fs_dict[eTIV_id])
            snr[i] = float(subj_fs_dict[temp_id + '_normMean'])/float(subj_fs_dict[temp_id + '_normStdDev'])


            reference_vols[:,i] = reference_df[temp_id + '_Volume_mm3']
            norm_reference_vols[:,i] = reference_df[temp_id + '_Volume_mm3']/reference_df[eTIV_id]
            reference_snr[:,i] = reference_df[temp_id + '_normMean']/reference_df[temp_id + '_normStdDev']

            mean_thickness[i] = np.nan
            std_thickness[i] = np.nan
            reference_mean_thickness[:,i] = np.nan
            reference_std_thickness[:,i] = np.nan

        else:

            vols[i] = float(subj_fs_dict[temp_id + '_GrayVol'])
            norm_vols[i] = float(subj_fs_dict[temp_id + '_GrayVol'])/float(subj_fs_dict[eTIV_id])
            snr[i] = np.nan


            reference_vols[:,i] = reference_df[temp_id + '_GrayVol']
            norm_reference_vols[:,i] = reference_df[temp_id + '_GrayVol']/reference_df[eTIV_id]
            reference_snr[:,i] = np.nan

            mean_thickness[i] = float(subj_fs_dict[temp_id + '_ThickAvg'])
            std_thickness[i] = float(subj_fs_dict[temp_id + '_ThickStd'])
            reference_mean_thickness[:,i] = reference_df[temp_id + '_ThickAvg']
            reference_std_thickness[:,i] = reference_df[temp_id + '_ThickStd']


    z_vols = scipy.stats.zscore(np.vstack((reference_vols, vols)))[-1,:]
    z_eTIV_normed_vols = scipy.stats.zscore(np.vstack((norm_reference_vols, norm_vols)))[-1,:]
    z_mean_thickness = scipy.stats.zscore(np.vstack((reference_mean_thickness, mean_thickness)), nan_policy='omit')[-1,:]
    z_std_thickness = scipy.stats.zscore(np.vstack((reference_std_thickness, std_thickness)), nan_policy='omit')[-1,:]
    z_snr = scipy.stats.zscore(np.vstack((reference_snr, snr)))[-1,:]


    nan_policy='omit'



    reference_vols = np.vstack((reference_vols, vols))
    z_normalized_ref_vols = scipy.stats.zscore(reference_vols, axis=0)
    prediction_errors = np.zeros(reference_vols.shape)


    pca_obj = sklearn.decomposition.PCA(n_components=num_pcs)
    pca_obj.fit(z_normalized_ref_vols)
    pca_scores = pca_obj.transform(z_normalized_ref_vols)

    for i in range(0,reference_vols.shape[1]):

        temp_lin_model = sklearn.linear_model.LinearRegression().fit(pca_scores, reference_vols[:,i])
        predictions = temp_lin_model.predict(pca_scores)
        temp_prediction_errors = reference_vols[:,i] - predictions
        prediction_errors[:,i] = temp_prediction_errors

    z_prediction_errors = scipy.stats.zscore(prediction_errors)[-1,:]




    #Now make dataframe to store results
    index_names = all_ids

    qc_df = pd.DataFrame(data=z_vols, index=index_names, columns=['vol_zscore'])
    qc_df['eTIV_adjusted_vol_zscore'] = z_eTIV_normed_vols
    qc_df['prediction_error_zscore'] = z_prediction_errors
    qc_df['thickness_mean_zscore'] = z_mean_thickness
    qc_df['thickness_std_zscore'] = z_std_thickness
    qc_df['snr_zscore'] = z_snr



    def absolute_viridis(s):

        #Function to do color
        #coating of viridis
        #where pos/neg values
        #have mirrrored colors

        vmax = 4

        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax, clip=False)
        cm = matplotlib.cm.ScalarMappable(norm=norm, cmap='viridis')
        c = [colors.rgb2hex(x) for x in cm.to_rgba(np.abs(s.values))]
        return ['background-color: %s' % color for color in c]


    styler = qc_df.style.apply(absolute_viridis)
    html_content = styler.render()

    if os.path.exists(report_path) == False:
        os.makedirs(report_path)

    with open(os.path.join(report_path, 'table.html'),'w') as temp_file:
        temp_file.write(html_content)

    qc_df.to_csv(os.path.join(report_path, 'subject_qc_stats.csv'))


    #Make a different plot for the number of surface holes
    num_lh_holes = subj_fs_dict{'extra_elements_lhSurfaceHoles'}
    num_rh_holes = subj_fs_dict{'extra_elements_rhSurfaceHoles'}

    num_entries = reference_df.shape[0]

    num_lh_holes_percentile = np.sum(reference_df['extra_elements_lhSurfaceHoles'].values < num_lh_holes)/(num_entries*100)
    num_rh_holes_percentile = np.sum(reference_df['extra_elements_rhSurfaceHoles'].values < num_rh_holes)/(num_entries*100)
    holes_stats = [num_lh_holes, num_lh_holes, percentile, num_rh_holes, num_rh_holes_percentile]
    holes_cols = ['LH Holes', 'LH Holes Percentile', 'RH Holes', 'RH Hoels Percentile']

    holes_df = pd.DataFrame(data=holes_stats, index=['Values'], columns=holes_cols)
    holes_df.to_csv(os.path.join(report_path, 'subject_holes_stats.csv'))



    return qc_df
