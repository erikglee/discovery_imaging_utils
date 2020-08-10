from discovery_imaging_utils.reports.qc.visualizations import plt_ind_on_dist
import os
import glob
import pandas as pd
import numpy as np



def construct_report(subject_path, report_path, reference_csv_path):



    os.chdir(subject_path)
    subject_name = subject_path.split('/')[-1]
    reference_df = pd.read_csv(reference_csv_path)

    functional_images = glob.glob('./ses-*/func/sub*_boldref.nii.gz')
    functional_image_ids = []
    ses_ids = []
    for temp_img_path in functional_images:
        if 'space' not in temp_img_path:
            end_path = temp_img_path.split('/')[-1]
            ses_run = '_'.join(end_path.split('_')[1:5])
            functional_image_ids.append(ses_run)
            ses_ids.append(ses_run.split('_')[0])


    for i, temp_func_id in enumerate(functional_image_ids):
        temp_run = temp_func_id
        temp_ses = ses_ids[i]

        path_to_confounds = './' + temp_ses + '/func/' + subject_name + '_' + temp_run + '_desc-confounds_regressors.tsv'
        confounds_dict = calc_run_stats(path_to_confounds)


        run_report_path = os.path.join(report_path, temp_func_id)
        if os.path.exists(run_report_path) == False:
            os.makedirs(run_report_path)

        plt_ind_on_dist(reference_df['mean_std_dvars'].values, confounds_dict['mean_std_dvars'], xlabel='mean_std_dvars', out_path = os.path.join(run_report_path, 'mean_std_dvars.jpg'))
        plt_ind_on_dist(reference_df['num_high_std_dvars_tps'].values, confounds_dict['num_high_std_dvars_tps'], xlabel='num_high_std_dvars_tps', out_path = os.path.join(run_report_path, 'num_high_std_dvars_tps.jpg'))
        plt_ind_on_dist(reference_df['mean_fd'].values, confounds_dict['mean_fd'], xlabel='mean_fd', out_path = os.path.join(run_report_path, 'mean_fd.jpg'))
        plt_ind_on_dist(reference_df['num_high_motion_tps'].values, confounds_dict['num_high_motion_tps'], xlabel='num_high_motion_tps', out_path = os.path.join(run_report_path, 'num_high_motion_tps.jpg'))
        plt_ind_on_dist(reference_df['mean_dvars'].values, confounds_dict['mean_dvars'], xlabel='mean_dvars', out_path = os.path.join(run_report_path, 'mean_dvars.jpg'))
        plt_ind_on_dist(reference_df['mean_gs'].values, confounds_dict['mean_gs'], xlabel='mean_gs', out_path = os.path.join(run_report_path, 'mean_gs.jpg'))




def calc_run_stats(path_to_confounds, high_std_dvars_thresh = 1.5, high_motion_thresh = 0.5):
    """Function to extract values from confounds.tsv

    Parameters
    ----------

    path_to_confounds : str
        path to fmriprep confounds tsv file
    high_std_dvars_thresh : float, optional
        the threshold to use for determining
        high std_dvars timepoints (defaults
        to 1.5)
    high_motion_thresh : float, optional
        the threshold to use for determining
        high motion timepoints (defaults to 1.5)


    Returns
    -------
        output_dict : dict
            dictionary with different statistics from the
            confounds file

    """

    confounds_df = pd.read_csv(path_to_confounds, delimiter='\t')
    output_dict = {}


    output_dict['mean_gs'] = np.nanmean(confounds_df['global_signal'].values)
    output_dict['mean_std_dvars'] = np.nanmean(confounds_df['std_dvars'].values)
    output_dict['num_high_std_dvars_tps'] = np.where(confounds_df['std_dvars'] > high_std_dvars_thresh)[0].shape[0]
    output_dict['max_std_dvars'] = np.max(confounds_df['dvars'].values)
    output_dict['mean_dvars'] = np.nanmean(confounds_df['dvars'].values)
    output_dict['mean_fd'] = np.max(confounds_df['framewise_displacement'].values)
    output_dict['num_high_motion_tps'] = np.where(confounds_df['framewise_displacement'] > high_motion_thresh)[0].shape[0]
    output_dict['max_fd'] = np.max(confounds_df['framewise_displacement'].values)

    return output_dict

def make_reference_csv(path_to_fmriprep_dir, output_reference_csv_path):
    """Generate reference cohort func stats


    Parameters
    ----------

    path_to_fmriprep_dir : str
        path to directory of fmriprep output, all runs
        found under this directory will be used to construct
        norms

    output_reference_csv_path : str
        path to location where reference csv file
        will be stored.


    """

    mean_gs = []
    mean_std_dvars = []
    num_high_std_dvars_tps = []
    max_std_dvars = []
    mean_dvars = []
    mean_fd = []
    num_high_motion_tps = []
    max_fd = []

    os.chdir(path_to_fmriprep_dir)
    subjects = glob.glob('sub*')

    for temp_subj in subjects:

        subject_path = os.path.join(path_to_fmriprep_dir, temp_subj)
        if os.path.isdir(subject_path):
            os.chdir(subject_path)
            subject_name = subject_path.split('/')[-1]

            functional_images = glob.glob('./ses-*/func/sub*_boldref.nii.gz')
            functional_image_ids = []
            ses_ids = []
            for temp_img_path in functional_images:
                if 'space' not in temp_img_path:
                    end_path = temp_img_path.split('/')[-1]
                    ses_run = '_'.join(end_path.split('_')[1:5])
                    functional_image_ids.append(ses_run)
                    ses_ids.append(ses_run.split('_')[0])


            for i, temp_func_id in enumerate(functional_image_ids):
                temp_run = temp_func_id
                temp_ses = ses_ids[i]

                path_to_confounds = './' + temp_ses + '/func/' + subject_name + '_' + temp_run + '_desc-confounds_regressors.tsv'
                confounds_dict = calc_run_stats(path_to_confounds)

                mean_gs.append(confounds_dict['mean_gs'])
                mean_std_dvars.append(confounds_dict['mean_std_dvars'])
                num_high_std_dvars_tps.append(confounds_dict['num_high_std_dvars_tps'])
                max_std_dvars.append(confounds_dict['max_std_dvars'])
                mean_dvars.append(confounds_dict['mean_dvars'])
                mean_fd.append(confounds_dict['mean_fd'])
                num_high_motion_tps.append(confounds_dict['num_high_motion_tps'])
                max_fd.append(confounds_dict['max_fd'])


    temp_dict = {'mean_gs' : mean_gs,
                 'mean_std_dvars' : mean_std_dvars,
                 'num_high_std_dvars_tps' : num_high_std_dvars_tps,
                 'max_std_dvars' : max_std_dvars,
                 'mean_dvars' : mean_dvars,
                 'mean_fd' : mean_fd,
                 'num_high_motion_tps' : num_high_motion_tps,
                 'max_fd' : max_fd}

    output_df = pd.DataFrame(temp_dict)
    output_df.to_csv(output_reference_csv_path)
