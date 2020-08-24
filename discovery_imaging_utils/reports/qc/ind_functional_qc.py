import os
import glob
from discovery_imaging_utils.reports.qc.visualizations import make_outline_overlay_underlay_plot

def construct_report(subject_path, report_path):



    os.chdir(subject_path)
    subject_name = subject_path.split('/')[-1]
    if len(subject_name) == 0:
        subject_name = subject_path.split('/')[-2]
    path_to_t1 = './anat/' + subject_name + '_desc-preproc_T1w.nii.gz'

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

        path_to_dseg_T1 = './' + temp_ses + '/func/' + subject_name + '_' + temp_run + '_space-T1w_desc-aparcaseg_dseg.nii.gz'
        path_to_T1_boldref = './' + temp_ses + '/func/' + subject_name + '_' + temp_run + '_space-T1w_boldref.nii.gz'

        print(path_to_dseg_T1)
        print(path_to_T1_boldref)

        if (os.path.exists(path_to_dseg_T1) == False) or (os.path.exists(path_to_T1_boldref) == False):

            print('Missing image for run:' + temp_run)
            continue


        #path_to_MNI_boldref = './' + temp_ses + '/func/' + subject_name + '_' + temp_run + '_space-MNI152NLin6Asym_res-2_boldref.nii.gz'
        #path_to_dseg_MNI = './' + temp_ses + '/func/' + subject_name + '_' + temp_run + '_space-MNI152NLin6Asym_res-2_desc-aparcaseg_dseg.nii.gz'

        run_report_path = os.path.join(report_path, temp_func_id)
        if os.path.exists(run_report_path) == False:
            os.makedirs(run_report_path)

        make_outline_overlay_underlay_plot(path_to_T1_boldref, path_to_dseg_T1, aparcaseg=True, underlay_cmap='gray', wmgm=True, output_path=os.path.join(run_report_path, 't1_reg'))

        #MNI reg should be reduntant if we already know MNI reg works in
        #structurals
        #make_outline_overlay_underlay_plot(path_to_MNI_boldref, path_to_dseg_MNI, aparcaseg=True, underlay_cmap='gray', wm=True, output_path=os.path.join(run_report_path, 'mni_reg'))
