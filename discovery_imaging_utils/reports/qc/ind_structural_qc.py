import glob
import os
import inspect

from discovery_imaging_utils.reports.qc.visualizations import make_outline_overlay_underlay_plot
from discovery_imaging_utils.reports.qc.visualizations import make_gmwmcsf_underlay_plot
from discovery_imaging_utils.reports.qc.visualizations import make_harv_oxf_qc_image
#Good for now


def construct_report(subject_path, report_path):
    """Construct structural QC report


    Parameters
    ----------

    subject_path : str
        path to subject's fMRIPREP output
    report_path : str
        path to folder where QC results will be stored


    """

    print('    running ind_structural_qc')

    subject_name = subject_path.split('/')[-1]
    if len(subject_name) == 0:
        subject_name = subject_path.split('/')[-2]

    os.chdir(subject_path)

    #Single subject image paths
    path_to_t1 = './anat/' + subject_name + '_desc-preproc_T1w.nii.gz'
    path_to_aparcaseg_dseg = './anat/' + subject_name + '_desc-aparcaseg_dseg.nii.gz'
    path_to_csf_mask = './anat/' + subject_name + '_label-CSF_probseg.nii.gz'
    path_to_wm_mask = './anat/' + subject_name + '_label-WM_probseg.nii.gz'
    path_to_gm_mask = './anat/' + subject_name + '_label-GM_probseg.nii.gz'

    #Reference image paths
    path_to_t1_mni = './anat/' + subject_name + '_space-MNI152NLin6Asym_desc-preproc_T1w.nii.gz'
    path_to_harv_oxf_mni = '/'.join(os.path.abspath(inspect.getfile(construct_report)).split('/')[:-1]) + '/HarvOxf-sub-maxprob-thr50-1mm.nii.gz'

    print(report_path)
    if os.path.exists(report_path) == False:
        os.makedirs(report_path)
    print(3.1)

    relevant_paths = [path_to_t1, path_to_aparcaseg_dseg, path_to_csf_mask, path_to_wm_mask, path_to_gm_mask, path_to_t1_mni, path_to_harv_oxf_mni]
    print(relevant_paths)
    for temp_path in relevant_paths:
        if os.path.exists(temp_path) == False:
            print('Missing: ' + temp_path)
        else:
            print('Found: ' + temp_path)

    print(4)

    make_outline_overlay_underlay_plot(path_to_t1, path_to_aparcaseg_dseg, aparcaseg=True, underlay_cmap='gray', output_path = os.path.join(report_path, 'fs_segmentation'))
    make_gmwmcsf_underlay_plot(path_to_t1, path_to_gm_mask, path_to_wm_mask, path_to_csf_mask, output_path = os.path.join(report_path, 't1_regressors'))
    make_harv_oxf_qc_image(path_to_t1_mni, path_to_harv_oxf_mni, output_path = os.path.join(report_path, 'mni_harv_oxf'))

    return
