import glob
import os
import inspect

#Good for now


def construct_ind_structural_qc(subject_path, report_path):


    subject_name = subject_path.split('/')[-1]
    os.chdir(subject_path)

    path_to_t1 = './anat/' + subject_name + '_desc-preproc_T1w.nii.gz'
    path_to_aparcaseg_dseg = './anat/' + subject_name + '_desc-aparcaseg_dseg.nii.gz'
    path_to_csf_mask = './anat/' + subject_name + '_label-CSF_probseg.nii.gz'
    path_to_wm_mask = './anat/' + subject_name + '_label-WM_probseg.nii.gz'
    path_to_gm_mask = './anat/' + subject_name + '_label-GM_probseg.nii.gz'

    path_to_t1_mni = './anat/' + subject_name + '_space-MNI152NLin6Asym_desc-preproc_T1w.nii.gz'
    path_to_harv_oxf_mni = '/'.join(os.path.abspath(inspect.getfile(construct_ind_structural_qc)).split('/')[:-1]) + '/HarvOxf-sub-maxprob-thr50-1mm.nii.gz'


    if os.path.exists(report_path) == False:
        os.makedirs(report_path)


    make_outline_overlay_underlay_plot(path_to_t1, path_to_aparcaseg_dseg, aparcaseg=True, underlay_cmap='gray', output_path = os.path.join(report_path, 'fs_segmentation'))
    make_gmwmcsf_underlay_plot(path_to_t1, path_to_gm_mask, path_to_wm_mask, path_to_csf_mask, output_path = os.path.join(report_path, 't1_regressors'))
    make_harv_oxf_qc_image(path_to_t1_mni, path_to_harv_oxf_mni, output_path = os.path.join(report_path, 'mni_harv_oxf'))
