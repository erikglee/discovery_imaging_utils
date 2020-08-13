import os
from discovery_imaging_utils.reports.qc import ind_functional_qc
from discovery_imaging_utils.reports.qc import ind_group_functional_qc
from discovery_imaging_utils.reports.qc import ind_structural_qc
from discovery_imaging_utils.reports.qc import ind_group_structural_qc
import glob
import inspect
import pandas as pd
import numpy as np


def construct_report(subject_fmriprep_path, subject_fs_path, report_path, structural_reference_csv_path = None, functional_reference_csv_path = None, overwrite = False):
    """Function to construct QC report from fMRIPREP output


    Parameters
    ----------

    subject_path : str
        path to a subject's fmriprep folder

    report_path : str
        path where the report should be saved

    dataset_qc_location : str or None
        if there are precalculated distributions of QC
        metrics for this dataset, give the path to the
        metrics and then group statistics will be included
    overwrite : boolean
        whether or not to overwrite the report


    """

    structural_qc_path = os.path.join(report_path, 'structural_qc')
    functional_qc_path = os.path.join(report_path, 'functional_qc')

    if os.path.exists(report_path) and (overwrite == False):
        raise NameError('Error: report path already exists')
    elif os.path.exists(report_path) == False:
        os.makedirs(report_path)
    if os.path.exists(structural_qc_path) == False:
        os.makedirs(structural_qc_path)
    if os.path.exists(functional_qc_path) == False:
        os.makedirs(functional_qc_path)

    #If path path to reference files not given, then
    #use template references stored in reference_data folder
    if type(structural_reference_csv_path) == type(None):
        structural_reference_csv_path = '/'.join(os.path.abspath(inspect.getfile(construct_report)).split('/')[:-1]) + '/reference_data/structural_reference_data.csv'
    if type(functional_reference_csv_path) == type(None):
        functional_reference_csv_path = '/'.join(os.path.abspath(inspect.getfile(construct_report)).split('/')[:-1]) + '/reference_data/functional_reference_data.csv'



    ind_functional_qc.construct_report(subject_fmriprep_path, functional_qc_path)
    ind_group_functional_qc.construct_report(subject_fmriprep_path, functional_qc_path, functional_reference_csv_path)
    ind_structural_qc.construct_report(subject_fmriprep_path, structural_qc_path)
    ind_group_structural_qc.construct_report(subject_fs_path, structural_qc_path, structural_reference_csv_path)

    _construct_html(report_path)


def _construct_html(report_path):
    """Internal function to make html out of QC

    Function builds an html structure to visualize report contents
    found under report_path

    Parameters
    ----------

    report_path : str
        path to the report folder that already contains report contents
        for the subject

    """

    functional_folder_name = os.path.join(report_path, 'functional_qc')
    structural_folder_name = os.path.join(report_path, 'structural_qc')

    html_folder = os.path.join(report_path, 'html')
    if os.path.exists(html_folder) == False:
        os.mkdir(html_folder)

    main_page_path = os.path.join(report_path, 'qc_report.html')
    with open(main_page_path, 'w') as html_file:

        startup_html_path = '/'.join(os.path.abspath(inspect.getfile(construct_report)).split('/')[:-1]) + '/reference_data/initial_html_content.txt'
        with open(startup_html_path, 'r') as startup_file:
            startup_contents = startup_file.read()

        html_file.write(startup_contents)

        if len(glob.glob(os.path.join(structural_folder_name, '*'))) > 0:
            contents_to_add = _construct_structural_html(report_path)
            html_file.write(contents_to_add)

        functional_runs = glob.glob(os.path.join(functional_folder_name, '*task*'))
        for temp_run in functional_runs:
            html_file.write('<p> </p>')
            contents_to_add = _construct_functional_html(report_path, temp_run)
            html_file.write(contents_to_add)



def _construct_structural_html(report_path):

    os.chdir(report_path)
    structural_html_path = os.path.join(report_path, 'html', 'structural_qc.html')
    with open(structural_html_path, 'w') as temp_html:

        temp_html.write('<h1>Structural Quality Control Report</h1>\n')
        temp_html.write('<h2>FreeSurfer Cortical Segmentation</h2>\n')
        temp_html.write('<a src="../structural_qc/fs_segmentation.jpeg">FreeSurfer Cortical Segmentation</a>\n')
        temp_html.write('<h2>Segmentation of GM/WM/CSF for Nuisance Regression</h2>\n')
        temp_html.write('<a src="../structural_qc/t1_regressors.jpg">Segmentation of ROIs for Nuisance Regression</a>\n')
        temp_html.write('<h2>MNI Alignment to Harvard Oxford Subcortical ROIs</h2>\n')
        temp_html.write('<a src="../structural_qc/mni_harv_oxf.jpg">Harvard Oxford Alignment</a>\n')
        temp_html.write('<h2>Normalized FreeSurfer QC Statistics</h2>\n')

        with open('./structural_qc/table.html', 'r') as table_file:
            table_contents = table_file.read()

        temp_html.write(table_contents)

#    stats = pd.read_csv('./structural_qc/subject_qc_stats.csv')
#    max_err = np.nanmax(np.abs(stats.values))
#    if max_err < 3:
#        color = 'green'
#    elif max_err < 4:
#        color = 'yellow'
    #else:
    #    color = 'red'

    html_output_txt = '<a href="./html/structural_qc.html">Structural QC</a>\n'


    return html_output_txt


def _construct_functional_html(report_path, temp_run):

    os.chdir(report_path)
    run_name = temp_run.split('/')[-1]
    if len(run_name) == 0:
        run_name = temp_run.split('/')[-2]
    functional_html_path = os.path.join(report_path, 'html', run_name + '.html')

    with open(functional_html_path, 'w') as temp_html:

        temp_html.write('<h1>Functional Quality Control Report (' + run_name + ')<h1>\n')
        temp_html.write('<h2>Structural Functional Alignment in Native Space</h2>\n')
        temp_html.write('<a src="' + os.path.join('../functional_qc', run_name, 't1_reg.jpeg') + '">Strucutral Functional Native Alignment</a>\n')
        temp_html.write('<h2>Structural Functional Alignment in MNI Space</h2>\n')
        temp_html.write('<a src="' + os.path.join('../functional_qc', run_name, 'mni_reg.jpeg') + '">Structural Functional MNI Alignment</a>\n')

        temp_html.write('<a src="' + os.path.join('../functional_qc', run_name, 'mean_gs.jpeg') + '"></a>\n')
        temp_html.write('<a src="' + os.path.join('../functional_qc', run_name, 'mean_dvars.jpeg') + '"></a>\n')
        temp_html.write('<a src="' + os.path.join('../functional_qc', run_name, 'num_high_std_dvars.jpeg') + '"></a>\n')
        temp_html.write('<a src="' + os.path.join('../functional_qc', run_name, 'num_high_motion_tps.jpeg') + '"></a>\n')
        temp_html.write('<a src="' + os.path.join('../functional_qc', run_name, 'mean_std_dvars.jpeg') + '"></a>\n')
        temp_html.write('<a src="' + os.path.join('../functional_qc', run_name, 'mean_fd.jpeg') + '"></a>\n')


    html_output_txt = '<a href="' + os.path.join('html',run_name + '.html') + '">Run ' + run_name + ' QC</a>\n'

    return html_output_txt
