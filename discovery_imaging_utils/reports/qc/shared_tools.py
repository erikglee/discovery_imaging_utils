import os
from discovery_imaging_utils.reports.qc import ind_functional_qc
from discovery_imaging_utils.reports.qc import ind_group_functional_qc
from discovery_imaging_utils.reports.qc import ind_structural_qc
from discovery_imaging_utils.reports.qc import ind_group_structural_qc


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

    individual_structural_path = os.path.join(report_path, 'ind_structural_qc')
    group_structural_path = os.path.join(report_path, 'group_structural_qc')
    individual_functional_path = os.path.join(report_path, 'ind_functional_qc')
    group_functional_path = os.path.join(report_path, 'group_functional_qc')

    if os.path.exists(report_path) and (overwrite == False):
        raise NameError('Error: report path already exists')
    elif os.path.exists(report_path) == False:
        os.makedirs(report_path)
    if os.path.exists(individual_structural_path) == False:
        os.makedirs(individual_structural_path)
    if os.path.exists(individual_functional_path) == False:
        os.makedirs(individual_functional_path)

    if type(structural_reference_csv_path) == type(None):
        #NEED TO IMPLEMENT
    if type(functional_reference_csv_path) == type(None):
        #NEED TO IMPLEMENT



    ind_functional_qc.construct_report(subject_fmriprep_path, individual_structural_path)
    ind_group_functional_qc.construct_report(subject_fmriprep_path, individual_functional_path)
    ind_structural_qc.construct_report(subject_fmriprep_path, )
    if type(structural_reference_csv_path) == type(None):
        #NEED TO IMPLEMENT
    ind_group_structural_qc.construct_report(subject_fs_path, group_structural_path, reference_csv_path, num_pcs=1, overwrite=False)
    #construct_group_structural_qc(subject_path, report_path)
    #construct_group_functional_qc(subject_path, report_path)
