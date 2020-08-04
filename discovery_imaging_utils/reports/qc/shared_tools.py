import os


def construct_report(subject_path, report_path, dataset_qc_location = None, overwrite = False):
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



    construct_ind_structural_qc(subject_path, individual_structural_path)
    construct_ind_functional_qc(subject_path, individual_functional_path)
    #construct_group_structural_qc(subject_path, report_path)
    #construct_group_functional_qc(subject_path, report_path)
