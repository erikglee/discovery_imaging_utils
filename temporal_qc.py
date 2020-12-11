import inspect


def make_temporal_qc_reports(path_to_boldref, path_to_output_dir, path_to_atlas = None):

    if type(path_to_atlas) == type(None):
        path_to_lh_atlas = path_to_harv_oxf_mni = '/'.join(os.path.abspath(inspect.getfile(construct_report)).split('/')[:-1]) + '/reference_data/lh.Schaefer2018_400Parcels_7Networks_order.annot'
