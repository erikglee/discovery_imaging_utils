import glob
import os
import pandas as pd
import numpy as np


def fs_anat_to_dict(path_to_fs_subject, flatten=False):
    #This function serves the function of collecting the aseg.stats file,
    #lh.aparc.stats file, and rh.aparc.stats files from a freesurfer subject
    #found at the path path_to_fs_subject, and grabs the volumes for all
    #subcortical structures, along with volumes, thicknesses, and surface
    #areas for all cortical structures, and saves them as .npy files under
    #folder_for_output_files. Also saves a text file with the names of the
    #regions (one for subcortical, and one for lh/rh)

    aseg_path = os.path.join(path_to_fs_subject, 'stats', 'aseg.stats')
    lh_aseg_path = os.path.join(path_to_fs_subject, 'stats', 'lh.aparc.stats')
    rh_aseg_path = os.path.join(path_to_fs_subject, 'stats', 'rh.aparc.stats')

    with open(aseg_path, 'r') as temp_file:
        aseg_contents = temp_file.readlines()

    with open(lh_aseg_path, 'r') as temp_file:
        lh_aparc_contents = temp_file.readlines()

    with open(rh_aseg_path, 'r') as temp_file:
        rh_aparc_contents = temp_file.readlines()

    anat_dictionary = {}

    header_found = False
    for temp_line in aseg_contents:
        if '# ColHeaders' in temp_line:
            headers = temp_line.split()[2:]
            header_found = True
        elif header_found:
            temp_dict = {}
            line_values = temp_line.split()
            if len(line_values) > 2:
                if len(line_values) != len(headers):
                    raise NameError('Error')
                else:
                    for i, temp_key in enumerate(headers):
                        temp_dict[temp_key] = line_values[i]
            anat_dictionary[line_values[4]] = temp_dict.copy()

    header_found = False
    for temp_line in lh_aparc_contents:
        if '# ColHeaders' in temp_line:
            headers = temp_line.split()[2:]
            header_found = True
        elif header_found:
            temp_dict = {}
            line_values = temp_line.split()
            if len(line_values) > 2:
                if len(line_values) != len(headers):
                    raise NameError('Error')
                else:
                    for i, temp_key in enumerate(headers):
                        temp_dict[temp_key] = line_values[i]
            anat_dictionary['lh_' + line_values[0]] = temp_dict.copy()


    header_found = False
    for temp_line in rh_aparc_contents:
        if '# ColHeaders' in temp_line:
            headers = temp_line.split()[2:]
            header_found = True
        elif header_found:
            temp_dict = {}
            line_values = temp_line.split()
            if len(line_values) > 2:
                if len(line_values) != len(headers):
                    raise NameError('Error')
                else:
                    for i, temp_key in enumerate(headers):
                        temp_dict[temp_key] = line_values[i]
            anat_dictionary['rh_' + line_values[0]] = temp_dict.copy()

    extra_elements = {}
    for temp_line in aseg_contents:
        if '# subjectname ' in temp_line:
            extra_elements['subjectname'] = temp_line.split()[2]

        if '# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume,' in temp_line:
            extra_elements['BrainSegVol_mm3'] = temp_line.split()[7]

        if '# Measure BrainSegNotVent, BrainSegVolNotVent, Brain Segmentation Volume Without Ventricles' in temp_line:
            extra_elements['BrainSegVolNotVent_mm3'] = temp_line.split()[9]

        if '# Measure lhCortex, lhCortexVol, Left hemisphere cortical gray matter volume,' in temp_line:
            extra_elements['lhCortexVol_mm3'] = temp_line.split()[10]

        if '# Measure rhCortex, rhCortexVol, Right hemisphere cortical gray matter volume,' in temp_line:
            extra_elements['rhCortexVol_mm3'] = temp_line.split()[10]

        if '# Measure Cortex, CortexVol, Total cortical gray matter volume' in temp_line:
            extra_elements['CortexVol_mm3'] = temp_line.split()[9]

        if '# Measure TotalGray, TotalGrayVol, Total gray matter volume,' in temp_line:
            extra_elements['TotalGrayVol_mm3'] = temp_line.split()[8]

        if '# Measure MaskVol-to-eTIV, MaskVol-to-eTIV, Ratio of MaskVol to eTIV,' in temp_line:
            extra_elements['MaskVol-to-eTIV-Ratio'] = temp_line.split()[9]

        if '# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume,' in temp_line:
            extra_elements['eTIV_mm3'] = temp_line.split()[8][:-1]

        if '# VoxelVolume_mm3 ' in temp_line:
            extra_elements['VoxelVolume_mm3'] = temp_line.split()[2]

        if '# Measure lhSurfaceHoles, lhSurfaceHoles,' in temp_line:
            extra_elements['lhSurfaceHoles'] = temp_line.split()[14][:-1]

        if '# Measure rhSurfaceHoles, rhSurfaceHoles,' in temp_line:
            extra_elements['rhSurfaceHoles'] = temp_line.split()[14][:-1]


            # Measure lhSurfaceHoles, lhSurfaceHoles, Number of defect holes in lh surfaces prior to fixing, 126, unitless
# Measure rhSurfaceHoles, rhSurfaceHoles, Number of defect holes in rh surfaces prior to fixing, 120, unitless

    anat_dictionary['extra_elements'] = extra_elements.copy()

    if flatten:

        anat_dictionary = flatten_dictionary(anat_dictionary.copy())

    return anat_dictionary



def flatten_dictionary(dictionary):


    def inner_function(sub_dict, name_beginning):

        inner_dict = {}

        for temp_key in sub_dict.keys():

            if name_beginning != '':
                new_name_beginning = name_beginning + '_' + temp_key
            else:
                new_name_beginning = temp_key

            if type(sub_dict[temp_key]) == dict:

                new_dictionary = inner_function(sub_dict[temp_key], new_name_beginning)
                for temp_inner_key in new_dictionary.keys():
                    inner_dict[temp_inner_key] = new_dictionary[temp_inner_key]

            else:
                inner_dict[new_name_beginning] = sub_dict[temp_key]

        return inner_dict

    flattened_dictionary = inner_function(dictionary, '')
    return flattened_dictionary

def anat_dictionaries_to_csv(list_of_anat_dictionaries, output_file_path, anonymize = False):

    import csv
    toCSV = list_of_anat_dictionaries
    keys = toCSV[0].keys()
    if anonymize == True:
        keys.remove('extra_elements_subjectname')
        
    with open(output_file_path, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)
