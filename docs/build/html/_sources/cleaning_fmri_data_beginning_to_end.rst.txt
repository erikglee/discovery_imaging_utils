Cleaning fMRI Data from Beginning to End
========================================

This package includes resources for cleaning resting-state fMRI data that has undergone initial preprocessing using fMRIPREP. The total processing procedure includes three steps: (1) running preprocessing with fMRIPREP, (2) parcellating the data and gathering metadata with the *parc_ts_dictionary* module, and (3) denoising the data using the *func_denoising* module. Instructions for these three steps can be seen below.

.. toctree::
   :maxdepth: 1

   running_through_fmriprep
   make_parcellated_dict
   cleaning_parcellated_dictionary
