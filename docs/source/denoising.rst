
Denoising fMRI Data
=====================================

The primary means for denoising fMRI data is to take fMRIPREP output formatted as an :doc:`image data dictionary <../image_data_dictionary>`, which contains the functional data, plus metadata that will be used in denoising such as confounds and TR. Denoising can also be conducted based on numpy arrays that are loaded into memory using the run_denoising function under denoise.general. The denoising discussed here is a wrapper around this function that gives additional functionality for the case when you have an image data dictionary structured with fMRIPREP output.

The denoising function follows a series of steps (some mandatory, others optional):

   1. DVARS is calculated on the input data
   2. If hpf_before_regression is a float, then any nuisance regressors and the input timeseries will be put through a high-pass filter with the cutoff frequency as the specified value. This is skipped if set to False.
   3. Timepoints to be excluded from analysis are identified (see Formatting Scrubbing Dictionary)
   4. Nuisance matrix containing (optionally) both 'clean' signal content and 'noise' signal content, along with offset and linear trend.
      a. If no 'noise' components are provided, nuisance regression is skipped
      b. If scrubbing is conducted, none of the scrubbed timepoints will be included while fitting nuisance regressors.
      c. If only noise components are provided, the full nuisance matrix will be fit to the input timeseries and any variance explained by this fit will be subtracted from the input timeseries
      d. If both noise components and clean components are provided, then the full matrix will be fit to the input timeseries, but only variance explained by the nuisance components will be subtracted from the input timeseries.
   5. Any missing timepoints from scrubbing will be interpolated over using the output (residuals) from the previous step. Recommended interpolation is spectral, but linear and cubic are also supported.
   6. The signal at this point can either be high-passed, low-passed, or band-passed. Otherwise set relevant parameters to False.
   7. The scrubbed timepoints that were previously interpolated over are now removed again from the signal and replaced with NaNs
   8. A new hdf5 image data dictionary is created with the cleaned timeseries and metadata relevant to the denoising process.

check out :doc:`idd <../image_data_dictionary>` 


Example
-------

TBD




Formatting Scrubbing Dictionary
-------------------------------

If scrubbing dictionary is set to False, then the initial timepoints to remove at the beginning of the scan (specified under fmriprep_out_dict) will be the only ones specified for removal.

Alternatively, scrubbing can be conducted by either (1) setting a combination of hard thresholds using any of the fMRIPREP confounds elements, or (2) a fixed proportion of timepoints can be removed from each scan based on a weighted combination of any fMRIPREP confounds elements (in this case each element will be z-scored and then the z-scored confound scores will be added). In either case, the timepoint before/after timepoints will be scrubbed. If several subjects are ran with the Uniform, this padding may lead to not small differences in the number of volumes included for each subject.

.. code-block:: python

   #no scrubbing
   scrubbing_dict = False

   #scrub timepoints where fd > 0.5mm or std_dvars > 1.5
   scrubbing_dict = {'framewise_displacement' : 0.5, 'std_dvars' : 1.5}

   #scrub 20% of timepoints (keep 80%) based on combination of
   #fd and std_dvars.
   scrubbing_dict = {'Uniform' : [0.8, ['framewise_displacement', 'std_dvars']]}


Formatting Denoising Dictionary
-------------------------------

The noise_comps_dict and clean_comps_dict that are used for denoising are formatted in the same way. When set to False, they are ignored. Otherwise any of the elements found in the fmriprep_metadata group of the image data dictionary can be used. The confound, or confounds of interest are represented as keys in the dictionary, and the value specifies if PCA should be done to reduce the data (set to False if no PC is desired, otherwise set to the number of components - in cases where confound isn't grouped, then value should be set to False).

.. code-block:: python

   #no denoising
   clean_comps_dict = False
   noise_comps_dict = False

   #regress wm, csf, gsr, and 24 motion_params
   clean_comps_dict = False
   noise_comps_dict = {'wmcsfgsr' : False, 'motion_params_24' : False}

   #regress wm, csf, and gsr, with 24 motion params but reduce
   #the 24 motion params to 6 components
   clean_comps_dict = False
   noise_comps_dict = {'wmcsfgsr' : False, 'motion_params_24' : False}   

   #regress aroma nuisance ICs, but only variance that isn't explained by
   #aroma clean ICs
   clean_comps_dict = {'aroma_clean_ics' : False}
   noise_comps_dict = {'aroma_noise_ics' : False}




`hello <Example>`_








