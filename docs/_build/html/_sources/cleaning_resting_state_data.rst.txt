Cleaning Resting State Data
===========================

fMRIPREP's workflow provides a variety of useful outputs that are helpful for cleaning fMRI data, including a variety of nuisance regressors, and fMRI data resampled to several possible output spaces. These tools assume you have ran fMRIPREP, and have data in some output space (either gifti or nifti) that you have an accompanying parcellation file for. Generally, *fsaverage* is used for the output space (which at this time needs to be pre-specified when running fMRIPREP), and the *--use-aroma* flag is desired to give more flexibility in denoising.

Although fMRIPREP can be ran using a variety of configurations, we typically run it in singularity as follows:: 

   singularity run --cleanenv --B insert_desired_fmriprep_output_dir:out_dir \
   --B insert_path_to_freesurfer_license:fs_license \
   --B insert_path_to_BIDS_directory:bids_dir \
   --B insert_path_to_desired_work_dir:work_dir
   insert_path_to_fmriprep_container_image bids_dir \
   out_dir participant --participant-label insert_name_of_participant_to_run \
   --fs-license-file fs_license \
   --mem_mb insert_amout_of_memory_to_use_in_mb \
   --use-aroma --output-spaces fsaverage fsnative T1w MNI152NLin6Asym:res-2 \
   --dummy-scans 4 --n_cpus insert_number_of_cpus_to_use \
   -w work_dir
   
The above options will run fMRIPREP with AROMA, output fMRI images in fsaverage, fsnative, MNI152, and the standard T1 space. Recommended memory is over 30000 mb when using AROMA (although older single-band datasets may be fine with much less), and at least 2 CPUs. More options are specified in fMRIPREP's documentation, and the specified options aren't necessary to use these tools.

In general the output from fMRIPREP will have undergone motion correction, distortion correction, slice-time correction, and spatial normalization. However, the output data still needs to be removed of spurious influences from motion and other non-neural sources using the variety of confounds fMRIPREP provides.
