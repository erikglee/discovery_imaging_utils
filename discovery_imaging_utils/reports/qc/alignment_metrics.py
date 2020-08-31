import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import binary_dilation
import glob


def mean_pow_deviation(arr, power):
    """Calculates the ave abs dif raised to given power

    Parameters
    ----------

    arr : numpy.ndarray
        1d array
    power : float
        the power to use to alter the weightings of deviations
        from the mean

    Returns
    -------

    mean_pow_deviation : float
        the output statistic

    """

    ave = np.nanmean(arr)
    abs_dif = np.abs(arr - ave)
    mean_pow_deviation = np.nanmean(np.power(abs_dif,power))


    return mean_pow_deviation


def resample_image(original_voxel_array, transformation_matrix):
    """Code to to nearest neighbor resampling in voxel space

    Takes a 3d array, and resamples it using the affine
    transformation_matrix with nearest neighbor interpolation.
    Background values are set to 0

    Parameters
    ----------

    original_voxel_array : numpy.ndarray
        3d numpy array to be resampled
    transformation_matrix : numpy.ndarray
        <4,4> affine matrix (units voxel space)

    Returns
    -------

    resampled_voxel_array : numpy.ndarray
        the resampled array

    """

    #Make a new volume to store output and
    #grab indices
    new_img = np.zeros(original_voxel_array.shape)
    new_img_inds = np.where(new_img < 1)

    #Put the indices into a form that can be
    #transformed all in one shot
    input_inds_arr = np.transpose(np.asarray(new_img_inds))
    input_inds_arr = np.hstack((input_inds_arr, np.ones((input_inds_arr.shape[0],1))))
    input_inds_arr = np.expand_dims(input_inds_arr,2)

    #Transform the indices and find which ones
    #are still in the original image
    out = np.round(np.matmul(transformation_affine, input_inds_arr)).astype(int)
    out = np.squeeze(out[:,0:3])
    good_inds = ((out >= [0,0,0]) & (out < list(new_img.shape))).all(axis=1)

    #Convert the indices present in both images
    #to tuples of arrays (might not actually
    #be necessary....)
    standard_inds = np.squeeze(input_inds_arr[:,0:3])[good_inds].astype(int)
    standard_inds = (standard_inds[:,0], standard_inds[:,1], standard_inds[:,2])
    transform_inds = out[good_inds]
    transform_inds = (transform_inds[:,0], transform_inds[:,1], transform_inds[:,2])

    #Update the new image's values to have
    #transformed input from the original
    #image
    new_img[standard_inds] = original_voxel_array[transform_inds]

    return new_img


def construct_transformation_matrix(x_trans, y_trans, z_trans, x_rot, y_rot, z_rot):
    """Constructs affine matrix to do rotation/translation

    First applies translations, then x rotation, y rotation
    and z rotation when making the new affine



    Parameters
    ----------

    x_trans : float
        number of voxels to translate in first dimension
    y_trans : float
        number of voxels to translate in second dimension
    z_trans : float
        number of voxels to translate in third dimension
    x_rot : float
        number of degrees to rotate in first dimension
    y_rot : float
        number of degrees to rotate in second dimension
    z_rot : float
        number of degrees to rotate in third dimension

    Returns
    -------

    affine : numpy.ndarray
        <4,4> array with affine

    """

    x_rot = np.radians(x_rot)
    y_rot = np.radians(y_rot)
    z_rot = np.radians(z_rot)

    original_affine = np.eye(4)

    original_affine[0,3] += x_trans
    original_affine[1,3] += y_trans
    original_affine[2,3] += z_trans


    x_rot_affine = np.eye(4)
    x_rot_affine[0,0] = np.cos(x_rot)
    x_rot_affine[0,1] = -1*np.sin(x_rot)
    x_rot_affine[1,0] = np.sin(x_rot)
    x_rot_affine[1,1] = np.cos(x_rot)

    y_rot_affine = np.eye(4)
    y_rot_affine[0,0] = np.cos(y_rot)
    y_rot_affine[0,2] = np.sin(y_rot)
    y_rot_affine[2,0] = -1*np.sin(y_rot)
    y_rot_affine[2,2] = np.cos(y_rot)

    z_rot_affine = np.eye(4)
    z_rot_affine[1,1] = np.cos(z_rot)
    z_rot_affine[1,2] = -1*np.sin(z_rot)
    z_rot_affine[2,1] = np.sin(z_rot)
    z_rot_affine[2,2] = np.cos(z_rot)

    original_affine = original_affine.dot(x_rot_affine)
    original_affine = original_affine.dot(y_rot_affine)
    original_affine = original_affine.dot(y_rot_affine)

    return original_affine


def batch_calc_alignment_metrics(reference_data, aparcaseg_data, brainmask_data):

    """Calculate alignment QC metrics based on fMRIPREP output

    Parameters
    ----------

    reference_data : numpy.ndarray
        Loaded data from a 3d BOLD reference
    aparcaseg_data : numpy.ndarray
        Loaded aparcaseg data (in same voxel space as reference data)
    brainmask_data : numpy.ndarray
        Loaded brainmask data (in same voxel space as reference data)


    Returns
    -------

    local_dev_ratio : float
        The mean deviation (from power = 4) in brain and non-brain samples
        divided by their joint deviation. This is calculated locally at 2000
        random points in the grey matter mask, only inlcuding points where a
        certain number of brain/non-brain voxels are present
    brainmask_var_component_ratio : float
        The mean deviation (from power = 4) in brainmask (which includes CSF)
        and non-brain samples divided by their joint deviation.
        This is calculated globally.
    gm_skin_1dil_var_component_ratio : float
        The mean deviation (from power = 2) in FreeSurfer GM Mask, and a mask
        generated by the new voxels found after dilating that GM Mask and
        removing any GM/WM voxels, divided by their joint deviation.
        This is calculated globablly.

    """


    gm_mask = np.zeros(reference_data.shape)
    gm_inds = np.where(np.logical_or(np.logical_and(aparcaseg_data>=1000, aparcaseg_data<=1035),np.logical_and(aparcaseg_data>=2000, aparcaseg_data<=2035)))
    wm_inds = np.where(np.logical_or(aparcaseg_data == 2, aparcaseg_data == 41))
    gm_mask[gm_inds] = 1

    #Find brainmask/other inds and calculate metric
    brain_inds = np.where(brainmask_data > 0.5)
    other_inds = np.where(brainmask_data < 0.5)

    brain_var = mean_pow_deviation(reference_data[brain_inds], 2)
    other_var = mean_pow_deviation(reference_data[other_inds], 2)
    total_var = mean_pow_deviation(np.hstack((reference_data[brain_inds],reference_data[other_inds])), 2)
    brainmask_var_component_ratio = (brain_var + other_var)/(2*total_var)


    #Find non-fs brainmask inds and calculate metric
    gm_skin_1_dilation = binary_dilation(gm_mask, iterations=1)
    gm_skin_1_dilation[np.where(aparcaseg_data > 0)] = False #remove any brain tissues
    gm_skin_1_dilation_inds = np.where(gm_skin_1_dilation > 0)

    gm_var = mean_pow_deviation(reference_data[gm_inds], 4)
    gm_skin_1_dilation_var = mean_pow_deviation(reference_data[gm_skin_1_dilation_inds], 4)
    gm_skin_1_dilation_and_gm_var = mean_pow_deviation(np.hstack((reference_data[gm_skin_1_dilation_inds],reference_data[gm_inds])), 4)
    gm_skin_1dil_var_component_ratio = (gm_var + gm_skin_1_dilation_var)/(2*gm_skin_1_dilation_and_gm_var)


    #Calculate local statistic
    num_neighborhoods=2000
    voxel_neighborhood_radius=3
    dev_ratio = np.zeros(num_neighborhoods)
    brainmask_inds = brain_inds

    sphere_offset_coords = calculate_radius_inds(voxel_neighborhood_radius)

    imgs_shape = reference_data.shape
    brain_vol = np.zeros(imgs_shape)
    brain_vol[gm_inds] = 1
    brain_vol[wm_inds] = 1



    i = 0
    while i < num_neighborhoods:

        index = np.random.randint(0, gm_inds[0].shape[0])
        neighborhood_coords = sphere_offset_coords + np.array((gm_inds[0][index], gm_inds[1][index], gm_inds[2][index]))
        in_bounds = inds_in_bounds(neighborhood_coords, imgs_shape)
        defined_coords = neighborhood_coords[in_bounds] #this doesnt actually work
        defined_coords_tuple = (np.squeeze(defined_coords[:,0]), np.squeeze(defined_coords[:,1]), np.squeeze(defined_coords[:,2]))


        neighborhood_brain = brain_vol[defined_coords_tuple]
        neighborhood_ref = reference_data[defined_coords_tuple]

        neighb_brain_dev = mean_pow_deviation(neighborhood_ref[neighborhood_brain == 1], 4)
        neighb_other_dev = mean_pow_deviation(neighborhood_ref[neighborhood_brain == 0], 4)
        neighb_full_dev  = mean_pow_deviation(neighborhood_ref, 4)


        dev_ratio[i] = (neighb_brain_dev + neighb_other_dev)/(2*neighb_full_dev)

        if np.isnan(dev_ratio[i]) == False:

            i += 1

        local_dev_ratio = np.mean(dev_ratio)


    return local_dev_ratio, brainmask_var_component_ratio, gm_skin_1dil_var_component_ratio



def local_pearson_correlation(reference_data, aparcaseg_data, brainmask_data, num_neighborhoods=2000, voxel_neighborhood_radius=3, deviation_power = 1):

    """

    NOTE: This is also calculated seperately in batch_calc_alignment_metrics

    Parameters
    ----------

    reference_data : numpy.ndarray
        Loaded data from a 3d BOLD reference
    aparcaseg_data : numpy.ndarray
        Loaded aparcaseg data (in same voxel space as reference data)
    brainmask_data : numpy.ndarray
        Loaded brainmask data (in same voxel space as reference data)
    num_neighborhoods : int
        The number of neighborhoods to sample during calculation
    voxel_neighborhood_radius : int
        The radius of each neighborhood (unit = voxels)
    deviation_power : int
        The deviation power to be used



    """

    dev_ratio = np.zeros(num_neighborhoods)
    gm_inds = np.where(np.logical_or(np.logical_and(aparcaseg_data>=1000, aparcaseg_data<=1035),np.logical_and(aparcaseg_data>=2000, aparcaseg_data<=2035)))
    wm_inds = np.where(np.logical_or(aparcaseg_data == 2, aparcaseg_data == 41))
    brainmask_inds = np.where(brainmask_data > 0.5)

    sphere_offset_coords = calculate_radius_inds(voxel_neighborhood_radius)

    imgs_shape = reference_data.shape

    brain_vol = np.zeros(imgs_shape)
    brain_vol[gm_inds] = 1
    brain_vol[wm_inds] = 1



    i = 0
    while i < num_neighborhoods:

        index = np.random.randint(0, gm_inds[0].shape[0])
        neighborhood_coords = sphere_offset_coords + np.array((gm_inds[0][index], gm_inds[1][index], gm_inds[2][index]))
        in_bounds = inds_in_bounds(neighborhood_coords, imgs_shape)
        defined_coords = neighborhood_coords[in_bounds] #this doesnt actually work
        defined_coords_tuple = (np.squeeze(defined_coords[:,0]), np.squeeze(defined_coords[:,1]), np.squeeze(defined_coords[:,2]))


        neighborhood_brain = brain_vol[defined_coords_tuple]
        neighborhood_ref = reference_data[defined_coords_tuple]

        neighb_brain_dev = mean_pow_deviation(neighborhood_ref[neighborhood_brain == 1], deviation_power)
        neighb_other_dev = mean_pow_deviation(neighborhood_ref[neighborhood_brain == 0], deviation_power)
        neighb_full_dev  = mean_pow_deviation(neighborhood_ref, deviation_power)


        dev_ratio[i] = (neighb_brain_dev + neighb_other_dev)/(2*neighb_full_dev)

        if np.isnan(dev_ratio[i]) == False:

            i += 1


    return np.mean(dev_ratio)


def calculate_radius_inds(radius):
    """Makes radius inds for sphere

    This function takes a radius (units voxels),
    for a 3d array (not provided), and returns
    "index_modifiers" for the voxels falling
    within the radius, such that any <i,j,k>
    can be added to these index modifiers,
    which will return elements within the
    given radius of <i,j,k> (note, at image
    boundaries, some indices will be out of
    bounds)

    Parameters
    ----------

    radius : int
        radius (inclusive) to find sphere inds
        for

    Returns
    -------

    index_modifiers : numpy.ndarray
        <num_points, 3>


    Usage
    -----

    For radius 5 around point i,j,k

    index_modifiers = calculate_radius_inds(5)
    circle_inds = index_modifiers + np.array((i,j,k))

    """

    index_modifiers = []



    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            for k in range(-radius,radius+1):

                if np.sqrt(i**2 + j**2 + k**2) <= radius:

                    index_modifiers.append(np.array((i,j,k)))

    return np.asarray(index_modifiers)

def inds_in_bounds(inds, shape):
    """Says whether inds are within shape

    Function to test whether or not a series
    of indices are within the bounds of a
    matrix with a given shape

    Parameters
    ----------

    inds : numpy.ndarray
        shape <num_points, num_dimensions>
    shape : tuple
        shape of array (given by *.shape)
        to use as bounding box

    Returns
    -------

    in_bounds : numpy.ndarray
        array shape <num_points> with
        boolean to say whether or not
        index is in bounding box

    """

    in_bounds = ((inds >= np.zeros(len(shape))) & (inds < list(shape))).all(axis=1)

    return in_bounds
