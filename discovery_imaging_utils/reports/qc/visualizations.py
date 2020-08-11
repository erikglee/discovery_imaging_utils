import glob
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import decomposition
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from pandas.plotting import table
import nibabel as nib
import inspect
import matplotlib.patches as mpatches
import seaborn as sns





def make_outline_overlay_underlay_plot(path_to_underlay, path_to_overlay, ap_buffer_size = 3, crop_buffer=20, num_total_images=16, dpi=400, aparcaseg=False, dseg=False, wm=False, underlay_cmap='Greys', linewidths=.1, output_path=None):

    """Function that makes contour plot with nifti mask and underlay.


    Takes paths to two nifti files, the overlay nifti file will
    be thresholded, and a contour created out of the resulting mask
    and then will be projected over the underlay. If the underlay
    and overlay don't have the same dimensions, then nibabel's
    resample_to_from function will be used to resample the
    underlay into the mask space.

    Parameters
    ----------
    path_to_underlay : str
        path to underlay file
    path_to_overlay : str
        path to overlay that will be masked and used
        to create contour
    ap_buffer_size : int
        ap buffer
    crob_buffer : int
        make this bigger to reduce cropping
    num_total_images : int
        number of images in the panel, must
        have a sqrt that is an integer so
        panel can be square
    aparcaseg : boolean
        if true, contours of the pial surface
        will be used from an aparcaseg vol
    dseg : boolean
        if true gray matter voxels used for
        contour from dseg file
    wm : boolean
        If true, will change aparcaseg and dseg
        functionality so that they make outline
        of wm surface instead
    underlay_cmap : str
        the matplotlib colormap to use for the
        underlay
    linewidths : float
        the width of contour line
    output_path : str or None
        optional path for file to be saved
        (do not include extension)

    """



    underlay_path = path_to_underlay
    underlay_obj = nib.load(underlay_path)
    underlay_data = underlay_obj.get_fdata()

    print(underlay_data.shape)

    overlay_img = nib.load(path_to_overlay)
    overlay_data = overlay_img.get_fdata()

    orig_overlay = overlay_data
    if aparcaseg == True:
        overlay_data = np.zeros(orig_overlay.shape)
        if wm == False:
            overlay_data[np.abs(orig_overlay - 1500) < 600] = 1
        else:
            overlay_data[orig_overlay == 2] = 1
            overlay_data[orig_overlay == 41] = 1
    elif dseg == True:
        orig_overlay = overlay_data
        if wm == False:
            overlay_data = np.zeros(overlay_data.shape)
            overlay_data[orig_overlay == 1] = 1
        else:
            overlay_data = np.zeros(overlay_data.shape)
            overlay_data[orig_overlay == 2] = 1
    else:
        overlay_data[np.abs(overlay_data) > 0.0001] = 1



    #resample if the two images have different shapes
    if np.array_equal(overlay_data.shape, underlay_data.shape) == False:
        resampled_underlay = proc.resample_from_to(underlay_obj, overlay_img)
        underlay_data = resampled_underlay.get_fdata()




    overlay_ap_max = np.max(overlay_data,axis=(0,1))
    non_zero_locations = np.where(overlay_ap_max > 0.5)[0]
    min_lim = np.min(non_zero_locations) - ap_buffer_size
    if min_lim < 0:
        min_lim = 0
    max_lim = np.max(non_zero_locations) + ap_buffer_size
    if max_lim > 254:
        max_lim = 254
    inds_to_capture = np.linspace(min_lim,max_lim,num_total_images,dtype=int)

    overlay_max_0 = np.max(overlay_data,axis=(1,2))
    overlay_max_1 = np.max(overlay_data,axis=(0,2))
    overlay_locations_0 = np.where(overlay_max_0 > 0.5)[0]
    overlay_locations_1 = np.where(overlay_max_1 > 0.5)[0]
    min0 = np.min(overlay_locations_0) - crop_buffer
    if min0 < 0:
        min0 = 0
    max0 = np.max(overlay_locations_0) + crop_buffer
    if max0 > 254:
        max0 = 254
    min1 = np.min(overlay_locations_1) - crop_buffer
    if min1 < 0:
        min1 = 0
    max1 = np.max(overlay_locations_1) + crop_buffer
    if max1 > 254:
        max1 = 254


    num_imgs_per_dim = int(np.sqrt(num_total_images))
    counting_index = 0
    for i in range(num_imgs_per_dim):

        temp_underlay_row = underlay_data[min0:max0,min1:max1,inds_to_capture[counting_index]]
        temp_overlay_row = overlay_data[min0:max0,min1:max1,inds_to_capture[counting_index]]
        counting_index += 1
        for j in range(1,num_imgs_per_dim):
            temp_underlay_row = np.vstack((temp_underlay_row,underlay_data[min0:max0,min1:max1,inds_to_capture[counting_index]]))
            temp_overlay_row = np.vstack((temp_overlay_row,overlay_data[min0:max0,min1:max1,inds_to_capture[counting_index]]))
            counting_index +=1
        if i == 0:
            temp_underlay_full = temp_underlay_row
            temp_overlay_full = temp_overlay_row
        else:
            temp_underlay_full = np.hstack((temp_underlay_full,temp_underlay_row))
            temp_overlay_full = np.hstack((temp_overlay_full,temp_overlay_row))



    underlay_panel = np.fliplr(np.rot90(temp_underlay_full,3))
    overlay_panel = np.fliplr(np.rot90(temp_overlay_full,3))





    plt.figure(dpi=dpi)
    im = plt.contour(overlay_panel, linewidths=linewidths, colors='m')
    im = plt.imshow(underlay_panel, cmap=underlay_cmap)
    plt.xticks([])
    plt.yticks([])

    if type(output_path) != type(None):
        plt.savefig(output_path + '.jpeg', dpi=dpi, bbox_inches='tight')


    return


def make_gmwmcsf_underlay_plot(path_to_underlay, path_to_gm_mask, path_to_wm_mask, path_to_csf_mask, ap_buffer_size = 3, crop_buffer=20, num_total_images=16, underlay_cmap='gray', alpha=0.15, output_path=None):

    #NEW USERS: BEFORE USING THIS FUNCTION, UPDATE THE PATH "path_to_fs_color_lut" TO POINT
    #TO THE COLOR LUT FILE FOUND IN YOUR FREESURFER INSTALLATION DIRECTORY (see load_color_lut
    #dictionary function below)

    #This is a script to make qc images for different freesurfer segmentions,
    #made with the subcortical subfield segmentations in mind (but it will probably work
    #for other aseg type volumes found under the subject's fs mri folder as well). Inputs include
    #(1) subj_path which is the path to the subject who you want to make a qc image
    #for, (2) list_of_overlays which is a list of images whose contents will be
    #summed to produce the overlay file that will be projected on top of the T1 from freesurfer.
    #The reason why this is a list is because sometimes subcortical segmentations produce seperate
    #lh/rh files... if you only have one file, you still need to put it in a list.
    #This assumes the overlays are under the subject's mri folder, so if that is not the case
    #you will need to jimmy rig the script. Some example overlay option lists include:
    #
    # ['ThalamicNuclei.v10.T1.FSvoxelSpace.mgz']
    # ['brainstemSsLabels.v12.FSvoxelSpace.mgz']
    # ['rh.hippoAmygLabels-T1.v21.HBT.FSvoxelSpace.mgz','lh.hippoAmygLabels-T1.v21.HBT.FSvoxelSpace.mgz']
    #
    #(3) ap_buffer_size is the number of anterior/posterior slices (on each side) that
    #will be included in the partition to determine where the slices begin/end. If equal
    #to zero, this will be the bounds determined from the overlay. (4) crop_buffer is the number of voxels beyond
    #the bounds of the overlay to be included for the in-plane dimensions (l/r and inf/sup).
    #(5) num_total_images can be between 4 and 49 any number that has a square root that is an integer.
    #This will be the total number of images included in the panel.
    #
    #subj_path and list_of_overlays are the only required arguments. If you don't want to include
    #any sort of cropping, you can just set the related variables to 255+, and then the images will
    #be made with all voxels for each panel image
    #
    #The output of the script will be images of the overlay on top of the anatomical image in the
    #color scheme described by the freesurfer color lut file (update the path to the color lut
    #in this script if necessary). The images will be saved in a new folder under /subject/stats/
    #named qc_images.


    t1_path = path_to_underlay
    t1_obj = nib.load(t1_path)
    t1_data = t1_obj.get_fdata()

    print(t1_data.shape)

    gm_data = nib.load(path_to_gm_mask).get_fdata()
    wm_data = nib.load(path_to_wm_mask).get_fdata()
    csf_data = nib.load(path_to_csf_mask).get_fdata()

    overlay_data = gm_data + wm_data*2 + csf_data*3

    overlay_ap_max = np.max(overlay_data,axis=(0,1))
    non_zero_locations = np.where(overlay_ap_max > 0.5)[0]
    min_lim = np.min(non_zero_locations) - ap_buffer_size
    if min_lim < 0:
        min_lim = 0
    max_lim = np.max(non_zero_locations) + ap_buffer_size
    if max_lim > 254:
        max_lim = 254
    inds_to_capture = np.linspace(min_lim,max_lim,num_total_images,dtype=int)

    overlay_max_0 = np.max(overlay_data,axis=(1,2))
    overlay_max_1 = np.max(overlay_data,axis=(0,2))
    overlay_locations_0 = np.where(overlay_max_0 > 0.5)[0]
    overlay_locations_1 = np.where(overlay_max_1 > 0.5)[0]
    min0 = np.min(overlay_locations_0) - crop_buffer
    if min0 < 0:
        min0 = 0
    max0 = np.max(overlay_locations_0) + crop_buffer
    if max0 > 254:
        max0 = 254
    min1 = np.min(overlay_locations_1) - crop_buffer
    if min1 < 0:
        min1 = 0
    max1 = np.max(overlay_locations_1) + crop_buffer
    if max1 > 254:
        max1 = 254


    num_imgs_per_dim = int(np.sqrt(num_total_images))
    counting_index = 0
    for i in range(num_imgs_per_dim):

        temp_t1_row = t1_data[min0:max0,min1:max1,inds_to_capture[counting_index]]
        temp_overlay_row = overlay_data[min0:max0,min1:max1,inds_to_capture[counting_index]]
        counting_index += 1
        for j in range(1,num_imgs_per_dim):
            temp_t1_row = np.vstack((temp_t1_row,t1_data[min0:max0,min1:max1,inds_to_capture[counting_index]]))
            temp_overlay_row = np.vstack((temp_overlay_row,overlay_data[min0:max0,min1:max1,inds_to_capture[counting_index]]))
            counting_index +=1
        if i == 0:
            temp_t1_full = temp_t1_row
            temp_overlay_full = temp_overlay_row
        else:
            temp_t1_full = np.hstack((temp_t1_full,temp_t1_row))
            temp_overlay_full = np.hstack((temp_overlay_full,temp_overlay_row))



    t1_panel = np.fliplr(np.rot90(temp_t1_full,3))
    overlay_panel = np.fliplr(np.rot90(temp_overlay_full,3))



    overlay_panel = np.ma.masked_where(overlay_panel < 0.1, overlay_panel)

    plt.figure(dpi=400)
    im = plt.imshow(t1_panel, cmap=underlay_cmap)
    im = plt.imshow(overlay_panel, alpha=alpha, cmap='jet')
    plt.xticks([])
    plt.yticks([])

    if type(output_path) != type(None):

        plt.savefig(output_path + '.jpg', bbox_inches='tight')

    return


def make_harv_oxf_qc_image(underlay_path, harv_oxf_path, ap_buffer_size = 3, crop_buffer=40, num_total_images=16, alpha=0.4, output_path=None):

    #NEW USERS: BEFORE USING THIS FUNCTION, UPDATE THE PATH "path_to_fs_color_lut" TO POINT
    #TO THE COLOR LUT FILE FOUND IN YOUR FREESURFER INSTALLATION DIRECTORY (see load_color_lut
    #dictionary function below)

    #This is a script to make qc images for different freesurfer segmentions,
    #made with the subcortical subfield segmentations in mind (but it will probably work
    #for other aseg type volumes found under the subject's fs mri folder as well). Inputs include
    #(1) subj_path which is the path to the subject who you want to make a qc image
    #for, (2) list_of_overlays which is a list of images whose contents will be
    #summed to produce the overlay file that will be projected on top of the T1 from freesurfer.
    #The reason why this is a list is because sometimes subcortical segmentations produce seperate
    #lh/rh files... if you only have one file, you still need to put it in a list.
    #This assumes the overlays are under the subject's mri folder, so if that is not the case
    #you will need to jimmy rig the script. Some example overlay option lists include:
    #
    # ['ThalamicNuclei.v10.T1.FSvoxelSpace.mgz']
    # ['brainstemSsLabels.v12.FSvoxelSpace.mgz']
    # ['rh.hippoAmygLabels-T1.v21.HBT.FSvoxelSpace.mgz','lh.hippoAmygLabels-T1.v21.HBT.FSvoxelSpace.mgz']
    #
    #(3) ap_buffer_size is the number of anterior/posterior slices (on each side) that
    #will be included in the partition to determine where the slices begin/end. If equal
    #to zero, this will be the bounds determined from the overlay. (4) crop_buffer is the number of voxels beyond
    #the bounds of the overlay to be included for the in-plane dimensions (l/r and inf/sup).
    #(5) num_total_images can be between 4 and 49 any number that has a square root that is an integer.
    #This will be the total number of images included in the panel.
    #
    #subj_path and list_of_overlays are the only required arguments. If you don't want to include
    #any sort of cropping, you can just set the related variables to 255+, and then the images will
    #be made with all voxels for each panel image
    #
    #The output of the script will be images of the overlay on top of the anatomical image in the
    #color scheme described by the freesurfer color lut file (update the path to the color lut
    #in this script if necessary). The images will be saved in a new folder under /subject/stats/
    #named qc_images.



    #Script to load the freesurfer color lut file as a dictionary.
    #This file can be found in your $FREESURFER_HOME directory.
    def load_color_lut_dictionary():

        path_to_fs_color_lut = '/'.join(os.path.abspath(inspect.getfile(make_harv_oxf_qc_image)).split('/')[:-1]) + '/ColorLUT.txt'

        color_lut_file = open(path_to_fs_color_lut,'r')
        color_lut_dictionary = {}
        for line in color_lut_file:
            if line[0].isnumeric():

                split_line = line.split(' ')
                reduced_line = list(filter(lambda a: a != '', split_line))
                reduced_line[-1] = reduced_line[-1].strip('\n')
                key = int(reduced_line[0])
                color = [float(reduced_line[2])/256,float(reduced_line[3])/256,float(reduced_line[4])/256,1]
                anat = reduced_line[1]
                color_lut_dictionary[key] = [anat, color]

        return color_lut_dictionary

    lut_dict = load_color_lut_dictionary()

    t1_obj = nib.load(underlay_path)
    t1_data = t1_obj.get_fdata()

    print(t1_data.shape)


    #Load overlay image, and remove gm/wm/csf
    overlay_obj = nib.load(harv_oxf_path)
    overlay_data = overlay_obj.get_fdata()
    overlay_data[overlay_data == 2] = 0
    overlay_data[overlay_data == 3] = 0
    overlay_data[overlay_data == 4] = 0
    overlay_data[overlay_data == 41] = 0
    overlay_data[overlay_data == 42] = 0
    overlay_data[overlay_data == 43] = 0


    overlay_ap_max = np.max(overlay_data,axis=(0,1))
    non_zero_locations = np.where(overlay_ap_max > 0.5)[0]
    min_lim = np.min(non_zero_locations) - ap_buffer_size
    if min_lim < 0:
        min_lim = 0
    max_lim = np.max(non_zero_locations) + ap_buffer_size
    if max_lim > 254:
        max_lim = 254
    inds_to_capture = np.linspace(min_lim,max_lim,num_total_images,dtype=int)

    overlay_max_0 = np.max(overlay_data,axis=(1,2))
    overlay_max_1 = np.max(overlay_data,axis=(0,2))
    overlay_locations_0 = np.where(overlay_max_0 > 0.5)[0]
    overlay_locations_1 = np.where(overlay_max_1 > 0.5)[0]
    min0 = np.min(overlay_locations_0) - crop_buffer
    if min0 < 0:
        min0 = 0
    max0 = np.max(overlay_locations_0) + crop_buffer
    if max0 > 254:
        max0 = 254
    min1 = np.min(overlay_locations_1) - crop_buffer
    if min1 < 0:
        min1 = 0
    max1 = np.max(overlay_locations_1) + crop_buffer
    if max1 > 254:
        max1 = 254


    num_imgs_per_dim = int(np.sqrt(num_total_images))
    counting_index = 0
    for i in range(num_imgs_per_dim):

        temp_t1_row = t1_data[min0:max0,min1:max1,inds_to_capture[counting_index]]
        temp_overlay_row = overlay_data[min0:max0,min1:max1,inds_to_capture[counting_index]]
        counting_index += 1
        for j in range(1,num_imgs_per_dim):
            temp_t1_row = np.vstack((temp_t1_row,t1_data[min0:max0,min1:max1,inds_to_capture[counting_index]]))
            temp_overlay_row = np.vstack((temp_overlay_row,overlay_data[min0:max0,min1:max1,inds_to_capture[counting_index]]))
            counting_index +=1
        if i == 0:
            temp_t1_full = temp_t1_row
            temp_overlay_full = temp_overlay_row
        else:
            temp_t1_full = np.hstack((temp_t1_full,temp_t1_row))
            temp_overlay_full = np.hstack((temp_overlay_full,temp_overlay_row))

    t1_panel = np.fliplr(np.rot90(temp_t1_full,3))
    overlay_panel = np.fliplr(np.rot90(temp_overlay_full,3))
    unique_values = np.unique(overlay_panel)[1:]

    overlay_panel_4d = np.zeros((overlay_panel.shape[0],overlay_panel.shape[1],4))
    for i in range(overlay_panel_4d.shape[0]):
        for j in range(overlay_panel_4d.shape[1]):
            if overlay_panel[i,j] < 0.5:
                continue
            else:
                overlay_panel_4d[i,j,:] = lut_dict[overlay_panel[i,j]][1]



    plt.figure(dpi=400)
    plt.imshow(t1_panel,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    im = plt.imshow(overlay_panel_4d,alpha=alpha)
    #im = plt.contour(overlay_panel_4d)

    values = unique_values
    colors = [ im.cmap(im.norm(value)) for value in values]
    colors = np.zeros((unique_values.shape[0],4))
    labels = []
    for i in range(unique_values.shape[0]):
        colors[i,:] = lut_dict[values[i]][1]
        labels.append(lut_dict[values[i]][0])



    # create some data
    #data = np.random.randint(0, 8, (5,5))
    # get the unique values from data
    # i.e. a sorted list of all values in data
    patches = [ mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values)) ]

    if len(values) > 25:
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize ='xx-small',ncol=2)
    else:
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize ='xx-small')


    if type(output_path) != type(None):

        plt.savefig(output_path + '.jpg', bbox_inches='tight')

    return


def plt_ind_on_dist(distribution, ind_val, xlabel='', dpi = 150, out_path = None):
    """Function plots an arrow on a distribution

    Makes a smooth illustration of a distribution
    and highlights a point on that distribution
    with an arrow (i.e. to show where an individual
    lies on that distribution). The distribution
    is estimated using a kernel density estimator
    through seaborn.

    Parameters
    ----------

    distribution : numpy.ndarray
        1d numpy array whose contents will be used to
        build a probability density function in the
        plot
    ind_val : float
        location where the arrow should point to on the
        plot
    xlabel : str, optional
        string to put on the xlabel
    dpi : int, optional
        dpi of the figure
    out_path : str or None, optional
        defaults to None, otherwise the name of the file
        to be created


    """

    plt.figure(dpi=dpi,figsize=[5,2.5])
    ax = sns.distplot(distribution, hist = False, kde = True,
                     kde_kws = {'shade': True, 'linewidth': 4})



    bottom, top = plt.ylim()
    testing_props = dict(arrowstyle="<-",mutation_scale=30,shrinkA=10,shrinkB=100)
    ax.annotate("", xy=(ind_val, 0.0), xytext=(ind_val, top*.75),
                 arrowprops=dict(shrinkA=1000,shrinkB=1000,facecolor='k')) #tailwidth=1))

    plt.xlabel(xlabel,fontsize='x-large')
    plt.ylabel('Density',fontsize='x-large')

    if type(out_file) != type(None):
        plt.savefig(out_path, dpi=dpi)
