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
from scipy import stats





def make_outline_overlay_underlay_plot(path_to_underlay, path_to_overlay, ap_buffer_size = 3, crop_buffer=20, num_total_images=16, dpi=400, aparcaseg=False, dseg=False, wm=False, wmgm=False, underlay_cmap='Greys', linewidths=.1, output_path=None, close_plot=True):

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
    wmgm : boolean
        If true, this will override the wm field
        and instead make a contour out of the aparcaseg
        segmentation for both GM and WM (for at least
        aparcaseg at this point)
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
        if wmgm == True:
            overlay_data[np.abs(orig_overlay - 1500) < 600] = 1
            overlay_data[orig_overlay == 2] = 1
            overlay_data[orig_overlay == 41] = 1
    elif dseg == True:
        orig_overlay = overlay_data
        if wm == False:
            overlay_data = np.zeros(overlay_data.shape)
            overlay_data[orig_overlay == 1] = 1
        elif wmgm == False:
            overlay_data = np.zeros(overlay_data.shape)
            overlay_data[orig_overlay == 2] = 1
        else:
            overlay_data = np.zeros(overlay_data.shape)
            overlay_data[orig_overlay == 1] = 1
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
    #if max_lim > 254:
    #    max_lim = 254
    if max_lim >= overlay_data.shape[2]:
        max_lim = overlay_data.shape[2] - 1
    inds_to_capture = np.linspace(min_lim,max_lim,num_total_images,dtype=int)

    overlay_max_0 = np.max(overlay_data,axis=(1,2))
    overlay_max_1 = np.max(overlay_data,axis=(0,2))
    overlay_locations_0 = np.where(overlay_max_0 > 0.5)[0]
    overlay_locations_1 = np.where(overlay_max_1 > 0.5)[0]
    min0 = np.min(overlay_locations_0) - crop_buffer
    if min0 < 0:
        min0 = 0
    max0 = np.max(overlay_locations_0) + crop_buffer
    if max0 >= overlay_data.shape[0]:
        max0 = overlay_data.shape[0] - 1
    #if max0 > 254:
    #    max0 = 254
    min1 = np.min(overlay_locations_1) - crop_buffer
    if min1 < 0:
        min1 = 0
    max1 = np.max(overlay_locations_1) + crop_buffer
    if max1 >= overlay_data.shape[1]:
        max1 = overlay_data.shape[1] - 1
    #if max1 > 254:
    #    max1 = 254


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

    if close_plot == True:
        plt.close()


    return


def make_gmwmcsf_underlay_plot(path_to_underlay, path_to_gm_mask, path_to_wm_mask, path_to_csf_mask, ap_buffer_size = 3, crop_buffer=20, num_total_images=16, underlay_cmap='gray', alpha=0.15, output_path=None, close_plot=True):

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

    if close_plot == True:
        plt.close()

    return


def make_harv_oxf_qc_image(underlay_path, harv_oxf_path, ap_buffer_size = 3, crop_buffer=30, num_total_images=16, alpha=0.4, output_path=None, close_plot=True):

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


    color_lut_dictionary = {}
    color_lut_dictionary[2+1] = ['Left Lateral Ventricle', [float(120)/256,float(18)/256,float(34)/256,1]]
    color_lut_dictionary[3+1] = ['Left Thalamus', [float(0)/256,float(118)/256,float(14)/256,1]]
    color_lut_dictionary[4+1] = ['Left Caudate', [float(122)/256,float(186)/256,float(220)/256,1]]
    color_lut_dictionary[5+1] = ['Left Putamen', [float(236)/256,float(13)/256,float(176)/256,1]]
    color_lut_dictionary[6+1] = ['Left Pallidum', [float(12)/256,float(48)/256,float(255)/256,1]]
    color_lut_dictionary[7+1] = ['Brain-Stem', [float(119)/256,float(159)/256,float(176)/256,1]]
    color_lut_dictionary[8+1] = ['Left Hippocampus', [float(220)/256,float(216)/256,float(20)/256,1]]
    color_lut_dictionary[9+1] = ['Left Amygdala', [float(103)/256,float(255)/256,float(255)/256,1]]
    color_lut_dictionary[10+1] = ['Left Accumbens', [float(255)/256,float(165)/256,float(0)/256,1]]
    color_lut_dictionary[13+1] = ['Right Lateral Ventricle', [float(120)/256,float(18)/256,float(34)/256,1]]
    color_lut_dictionary[14+1] = ['Right Thalamus', [float(0)/256,float(118)/256,float(14)/256,1]]
    color_lut_dictionary[15+1] = ['Right Caudate', [float(122)/256,float(186)/256,float(220)/256,1]]
    color_lut_dictionary[16+1] = ['Right Putamen', [float(236)/256,float(13)/256,float(176)/256,1]]
    color_lut_dictionary[17+1] = ['Right Pallidum', [float(12)/256,float(48)/256,float(255)/256,1]]
    color_lut_dictionary[18+1] = ['Right Hippocampus', [float(220)/256,float(216)/256,float(20)/256,1]]
    color_lut_dictionary[19+1] = ['Right Amygdala', [float(103)/256,float(255)/256,float(255)/256,1]]
    color_lut_dictionary[20+1] = ['Right Accumbens', [float(255)/256,float(165)/256,float(0)/256,1]]

    lut_dict = color_lut_dictionary

    t1_obj = nib.load(underlay_path)
    t1_data = t1_obj.get_fdata()


    #Load overlay image, and remove gm/wm/csf
    overlay_obj = nib.load(harv_oxf_path)
    overlay_data = overlay_obj.get_fdata()
    overlay_data[overlay_data == 1] = 0
    overlay_data[overlay_data == 2] = 0
    overlay_data[overlay_data == 12] = 0
    overlay_data[overlay_data == 13] = 0



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
                if overlay_panel[i,j] in lut_dict.keys():
                    overlay_panel_4d[i,j,:] = lut_dict[overlay_panel[i,j]][1]



    plt.figure(dpi=400)
    plt.imshow(t1_panel,cmap='gray')
    plt.xticks([])
    plt.yticks([])
    im = plt.imshow(overlay_panel_4d,alpha=alpha)
    #im = plt.contour(overlay_panel_4d)

    values = np.asarray(list(lut_dict.keys()))
    colors = [ im.cmap(im.norm(value)) for value in values]
    colors = np.zeros((values.shape[0],4))
    labels = []
    for i in range(values.shape[0]):
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

    if close_plot == True:
        plt.close()

    return


def plt_ind_on_dist(distribution, ind_val, xlabel='', dpi = 150, out_path = None, close_plot=True):
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

    if type(out_path) != type(None):
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')

    if close_plot == True:
        plt.close()

    return


def make_kde_plot_with_line(distribution_vals, individual_val, out_path = None, dpi = 150, title = '', xlabel = '',
                            range_buffer = .2, num_density_points = 100, close_plot = True):
    """
    Function to make plot showing vertical line on top of density estimate.


    This function is for visualizing normative values for a single subject
    and supports both single-session and longitudinal data, unceartainty,
    three different color map schemes, and solutions for cases where outliers
    are present.

    distribution_vals : numpy.ndarry
        1d numpy array to build distribution for visualization
    individual_val : float
        value for vertical line
    out_path : str, default None
        If string is provided, the plot will be saved at path
    dpi : float, default 150
        quality of image
    title : str, default ''
        title for main plot
    xlabel : str, default ''
        xlabel description for main plot
    range_buffer : float, default 0.2
        >0 -> proportion to add to min/max on plot xlim
    num_density_points: int, default 100
        num points to use to make the density distribution



    """


    #Make figure
    plt.figure(dpi = dpi, figsize = (5,2))

    #Generate locations for xticks
    min_range = np.min([np.min(distribution_vals), individual_val])
    min_range = min_range - np.abs(min_range)*range_buffer

    max_range = np.max([np.max(distribution_vals), individual_val])
    max_range = max_range + np.abs(min_range)*range_buffer

    #Generate values for normal distribution and plot them
    x_vals = np.arange(np.min(distribution_vals), np.max(distribution_vals), (max_range - min_range)/num_density_points)
    dist = stats.gaussian_kde(distribution_vals)
    y_vals = dist(x_vals)

    plt.plot(x_vals, y_vals, linestyle = '-', color = 'black', linewidth = 1.5)


    #top value in plot
    plot_ymax = np.max(y_vals)*1.25

    #fill distribution
    plt.fill_between(x_vals, y_vals, color = 'grey')

    #Put in the ticks and set limits
    plt.ylim(0, plot_ymax)
    plt.xlim(min_range, max_range)
    plt.yticks([])


    # get rid of the frame
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    #Now put in an arrow to show the score
    plt.axvline(individual_val, color = 'black', linestyle = '--')

    plt.title(title)
    plt.xlabel(xlabel)

    if type(out_path) != type(None):
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')

    if close_plot == True:
        plt.close()

    return
