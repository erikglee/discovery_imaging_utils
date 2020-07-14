import numpy as np
import nibabel as nib

def arr2gifti(array, output_path, hemi = ''):

    """Function to convert numpy array to gifti

    Parameters
    ----------
    array : numpy.ndarray
        1d or 2d array with features <n_vertices, n_dims> to be put in
        gifti image

    output_path : str
        path to gifti file to be created (choose your own extension)

    hemi : str
        TO BE IMPLEMENTED - 'LH' or 'RH' will specify which surface gifti
        file will match


    """

    darrays = []

    if array.ndim == 1:
        temp_darray = nib.gifti.gifti.GiftiDataArray(data=array.astype(np.float32))
        darrays.append(temp_darray)
    else:
        for i in range(array.shape[1]):
            temp_darray = nib.gifti.gifti.GiftiDataArray(data=array.astype(np.float32))
            darrays.append(temp_darray)
            print(i)

    #How to upgrade to have anat built in:
    #saved_from_wb._meta.data[0].__dict__
    #{'name': 'AnatomicalStructurePrimary', 'value': 'CortexLeft'}


    img = nib.gifti.gifti.GiftiImage(darrays=darrays)
    nib.save(img, output_path)

    return
