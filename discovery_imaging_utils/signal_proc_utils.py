from scipy.fft import fft
import numpy as np



def estimate_broadband_noise(signal, TR, low_freq_cutoff = 0.2):
    '''Function that estimates broadband noise of 1/f like BOLD data

    This function calculates the fourier transform of the input signals,
    and then estimates the average broadband power as the median intensity
    of spectral components above low_freq_cutoff. This is a simple scheme based
    off of the observation that (1) the classic 1/f component of the BOLD signal
    is mainly eliminated after ~.2 Hz, and (2) median is used instead of mean as
    a cheap solution to overcome the fact that there may be some noise components
    that crop up because of breathing, or cardiac effects - it would be better
    to handle this with a more principled approach though..

    Parameters
    ----------

    signal : numpy.ndarray
        numpy array with shape <n_regions, n_timepoints>

    TR : float
        repitition time in seconds

    low_freq_cutoff : float, default = 0.2
        only use frequency content above
        this threshold (specified in Hz)

    Returns
    -------

    broadband_pow : float
        the median power in broadband range
        (or amplitude? need to double check...)

    '''

    regions = signal.shape[0]
    N = signal.shape[1]
    broadband_pow = np.zeros(regions)

    T = TR
    x = np.linspace(0.0, N*T, N)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    broadband_inds = np.where(xf >= low_freq_cutoff)[0]

    for i in range(regions):

        yf = np.abs(fft(signal[i,:]))
        broadband_pow[i] = np.median(yf[broadband_inds])

    return broadband_pow

def calc_tsnr(data_array, defined_timepoints = None):
    '''Calculates temporal signal to noise ratio

    Function to calculate the temporal signal to noise
    ratio (mean/std) for a region or series of regions.
    Unless you have a good reason for doing otherwise,
    this should probably be done prior to any denoising.

    Parameters
    ----------

    data_array : numpy.ndarray
        data array shape <num_regions, num_timepoints>
    defined_timepoints : list, or None
        if None, then all timepoints will be used, if
        defined timepoints is specified, only the
        specified timepoints will be used

    Returns
    -------

    tsnr : numpy.ndarray
        numpy array with tsnr for each region

    '''

    if type(defined_timepoints) != type(None):
        means = np.mean(data_array[:,defined_timepoints],axis=1)
        stds = np.std(data_array[:,defined_timepoints], axis=1)
    else:
        means = np.mean(data_array,axis=1)
        stds = np.std(data_array, axis=1)

    return means/stds
