import numpy as np
import scipy
import os
import matplotlib.pyplot as plt


def compute_F0(F, fs, window):
    
    # Parameters --------------------------------------------------------------
    nfilt = 30  # Number of taps to use in FIR filter
    fw_base = 1  # Cut-off frequency for lowpass filter, in Hz
    base_pctle = 5  # Percentile to take as baseline value

    # Main --------------------------------------------------------------------
    # Ensure array_like input is a numpy.ndarray
    F = np.asarray(F)

    # For short measurements, we reduce the number of taps
    nfilt = min(nfilt, max(3, int(F.shape[1] / 3)))

    if fs <= fw_base:
        # If our sampling frequency is less than our goal with the smoothing
        # (sampling at less than 1Hz) we don't need to apply the filter.
        filtered_f = F
    else:
        # The Nyquist rate of the signal is half the sampling frequency
        nyq_rate = fs / 2.0

        # Cut-off needs to be relative to the nyquist rate. For sampling
        # frequencies in the range from our target lowpass filter, to
        # twice our target (i.e. the 1Hz to 2Hz range) we instead filter
        # at the Nyquist rate, which is the highest possible frequency to
        # filter at.
        cutoff = min(1.0, fw_base / nyq_rate)

        # Make a set of weights to use with our taps.
        # We use an FIR filter with a Hamming window.
        b = scipy.signal.firwin(nfilt, cutoff=cutoff, window='hamming')

        # The default padlen for filtfilt is 3 * nfilt, but in case our
        # dataset is small, we need to make sure padlen is not too big
        padlen = min(3 * nfilt, F.shape[1] - 1)

        # Use filtfilt to filter with the FIR filter, both forwards and
        # backwards.
        filtered_f = scipy.signal.filtfilt(b, [1.0], F, axis=1,
                                           padlen=padlen)

    # Take a percentile of the filtered signal and windowed signal
    F0 = scipy.ndimage.percentile_filter(filtered_f, percentile=base_pctle, size=(1,(fs*2*window + 1)), mode='constant', cval=+np.inf)

    # Ensure filtering doesn't take us below the minimum value which actually
    # occurs in the data. This can occur when the amount of data is very low.
    F0 = np.maximum(F0, np.nanmin(F, axis=1, keepdims=True))

    return F0


def compute_dff(F, Fneu, fs, window=60):
    
    Fcor = F - .7 * Fneu  # Neuropil correction.
    F0 = compute_F0(Fcor, fs, window)
    dff = (F - F0) / F0
    
    return F0, dff


def add_suite2p_roi(ps, stat, is_cell, dim_x, dim_y):
    pixel_masks = []
    for cell in np.arange(len(stat)):
        if is_cell[cell][0] == 0:
            continue
        pix_mask = [(y, x, 1) for x, y in zip(stat[cell]["xpix"], stat[cell]["ypix"])]
        image_mask = np.zeros((dim_y, dim_x))  # recently inverted (12th of may 2020)
        for pix in pix_mask:
            image_mask[int(pix[0]), int(pix[1])] = pix[2]  # pix[0] pix[1] recently inverted (12th of may 2020)
        # we can id to identify the cell (int) otherwise it will be incremented at each step
        pixel_masks.append(pix_mask)
        ps.add_roi(pixel_mask=pix_mask, image_mask=image_mask)


def get_processed_ci(suite2p_folder):
    suite2p_folder = os.path.join(suite2p_folder, "plane0")
    if not os.path.isfile(os.path.join(suite2p_folder, "stat.npy")):
        print(f"Stat file missing in {suite2p_folder}")
        return
    else:
        stat = np.load(os.path.join(suite2p_folder, "stat.npy"), allow_pickle=True)
        is_cell = np.load(os.path.join(suite2p_folder, "iscell.npy"), allow_pickle=True)
        F = np.load(os.path.join(suite2p_folder, "F.npy"), allow_pickle=True)
        Fneu = np.load(os.path.join(suite2p_folder, "Fneu.npy"), allow_pickle=True)
        dcnv = np.load(os.path.join(suite2p_folder, "spks.npy"), allow_pickle=True)

    return stat, is_cell, F, Fneu, dcnv


