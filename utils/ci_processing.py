import numpy as np
import scipy
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
    dff = (Fcor - F0) / F0
    
    return F0, dff