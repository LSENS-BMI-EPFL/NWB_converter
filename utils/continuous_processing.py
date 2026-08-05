import os
import cv2
import itertools
import numpy as np
import matplotlib
#matplotlib.use("TkAgg")  # or "Qt5Agg" if you have PyQt5/PySide installed — pops out a window with zoom/pan toolbar
# mpl.use('TkAgg') # Commented out because this causes problem on the server
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.signal as sci_si
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from ScanImageTiffReader import ScanImageTiffReader

def get_continuous_time_periods(binary_array):
    """
    Take a binary array and return a list of tuples representing the first and last position(included) of continuous
    positive period.
    This code was copied from another project or from a forum, but the reference was lost.
    :param binary_array:
    :return:
    """
    binary_array = np.copy(binary_array).astype(int)
    # first we make sure it's binary
    if np.max(binary_array) > 1:
        binary_array[binary_array > 1] = 1
    if np.min(binary_array) < 0:
        binary_array[binary_array < 0] = 0
    n_times = len(binary_array)
    d_times = np.diff(binary_array)
    # show the +1 and -1 edges
    pos = np.where(d_times == 1)[0] + 1
    neg = np.where(d_times == -1)[0] + 1

    if (pos.size == 0) and (neg.size == 0):
        if len(np.nonzero(binary_array)[0]) > 0:
            return [(0, n_times-1)]
        else:
            return []
    elif pos.size == 0:
        # i.e., starts on an spike, then stops
        return [(0, neg[0])]
    elif neg.size == 0:
        # starts, then ends on a spike.
        return [(pos[0], n_times-1)]
    else:
        if pos[0] > neg[0]:
            # we start with a spike
            pos = np.insert(pos, 0, 0)
        if neg[-1] < pos[-1]:
            #  we end with aspike
            neg = np.append(neg, n_times - 1)
        # NOTE: by this time, length(pos)==length(neg), necessarily
        # h = np.matrix([pos, neg])
        h = np.zeros((2, len(pos)), dtype=int)
        h[0] = pos
        h[1] = neg
        if np.any(h):
            result = []
            for i in np.arange(h.shape[1]):
                if h[1, i] == n_times-1:
                    result.append((h[0, i], h[1, i]))
                else:
                    result.append((h[0, i], h[1, i]-1))
            return result
    return []


def read_binary_continuous_log(bin_file, channels_dict, ni_session_sr=5000, t_stop=None):
    """
        Read behaviour binary file containing continuous data and return a dictionary containing the data and log timestamps.
    Args:
        bin_file: Behaviour logging file.
        channels_dict: Dictionary of channels to read from the binary file, defined in behaviour GUI logging session.
        ni_session_sr: Logging session sampling rate.
        t_stop: Optional parameter to stop reading the file at a certain time point.

    Returns:

    """
    # Get logged channel information
    channel_names = list(channels_dict.keys())
    n_channels = len(channel_names)
    continuous_data_dict = {}

    # Read binary data
    try:
        continuous_file = open(bin_file, mode="rb")
    except FileNotFoundError:
        print("No continuous log file found for this session. No continuous processing available.")
        return None

    # Get number of samples to read
    if t_stop is not None:
        count_param = t_stop * ni_session_sr * n_channels
    else:
        count_param = -1

    # Read specific data points
    continuous_data = np.fromfile(continuous_file, np.dtype('float'), count=count_param)

    # Rearrange data points per channel and store as dictionary
    for key, channel_index in channels_dict.items():
        channel_data = continuous_data[np.arange(start=int(channel_index), stop=int(len(continuous_data)),
                                                 step=n_channels)]
        if key == "lick_trace":
            # Convert lick trace to absolute values, like in the behaviour control GUI
            channel_data = np.abs(channel_data)
        continuous_data_dict[key] = channel_data

    # Add timestamps to dictionary
    timestamps = np.arange(0, int(len(continuous_data) / n_channels)) / ni_session_sr
    continuous_data_dict["timestamps"] = timestamps
    print(f"start : {timestamps[0]}s, end at {np.round(timestamps[-1], 2)}s")

    return continuous_data_dict



def estimate_artifact_frequency(lick_data, ni_session_sr, baseline_start_s, baseline_stop_s,
                                 freq_search_range=(5, 1000), nperseg_s=2.0):
    """Estimate the dominant artifact frequency from a known artifact-only baseline window."""
    i0 = int(baseline_start_s * ni_session_sr)
    i1 = int(baseline_stop_s * ni_session_sr)
    baseline_seg = lick_data[i0:i1]

    nperseg = min(len(baseline_seg), int(nperseg_s * ni_session_sr))
    f, psd = sci_si.welch(baseline_seg, fs=ni_session_sr, nperseg=nperseg)

    band_mask = (f >= freq_search_range[0]) & (f <= freq_search_range[1])
    carrier_freq = f[band_mask][np.argmax(psd[band_mask])]

    peak_power = psd[band_mask].max()
    median_power = np.median(psd[band_mask])
    snr = peak_power / median_power
    if snr < 5:
        print(f"WARNING: artifact peak not clearly dominant (SNR={snr:.1f}) — check baseline window")

    return carrier_freq, f, psd


def remove_artifact_notch(lick_data, ni_session_sr, carrier_freq, Q=15, n_harmonics=1):
    """Apply notch filter(s) at carrier_freq and optionally its harmonics."""
    x = lick_data.copy()
    filters = []  # store (b, a) per notch for later frequency-response plotting
    for k in range(1, n_harmonics + 1):
        f_k = carrier_freq * k
        if f_k < ni_session_sr / 2:
            b, a = sci_si.iirnotch(w0=f_k, Q=Q, fs=ni_session_sr)
            x = sci_si.filtfilt(b, a, x)
            filters.append((b, a, f_k))
    return x, filters

def _run_peak_detection(signal, ni_session_sr, sigma_s, lick_threshold, min_isi_s, prominence_frac):
    """Shared peak-detection routine, used identically for raw and corrected signals."""
    sigma = int(round(sigma_s * ni_session_sr))
    smooth = gaussian_filter1d(signal, sigma=sigma)
    min_distance = max(1, int(round(min_isi_s * ni_session_sr)))
    prominence = lick_threshold * prominence_frac
    peak_idx, props = sci_si.find_peaks(smooth, height=lick_threshold, distance=min_distance, prominence=prominence)
    lick_times = peak_idx / float(ni_session_sr)
    return lick_times, smooth

def match_lick_times(times_a, times_b, tol_s=0.05):
    """
    Match detections between two lick-time arrays within tol_s.
    Returns:
        matched_a, matched_b: booleans same length as times_a/times_b, True if matched to the other set
    """
    matched_a = np.zeros(len(times_a), dtype=bool)
    matched_b = np.zeros(len(times_b), dtype=bool)
    if len(times_a) == 0 or len(times_b) == 0:
        return matched_a, matched_b
    for i, ta in enumerate(times_a):
        j = np.argmin(np.abs(times_b - ta))
        if np.abs(times_b[j] - ta) <= tol_s and not matched_b[j]:
            matched_a[i] = True
            matched_b[j] = True
    return matched_a, matched_b


def detect_piezo_lick_times(continuous_data_dict, ni_session_sr=5000, lick_threshold=None,
                                 sigma_s=0.004, min_isi_s=0.1, prominence_frac=0.8,
                                 baseline_start_s=5, baseline_stop_s=20,
                                 notch_Q=15, n_harmonics=1, match_tol_s=0.02,
                                 do_plot=False, zoom_window_s=50, save_path=None, session_name=None):
    """
    Detect lick times, comparing detection before vs. after artifact removal.

    Returns:
        lick_times: np.array of detected lick times AFTER artifact correction
            (this is the recommended output to use downstream)
        comparison: dict with keys:
            'lick_times_raw': detections on uncorrected signal
            'lick_times_clean': detections on corrected signal (== lick_times)
            'matched_raw', 'matched_clean': booleans indicating which detections
                in each set have a corresponding match in the other (within match_tol_s)
            'n_raw', 'n_clean': total counts
            'n_removed': detections present in raw but not corrected (likely artifact false positives)
            'n_added': detections present in corrected but not raw (licks recovered
                after removing the artifact, e.g. previously obscured by it)
            'n_common': detections present in both (robust, real licks)
    """
    lick_data_raw = continuous_data_dict.get("lick_trace")
    if lick_data_raw is None:
        raise ValueError("continuous_data_dict has no 'lick_trace' entry.")
    lick_data_raw = np.asarray(lick_data_raw)

    # --- Artifact frequency estimation + removal ---
    carrier_freq, f_psd, psd = estimate_artifact_frequency(
        lick_data_raw, ni_session_sr, baseline_start_s, baseline_stop_s
    )
    #print(f"Estimated lick artifact frequency: {carrier_freq:.1f} Hz")
    lick_data_clean, filters = remove_artifact_notch(
        lick_data_raw, ni_session_sr, carrier_freq, Q=notch_Q, n_harmonics=n_harmonics
    )

    if lick_threshold is None:
        lick_threshold = 0.1
    lick_threshold_eff = lick_threshold * 1.05

    # --- Run detection on BOTH signals, identically ---
    lick_times_raw, lick_data_raw_smooth = _run_peak_detection(
        lick_data_raw, ni_session_sr, sigma_s, lick_threshold_eff, min_isi_s, prominence_frac
    )
    lick_times_clean, lick_data_smooth = _run_peak_detection(
        lick_data_clean, ni_session_sr, sigma_s, lick_threshold_eff, min_isi_s, prominence_frac
    )

    matched_raw, matched_clean = match_lick_times(lick_times_raw, lick_times_clean, tol_s=match_tol_s)

    comparison = {
        'lick_times_raw': lick_times_raw,
        'lick_times_clean': lick_times_clean,
        'matched_raw': matched_raw,
        'matched_clean': matched_clean,
        'n_raw': len(lick_times_raw),
        'n_clean': len(lick_times_clean),
        'n_removed': int(np.sum(~matched_raw)),   # in raw only -> likely artifact false positives
        'n_added': int(np.sum(~matched_clean)),   # in clean only -> recovered real licks
        'n_common': int(np.sum(matched_clean)),   # in both -> robust detections
    }
    #print(f"Raw: {comparison['n_raw']} | Corrected: {comparison['n_clean']} | "
    #      f"Common: {comparison['n_common']} | Removed (artifact-driven): {comparison['n_removed']} | "
    #      f"Added (recovered): {comparison['n_added']}")

    if do_plot:
        save_path = r'M:\analysis\Axel_Bisi\processing\piezo_lick_trace'
        plot_piezo_lick_detection_diagnostics(
            lick_data_raw, lick_data_raw_smooth,
            lick_data_clean, lick_data_smooth,
            comparison, lick_threshold_eff, ni_session_sr,
            carrier_freq, f_psd, psd, filters,
            baseline_start_s, baseline_stop_s,
            zoom_window_s=zoom_window_s, save_path=save_path, session_name=session_name
        )

    return lick_times_clean, comparison

def plot_piezo_lick_detection_diagnostics(lick_data_raw, lick_data_raw_smooth,
                           lick_data_clean, lick_data_smooth,
                           comparison, lick_threshold, ni_session_sr,
                           carrier_freq, f_psd, psd, filters,
                           baseline_start_s, baseline_stop_s,
                           zoom_window_s=50, example_window_s=1.0, save_path=None, session_name=None):
    """
    Single-figure diagnostic, formatted for A4 print in a thesis:
      Row 1: artifact diagnosis -- PSD, notch filter response, baseline before/after
      Row 2: zoomed lick detection (beginning/middle/end), raw vs corrected detections marked separately
      Row 3: full session BEFORE correction with raw-detected licks
      Row 4: full session AFTER correction with corrected-detected licks
      Row 5: example single-event zooms -- common / removed / added
      Row 6: summary bar chart -- raw / common / removed / added counts
    """
    from matplotlib.lines import Line2D

    fs = {
        'suptitle': 12,
        'title': 8.5,
        'label': 8,
        'tick': 7,
        'legend': 6.5,
        'annot': 7,
    }
    lw_thin = 0.3   # for lick data
    lw_env = 0.7
    lw_event = 0.2       # full session events
    lw_event_zoom = 0.5  # zoomed plot events

    lick_times_raw = comparison['lick_times_raw']
    lick_times_clean = comparison['lick_times_clean']
    matched_raw = comparison['matched_raw']
    matched_clean = comparison['matched_clean']

    n_samples = len(lick_data_raw)
    session_duration_s = n_samples / float(ni_session_sr)
    t = np.arange(n_samples) / float(ni_session_sr)

    fig = plt.figure(figsize=(12.0, 18.5), dpi=400)
    gs = fig.add_gridspec(6, 3, height_ratios=[1.1, 1.1, 0.9, 0.9, 0.9, 0.9],
                           hspace=0.4, wspace=0.35)

    def style_axis(ax):
        ax.tick_params(axis='both', labelsize=fs['tick'], length=2.5, pad=2)
        ax.xaxis.label.set_size(fs['label'])
        ax.yaxis.label.set_size(fs['label'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ============ ROW 1: ARTIFACT DIAGNOSIS ============
    ax_psd = fig.add_subplot(gs[0, 0])
    ax_psd.semilogy(f_psd, psd, c='k', lw=0.6)
    for b, a, f_k in filters:
        ax_psd.axvline(f_k, color='red', ls='--', lw=1.0, alpha=0.8)
    ax_psd.axvline(carrier_freq, color='red', ls='--', lw=1.0, alpha=0.8,
                    label=f"$f_c$ ≈ {carrier_freq:.1f} Hz")
    ax_psd.set_xlim(0, min(1000, ni_session_sr / 2))
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power spectral density")
    ax_psd.set_title(f"Baseline power spectral density ({baseline_start_s:.0f}-{baseline_stop_s:.0f}s)",
                      fontsize=fs['title'])
    ax_psd.legend(fontsize=fs['legend'], frameon=False, loc='upper right')
    style_axis(ax_psd)

    ax_filt = fig.add_subplot(gs[0, 1])
    for b, a, f_k in filters:
        w, h = sci_si.freqz(b, a, worN=4096, fs=ni_session_sr)
        ax_filt.plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-12)), lw=1.0, label=f"notch @ {f_k:.1f} Hz")
    ax_filt.set_xlim(0, min(1000, ni_session_sr / 2))
    ax_filt.set_ylim(-40, 5)
    ax_filt.axhline(0, color='gray', lw=0.4)
    ax_filt.set_xlabel("Frequency (Hz)")
    ax_filt.set_ylabel("Gain (dB)")
    ax_filt.set_title("Notch filter frequency response", fontsize=fs['title'])
    ax_filt.legend(fontsize=fs['legend'], frameon=False, loc='upper right')
    style_axis(ax_filt)

    ax_base = fig.add_subplot(gs[0, 2])
    i0 = int(2 * baseline_start_s * ni_session_sr)
    i1 = min(i0 + int(1.0 * ni_session_sr), n_samples)
    ax_base.plot(t[i0:i1], lick_data_raw[i0:i1], c='k', lw=0.6, alpha=0.7, label="raw")
    ax_base.plot(t[i0:i1], lick_data_clean[i0:i1], c='tab:blue', lw=0.6, label="corrected")
    ax_base.set_xlabel("Time (s)")
    ax_base.set_ylabel("Signal")
    ax_base.set_title("Baseline window: raw vs corrected", fontsize=fs['title'])
    ax_base.legend(fontsize=fs['legend'], frameon=False, loc='upper right')
    style_axis(ax_base)

    # ============ ROW 2: ZOOMED, RAW vs CORRECTED DETECTIONS ============
    def plot_zoom(ax, t_start, t_stop, title, show_legend=False):
        i_s = int(t_start * ni_session_sr)
        i_e = int(min(t_stop * ni_session_sr, n_samples))
        ax.axhline(y=lick_threshold, color='gray', lw=0.6, ls='--', alpha=0.7, zorder=0)
        ax.plot(t[i_s:i_e], lick_data_raw[i_s:i_e], c='gray', lw=lw_thin, alpha=0.5, label="raw signal")
        ax.plot(t[i_s:i_e], lick_data_raw_smooth[i_s:i_e], c='orange', lw=lw_env, alpha=0.8, label="raw envelope")
        ax.plot(t[i_s:i_e], lick_data_clean[i_s:i_e], c='k', lw=lw_thin, label="corrected signal")
        ax.plot(t[i_s:i_e], lick_data_smooth[i_s:i_e], c='green', lw=lw_env, label="corrected envelope")

        raw_win = (lick_times_raw >= t_start) & (lick_times_raw <= t_stop) & (~matched_raw)
        for lt in lick_times_raw[raw_win]:
            ax.axvline(x=lt, color='orange', lw=lw_event_zoom, alpha=0.9, ls=':')
        common_win = (lick_times_clean >= t_start) & (lick_times_clean <= t_stop) & matched_clean
        for lt in lick_times_clean[common_win]:
            ax.axvline(x=lt, color='purple', lw=lw_event_zoom, alpha=0.8)
        added_win = (lick_times_clean >= t_start) & (lick_times_clean <= t_stop) & (~matched_clean)
        for lt in lick_times_clean[added_win]:
            ax.axvline(x=lt, color='tab:blue', lw=lw_event_zoom, alpha=0.9, ls='--')

        ax.set_xlim(t_start, t_stop)
        ax.set_title(title, fontsize=fs['title'])
        ax.set_xlabel("Time (s)")
        style_axis(ax)
        if show_legend:
            ax.legend(fontsize=fs['legend'], frameon=False, loc='upper right')

    ax_z1 = fig.add_subplot(gs[1, 0])
    t0, t1 = 0, min(zoom_window_s, session_duration_s)
    plot_zoom(ax_z1, t0, t1, f"Beginning ({t0:.0f}-{t1:.0f}s)", show_legend=True)
    ax_z1.set_ylabel("Signal")

    ax_z2 = fig.add_subplot(gs[1, 1])
    mid = session_duration_s / 2.0
    t0, t1 = max(mid - zoom_window_s / 2.0, 0), min(mid + zoom_window_s / 2.0, session_duration_s)
    plot_zoom(ax_z2, t0, t1, f"Middle ({t0:.0f}-{t1:.0f}s)")

    ax_z3 = fig.add_subplot(gs[1, 2])
    t1 = session_duration_s
    t0 = max(t1 - zoom_window_s, 0)
    plot_zoom(ax_z3, t0, t1, f"End ({t0:.0f}-{t1:.0f}s)")

    # ============ ROW 3: FULL SESSION BEFORE (raw detections, color-coded) ============
    ax_full_raw = fig.add_subplot(gs[2, :])
    ax_full_raw.axhline(y=lick_threshold, color='gray', lw=0.4, ls='--', alpha=0.7, zorder=0)
    ax_full_raw.plot(t, lick_data_raw, c='k', lw=lw_thin, label="lick_data (raw)")
    ax_full_raw.plot(t, lick_data_raw_smooth, c='gray', lw=lw_env, label="raw envelope")

    common_raw_times = lick_times_raw[matched_raw]
    removed_times = lick_times_raw[~matched_raw]
    for lt in removed_times:
        ax_full_raw.axvline(x=lt, color='orange', lw=lw_event, alpha=0.6)
    for lt in common_raw_times:
        ax_full_raw.axvline(x=lt, color='purple', lw=lw_event, alpha=0.5)

    ax_full_raw.set_xlim(0, session_duration_s)
    ax_full_raw.set_title(
        f"Full session before correction ({comparison['n_raw']} detected: "
        f"{comparison['n_common']} common, {comparison['n_removed']} removed)", fontsize=fs['title']
    )
    ax_full_raw.set_xlabel("Time (s)")
    ax_full_raw.set_ylabel("Signal")
    style_axis(ax_full_raw)
    legend_elems = [
        Line2D([0], [0], color='k', lw=1.0, label="lick_data (raw)"),
        Line2D([0], [0], color='gray', lw=1.0, label="raw envelope"),
        Line2D([0], [0], color='purple', lw=1.2, alpha=0.6, label="common"),
        Line2D([0], [0], color='orange', lw=1.2, alpha=0.8, label="removed (artifact)"),
    ]
    ax_full_raw.legend(handles=legend_elems, fontsize=fs['legend'], frameon=False, loc='upper right')

    # ============ ROW 4: FULL SESSION AFTER (corrected detections, color-coded) ============
    ax_full_clean = fig.add_subplot(gs[3, :])
    ax_full_clean.axhline(y=lick_threshold, color='gray', lw=0.4, ls='--', alpha=0.7, zorder=0)
    ax_full_clean.plot(t, lick_data_clean, c='k', lw=lw_thin, label="lick_data (corrected)")
    ax_full_clean.plot(t, lick_data_smooth, c='green', lw=lw_env, label="corrected envelope")

    common_clean_times = lick_times_clean[matched_clean]
    added_times = lick_times_clean[~matched_clean]
    for lt in common_clean_times:
        ax_full_clean.axvline(x=lt, color='purple', lw=lw_event, alpha=0.5)
    for lt in added_times:
        ax_full_clean.axvline(x=lt, color='tab:blue', lw=lw_event, alpha=0.7)

    ax_full_clean.set_xlim(0, session_duration_s)
    ax_full_clean.set_title(
        f"Full session after correction ({comparison['n_clean']} detected: "
        f"{comparison['n_common']} common, {comparison['n_added']} added)", fontsize=fs['title']
    )
    ax_full_clean.set_xlabel("Time (s)")
    ax_full_clean.set_ylabel("Signal")
    style_axis(ax_full_clean)
    legend_elems2 = [
        Line2D([0], [0], color='k', lw=1.0, label="lick_data (corrected)"),
        Line2D([0], [0], color='green', lw=1.0, label="corrected envelope"),
        Line2D([0], [0], color='purple', lw=1.2, alpha=0.6, label="common"),
        Line2D([0], [0], color='tab:blue', lw=1.2, alpha=0.8, label="added (recovered)"),
    ]
    ax_full_clean.legend(handles=legend_elems2, fontsize=fs['legend'], frameon=False, loc='upper right')

    # ============ ROW 5: EXAMPLE SINGLE EVENTS -- common / removed / added ============
    def plot_example_event(ax, event_time, marker_color, title, show_legend=False):
        if event_time is None:
            ax.text(0.5, 0.5, "no example found", ha='center', va='center',
                     fontsize=fs['legend'], transform=ax.transAxes)
            ax.set_title(title, fontsize=fs['title'])
            style_axis(ax)
            return
        t_start = max(event_time - example_window_s / 2.0, 0)
        t_stop = min(event_time + example_window_s / 2.0, session_duration_s)
        i_s = int(t_start * ni_session_sr)
        i_e = int(min(t_stop * ni_session_sr, n_samples))
        ax.axhline(y=lick_threshold, color='gray', lw=0.6, ls='--', alpha=0.7, zorder=0)
        ax.plot(t[i_s:i_e], lick_data_raw[i_s:i_e], c='gray', lw=lw_thin, alpha=0.5, label="raw signal")
        ax.plot(t[i_s:i_e], lick_data_raw_smooth[i_s:i_e], c='orange', lw=lw_env, alpha=0.8, label="raw envelope")
        ax.plot(t[i_s:i_e], lick_data_clean[i_s:i_e], c='k', lw=lw_thin, label="corrected signal")
        ax.plot(t[i_s:i_e], lick_data_smooth[i_s:i_e], c='green', lw=lw_env, label="corrected envelope")
        ax.axvline(x=event_time, color=marker_color, lw=1.2, alpha=0.9)
        ax.set_xlim(t_start, t_stop)
        ax.set_title(title, fontsize=fs['title'])
        ax.set_xlabel("Time (s)")
        style_axis(ax)
        if show_legend:
            ax.legend(fontsize=fs['legend'], frameon=False, loc='upper right')

    # pick one representative example per category (first occurrence of each)
    common_example = lick_times_clean[matched_clean][0] if comparison['n_common'] > 0 else None
    removed_example = lick_times_raw[~matched_raw][0] if comparison['n_removed'] > 0 else None
    added_example = lick_times_clean[~matched_clean][0] if comparison['n_added'] > 0 else None

    ax_ex1 = fig.add_subplot(gs[4, 0])
    plot_example_event(ax_ex1, common_example, 'purple',
                        f"Example: common (retained)" if common_example is not None else "Example: common (retained)",
                        show_legend=True)
    ax_ex1.set_ylabel("Signal")

    ax_ex2 = fig.add_subplot(gs[4, 1])
    plot_example_event(ax_ex2, removed_example, 'orange',
                        f"Example: removed (artifact)" if removed_example is not None else "Example: removed (artifact)")

    ax_ex3 = fig.add_subplot(gs[4, 2])
    plot_example_event(ax_ex3, added_example, 'tab:blue',
                        f"Example: added (recovered)" if added_example is not None else "Example: added (recovered)")

    # ============ ROW 6: SUMMARY BAR CHART ============
    ax_summary = fig.add_subplot(gs[5, 1])
    labels = ['Raw\n(total)', 'Corrected\n(total)', 'Common', 'Removed\n(artifact)', 'Added\n(recovered)']
    values = [comparison['n_raw'], comparison['n_clean'], comparison['n_common'],
              comparison['n_removed'], comparison['n_added']]
    colors = ['gray', 'green', 'purple', 'orange', 'tab:blue']
    ax_summary.bar(labels, values, color=colors, alpha=0.8)
    for i, v in enumerate(values):
        ax_summary.text(i, v, str(v), ha='center', va='bottom', fontsize=fs['annot'])
    ax_summary.set_ylabel("Count")
    ax_summary.set_title("Detection comparison summary", fontsize=fs['title'], pad=6)
    ax_summary.tick_params(axis='x', labelsize=fs['tick'], rotation=0)
    ax_summary.tick_params(axis='y', labelsize=fs['tick'])
    ax_summary.yaxis.label.set_size(fs['label'])
    ax_summary.margins(y=0.15)
    style_axis(ax_summary)

    if os.path.exists(save_path) and save_path is not None:
        figname = f"{session_name}.pdf"
        fig_path = os.path.join(save_path, figname)
        fig.savefig(fig_path, bbox_inches='tight')

    #plt.show()
    return



def plot_continuous_data_dict(continuous_data_dict, timestamps_dict, ni_session_sr=5000, t_start=None, t_stop=None,
                              black_background=False):
    """
    Plot continuous data from a dictionary containing the data and timestamps.
    Args:
        continuous_data_dict: Dictionary containing continuous data
        timestamps_dict: Dictionary containing timestamps for each channel
        ni_session_sr: Sampling rate of session
        t_start: Optional parameter to start plotting the figure at a certain time point.
        t_stop: Optional parameter to stop plotting the figure at a certain time point.
        black_background: Optional parameter to plot the figure with a black background.

    Returns:

    """
    channel_names = list(continuous_data_dict.keys())
    n_channels = len(channel_names)

    fig, axes = plt.subplots(nrows=n_channels - 1, ncols=1, figsize=(8, 8), dpi=400, sharex='all')
    axes = axes.flatten()

    timestamps = continuous_data_dict["timestamps"]

    for index, ax in enumerate(axes):
        channel_name = channel_names[index]
        if channel_name == "timestamps":
            continue
        else:
            data = continuous_data_dict.get(channel_name)
            if t_start is not None and t_stop is not None:
                if t_stop * ni_session_sr > len(data):
                    if index == 1:
                        print(f"Plot data from {t_start}s to end")
                    data_to_plot = data[np.arange(t_start * ni_session_sr, len(data))]
                    timestamps_to_plot = timestamps[np.arange(t_start * ni_session_sr, len(timestamps))]
                else:
                    if index == 1:
                        print(f"Plot data from {t_start}s to {t_stop}s")
                    data_to_plot = data[np.arange(t_start*ni_session_sr, t_stop*ni_session_sr)]
                    timestamps_to_plot = timestamps[np.arange(t_start*ni_session_sr, t_stop*ni_session_sr)]
                ax.plot(timestamps_to_plot, data_to_plot, color='darkblue')
            else:
                if index == 1:
                    print(f"Plot data")
                data_to_plot = continuous_data_dict.get(channel_name)
                timestamps_to_plot = timestamps
                timestamps_to_plot = timestamps_to_plot[np.arange(0, len(data_to_plot))]
                ax.plot(timestamps_to_plot, data_to_plot, color='darkblue')

            if timestamps_dict is not None and timestamps_dict.get(channel_name) is not None:
                on_off_times = timestamps_dict.get(channel_name)
                if channel_name in ["trial_TTL", "cam1", "cam2", "context", "widefield"]:
                    for on_off in on_off_times:
                        if t_start is not None and on_off[0] < t_start:
                            continue
                        if t_stop is not None and on_off[0] > t_stop:
                            continue
                        ax.axvline(x=on_off[0], color="green")
                        ax.axvline(x=on_off[1], color="red")

                elif channel_name == 'lick_trace':
                    for x_pos in list(on_off_times):
                        if t_start is not None and x_pos[0] < t_start:
                            continue
                        if t_stop is not None and x_pos[0] > t_stop:
                            continue
                        ax.axvline(x=x_pos[0], color="green")
                elif channel_name == 'empty':
                    continue
                else:
                    for x_pos in list(on_off_times):
                        if t_start is not None and x_pos < t_start:
                            continue
                        if t_stop is not None and x_pos > t_stop:
                            continue
                        ax.axvline(x=x_pos, color="green")

            if min(data_to_plot) < 0:
                y_bottom = 1.2 * min(data_to_plot)
            else:
                y_bottom = 0.8 * min(data_to_plot)

            ax.set_ylim(y_bottom, 1.2 * max(data_to_plot))
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='x', labelsize=8)
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.set_ylabel(channel_name)
            if black_background:
                fig.set_facecolor('black')
                ax.set_facecolor('black')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_color('black')
                ax.spines['right'].set_color('black')
                ax.spines['left'].set_color('white')
                ax.tick_params(axis='y', labelsize=8, colors='white')
                ax.tick_params(axis='x', labelsize=8, colors='white')
    plt.show()


def filter_cameras_live_timestamps(on_off_timestamps):
    exposure_time = [on_off_timestamps[i][1] - on_off_timestamps[i][0] for i in range(len(on_off_timestamps))]
    long_exposure_idx = np.where(exposure_time > 2 * np.median(exposure_time))[0]
    if len(long_exposure_idx) > 0:
        filtered_on_off_timestamps = on_off_timestamps[long_exposure_idx[0] + 1: long_exposure_idx[1]]
    else:
        filtered_on_off_timestamps = on_off_timestamps

    return filtered_on_off_timestamps


def filter_wf_camera_live_timestamps(on_off_timestamps):
    inter_frame_interval = np.diff(on_off_timestamps)
    long_pause_idx = np.where(inter_frame_interval > 2 * np.median(inter_frame_interval))[0]
    if len(long_pause_idx) > 0:
        filtered_on_off_timestamps = on_off_timestamps[:long_pause_idx[0]]
    else:
        filtered_on_off_timestamps = on_off_timestamps

    return filtered_on_off_timestamps


def detect_ci_pause(ci_frame_times):
    iti_distribution = np.diff(ci_frame_times)
    pause_thr = np.median(iti_distribution) + 10 * np.std(iti_distribution)
    pause_index = np.where((iti_distribution > pause_thr) & (iti_distribution > 1))[0]
    n_pauses = len(pause_index)
    if n_pauses > 0:
        has_pause = True
        return has_pause, n_pauses, pause_index
    else:
        has_pause = False
        return has_pause, None, None


def extract_timestamps(continuous_data_dict, threshold_dict, ni_session_sr, scanimage_dict=None, filter_cameras=False, wf_file=False):
    """
    Extract timestamps from continuous logging data.
    Args:
        continuous_data_dict:  Dictionary with continuous data
        threshold_dict: Dictionary with threshold values for each channel, in Volt
        ni_session_sr: Sampling rate of session
        scanimage_dict: Dictionary with ScanImage information
        filter_cameras: Boolean, whether to filter camera timestamps

    Returns:

    """
    binary_data_dict = {}
    timestamps_dict = {}
    n_frames_dict = {}
    timestamps = continuous_data_dict['timestamps']
    print('Content of continuous data:', continuous_data_dict.keys())

    for key, data in continuous_data_dict.items():

        # Do not extract timestamps for these keys
        if key in ["timestamps", 'empty_1', 'empty_2', 'dummy_1', 'dummy_2']:
            continue

        if key == "lick_trace":

            if threshold_dict.get(key) is not None:

                # Detect lick times using behaviour GUI lick threshold
                lick_threshold = float(threshold_dict.get(key))
                lick_timestamps, _ = detect_piezo_lick_times(continuous_data_dict, ni_session_sr=ni_session_sr,
                                                          lick_threshold=lick_threshold, do_plot=False)

                # Format as tuples of on/off times for NWB
                lick_timestamps_on_off = list(zip(lick_timestamps, itertools.repeat(np.nan)))
                timestamps_dict[key] = lick_timestamps_on_off

            else:
                timestamps_dict[key] = None

        elif key == "galvo_position":

            # If no actual imaging data, do not extract timestamps
            if scanimage_dict is None:
                continue

            scan_image_rate = float(scanimage_dict.get("theoretical_ci_sampling_rate"))
            scan_image_zoom = str(scanimage_dict.get("zoom"))
            ci_movie_frame_gap = (1 / scan_image_rate) / 3

            galvo_dict_thr = threshold_dict.get(key)
            threshold = float(galvo_dict_thr.get(scan_image_zoom))
            frame_points = sci_si.find_peaks(data, height=threshold,
                                             distance=int(ci_movie_frame_gap * ni_session_sr))[0]
            if len(frame_points) == 0:
                print(f"No detected CI frames from galvo position")
                timestamps_dict[key] = []
                n_frames_dict[key] = 0
                continue
            ci_frame_times = frame_points / ni_session_sr
            ci_has_pause, n_pauses, pause_frame_index = detect_ci_pause(ci_frame_times)
            if ci_has_pause:
                # TODO : deal with pause in CI recordings, correct for ci timestamps (to be checked with frames count)
                print(f"{n_pauses} pauses detected in CI recording")
                print(f"CI pauses times (s): {ci_frame_times[pause_frame_index]}")
                # Remove the last 2 detected frames at each pause
                false_ci_fame_times = []
                for pause_index in pause_frame_index:
                    false_ci_fame_times.extend(np.arange(pause_index - 1, pause_index + 1))
                ci_timestamps_to_keep = [True if i not in false_ci_fame_times else False
                                         for i in range(len(ci_frame_times))]
                filtered_ci_frame_times = ci_frame_times[ci_timestamps_to_keep]
                # Remove the 2 last detected frames
                # Todo : always true so far but check every time
                end_filtered_ci_frame_times = filtered_ci_frame_times[0: -2]
                # Save this
                timestamps_dict[key] = end_filtered_ci_frame_times
                n_frames_dict[key] = len(end_filtered_ci_frame_times)
            else:
                # Remove the 2 last detected frames
                # Todo : always true so far but check every time this could be either -1 or -2
                filtered_ci_frame_times = ci_frame_times[0: -1]
                # Save this
                timestamps_dict[key] = filtered_ci_frame_times
                n_frames_dict[key] = len(filtered_ci_frame_times)

        else:
            threshold = int(threshold_dict.get(key))
            binary_data = np.zeros(len(data))
            binary_data[np.where(data > threshold)[0]] = 1
            binary_data_dict[key] = binary_data
            on_off_times = get_continuous_time_periods(binary_data)

            on_off_timestamps = [(timestamps[on_off_times[i][0]], timestamps[on_off_times[i][1]])
                                 for i in range(len(on_off_times))]

            if key in ["cam1", "cam2"] and len(on_off_timestamps) > 1:
                if filter_cameras:
                    print(f"Filtering camera signal")
                    filtered_on_off_timestamps = filter_cameras_live_timestamps(on_off_timestamps)
                    on_off_timestamps = filtered_on_off_timestamps

            if key in ["cam1", "cam2"] and len(on_off_timestamps) > 1:
                print('Checking cam TTL content:')
                exposure_time = [on_off_timestamps[i][1] - on_off_timestamps[i][0] for i in
                                 range(len(on_off_timestamps))]
                median_exposure = np.median(exposure_time)
                last_exposure = exposure_time[-1]
                print(f"{key} Median exposure time : {np.round(median_exposure, 4) * 1000} ms")
                if last_exposure < 0.9 * median_exposure or last_exposure > 1.1 * median_exposure:
                    print(f"{key} Last exposure: {np.round(last_exposure, 4) * 1000} ms")
                    print(f"Session likely stopped during last exposure of {key} (before image saving), "
                          f"cut the last detected frame TTL")
                    filtered_on_off_timestamps = on_off_timestamps[0: -1]
                    on_off_timestamps = filtered_on_off_timestamps

                    # Update dict with info key
                    n_frames_dict.update({f"{key}_info": {"last_exposure_cut":True}})


            if key in ["trial_TTL"]:
                print('Checking trial TTL content:')
                # Detection of early licks (whether there is a baseline window or not)
                iti = np.array([on_off_timestamps[i+1][0] - on_off_timestamps[i][1]
                                for i in range(len(on_off_timestamps) - 1)])
                early_licks = np.where(iti < 0.4)[0]  # reset trial signal in less than 0.25 s (specific to early lick)
                print(f"{len(early_licks)} early licks detected")

                if len(early_licks) > 0:
                    early_licks = list(early_licks)
                    early_licks_true_ind = [i - early_licks.index(i) for i in early_licks]
                    on_off_to_remove = np.array([i + 1 for i in early_licks])
                    filtered_on_off_timestamps = np.delete(on_off_timestamps, on_off_to_remove, axis=0)
                    on_off_timestamps = list(filtered_on_off_timestamps)

            if key in ["trial_TTL"] and binary_data[-1] == 1:
                print(f"Session likely stopped before end of last {key}, cut the last detected trial TTL")
                filtered_on_off_timestamps = on_off_timestamps[0: -1]  # remove last timestamp that signals session end
                on_off_timestamps = filtered_on_off_timestamps

            if key in ["widefield"] and wf_file is not None:
                import imageio as iio
                if wf_file.split('\\')[-1][:-4] not in ['PB175_20240308_140045', 'PB185_20240824_121743', 'PB187_20240823_131743', 'PB197_20241128_161907']:
                    props = iio.v3.improps(wf_file, plugin='pyav', format='gray16be')
                    print(f"Images in WF file: {props.shape[0]}")

                exposure_time = [on_off_timestamps[i][1] - on_off_timestamps[i][0] for i in
                                 range(len(on_off_timestamps))]
                median_exposure = np.median(exposure_time)
                last_exposure = exposure_time[-1]
                if last_exposure > median_exposure - 2 * np.std(exposure_time):
                    print(f"Cutting 1 extra widefield frame after stop signal")
                    filtered_on_off_timestamps = on_off_timestamps[:-1]
                    on_off_timestamps = filtered_on_off_timestamps

            timestamps_dict[key] = on_off_timestamps
            n_frames_dict[key] = len(on_off_timestamps)

    return timestamps_dict, n_frames_dict


def plot_exposure_times(timestamps_dict):
    trial_length = [timestamps_dict["trial_TTL"][i][1] - timestamps_dict["trial_TTL"][i][0]
                    for i in range(len(timestamps_dict["trial_TTL"]))]
    cam1_on_off_ts = timestamps_dict.get("cam1")
    exposure_time = [cam1_on_off_ts[i][1] - cam1_on_off_ts[i][0] for i in range(len(cam1_on_off_ts))]
    onset_diff = [cam1_on_off_ts[i+1][0] - cam1_on_off_ts[i][0] for i in range(len(cam1_on_off_ts) - 1)]
    ci_frames_diff = np.diff(timestamps_dict['galvo_position'])

    fig, [ax0, ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=4, figsize=(12, 4), dpi=300)
    ax0.hist(ci_frames_diff)
    ax0.set_title('Calcium imaging intervals')
    ax1.hist(exposure_time, bins=np.arange(0, 0.05, 0.005))
    ax1.set_title('Exposure times cam1')
    ax2.hist(onset_diff, bins=np.arange(0, 0.05, 0.005))
    ax2.set_title('Behaviour filming intervals')
    ax3.hist(trial_length)
    ax3.set_title('Trial durations')
    plt.tight_layout()
    plt.show()


def read_behavior_avi_movie(movie_file):
    """
    Open behaviour movie file with OpenCV and return the number of frames and the frame rate.
    Args:
        movie_file: path to movie file

    Returns:

    """
    movie_name = os.path.split(movie_file)[1]
    print(f"AVI name : {movie_name}")
    video_capture = cv2.VideoCapture(movie_file)

    # Check if camera opened successfully
    if not video_capture.isOpened():
        print("Error opening video stream or file")
    else:
        print("Video stream is opened")

    video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = np.round(video_capture.get(cv2.CAP_PROP_FPS), 2)
    print(f"AVI video frames: {video_length}, @ {video_frame_rate} Hz")

    return video_length, video_frame_rate


def print_info_dict(my_dict):
    """ Print a dictionary in a nice way. """
    for key, data in my_dict.items():
        print(f"- {key}: {data}")