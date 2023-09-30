import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sci_si
import cv2
from ScanImageTiffReader import ScanImageTiffReader
import time
import os


def get_continuous_time_periods(binary_array):
    """
    take a binary array and return a list of tuples representing the first and last position(included) of continuous
    positive period
    This code was copied from another project or from a forum, but i've lost the reference.
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
    channel_names = list(channels_dict.keys())
    n_channels = len(channel_names)
    continuous_data_dict = {}
    continuous_file = open(bin_file, mode="rb")
    if t_stop is not None:
        count_param = t_stop * ni_session_sr * n_channels
    else:
        count_param = -1

    continuous_data = np.fromfile(continuous_file, np.dtype('float'), count=count_param)

    for key, channel_index in channels_dict.items():
        channel_data = continuous_data[np.arange(start=int(channel_index), stop=int(len(continuous_data)),
                                                 step=n_channels)]
        if key == "lick_trace":
            channel_data = np.abs(channel_data)
        continuous_data_dict[key] = channel_data

    timestamps = np.arange(0, int(len(continuous_data) / n_channels)) / ni_session_sr
    continuous_data_dict["timestamps"] = timestamps
    print(f"start : {timestamps[0]}s, end at {np.round(timestamps[-1], 2)}s")

    return continuous_data_dict


def plot_continuous_data_dict(continuous_data_dict, timestamps_dict, ni_session_sr=5000, t_start=None, t_stop=None,
                              black_background=False):
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


def extract_timestamps(continuous_data_dict, threshold_dict, ni_session_sr, scanimage_dict=None, filter_cameras=False):
    binary_data_dict = {}
    timestamps_dict = {}
    n_frames_dict = {}
    timestamps = continuous_data_dict['timestamps']

    for key, data in continuous_data_dict.items():

        # Do not extract timestamps for these keys
        if key in ["timestamps", "lick_trace", 'empty_1', 'empty_2']:
            continue

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
                # Todo : always true so far but check every time
                filtered_ci_frame_times = ci_frame_times[0: -2]
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
                exposure_time = [on_off_timestamps[i][1] - on_off_timestamps[i][0] for i in
                                 range(len(on_off_timestamps))]
                median_exposure = np.median(exposure_time)
                last_exposure = exposure_time[-1]
                if last_exposure < median_exposure - 2 * np.std(exposure_time):
                    print(f"Session likely stopped during last exposure of {key} (before image acquisition)")
                    filtered_on_off_timestamps = on_off_timestamps[0: -1]
                    on_off_timestamps = filtered_on_off_timestamps

            if key in ["trial_TTL"]:
                # Detection of early licks
                iti = np.array([on_off_timestamps[i+1][0] - on_off_timestamps[i][1]
                                for i in range(len(on_off_timestamps) - 1)])
                early_licks = np.where(iti < 0.4)[0]  # reset trial signal in less than 0.25 s (specific to early lick)
                print(f"{len(early_licks)} early licks")

                if len(early_licks) > 0:
                    early_licks = list(early_licks)
                    early_licks_true_ind = [i - early_licks.index(i) for i in early_licks]
                    print(f"early licks trial indexes: {early_licks_true_ind}")
                    on_off_to_remove = np.array([i + 1 for i in early_licks])
                    filtered_on_off_timestamps = np.delete(on_off_timestamps, on_off_to_remove, axis=0)
                    on_off_timestamps = filtered_on_off_timestamps

            if key in ["trial_TTL"] and binary_data[-1] == 1:
                print(f"Session likely stopped before end of last {key}")
                filtered_on_off_timestamps = on_off_timestamps[0: -1]  # remove last timestamp that signals session end
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


def read_behavior_avi_movie(movie_files):

    for movie_file in movie_files:
        movie_name = os.path.split(movie_file)[1]
        video_capture = cv2.VideoCapture(movie_file)

        # Check if camera opened successfully
        if not video_capture.isOpened():
            print("Error opening video stream or file")

        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = np.round(video_capture.get(cv2.CAP_PROP_FPS), 2)

        print(f"AVI name : {movie_name}")
        print(f"AVI video frames: {video_length}, @ {video_frame_rate} Hz")

        return video_length, video_frame_rate


def print_info_dict(my_dict):
    for key, data in my_dict.items():
        print(f"- {key}: {data}")

