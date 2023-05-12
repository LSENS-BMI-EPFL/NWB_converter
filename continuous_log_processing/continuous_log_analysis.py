import os
import yaml
import numpy as np
from continuous_log_processing.misc_functions.utils import get_file_size, read_binary_continuous_log, \
    plot_continuous_data_dict, extract_timestamps, read_behavior_avi_movie, read_tiff_ci_movie, \
    read_timestamps_from_camera_control, read_tiff_ci_movie_frames, print_info_dict


def analyze_continuous_log(config_file, do_plot=False, plot_start=None, plot_stop=None):
    with open(config_file, 'r') as stream:
        config_file_data = yaml.safe_load(stream)
    root_path = config_file_data.get("root_path")
    session_folder = config_file_data.get("session_folder")
    bin_file_path = os.path.join(root_path, session_folder, "log_continuous.bin")
    channels_dict = config_file_data.get("channels_dict")
    threshold_dict = config_file_data.get("threshold_dict")
    movie_files = config_file_data.get("movie_files_path")
    timestamps_file = config_file_data.get("movie_timestamps_path")
    tiff_file = config_file_data.get("ci_tiff_path")
    scanimage_dict = config_file_data.get("scan_image_dict")

    camera_filtering = False
    save_ts_dict = False

    bin_file_size = get_file_size(bin_file=bin_file_path)

    continuous_data_dict = read_binary_continuous_log(bin_file=bin_file_path,
                                                      channels_dict=channels_dict, ni_session_sr=5000, t_stop=None)

    if movie_files is None:
        camera_filtering = False
    timestamps_dict, n_frames_dict = extract_timestamps(continuous_data_dict, threshold_dict, scanimage_dict,
                                                        ni_session_sr=5000, filter_cameras=camera_filtering)
    print(f"Number of frames per acquisition : ")
    print_info_dict(n_frames_dict)

    if save_ts_dict:
        saving_timestamps = os.path.join(root_path, session_folder, "timestamps_dict")
        np.savez(saving_timestamps, **timestamps_dict)

    if do_plot:
        plot_continuous_data_dict(continuous_data_dict, timestamps_dict, ni_session_sr=5000,
                                  t_start=plot_start, t_stop=plot_stop,
                                  black_background=False)

    if movie_files is not None:
        read_behavior_avi_movie(movie_files=movie_files)

    # Was used in previous version where behavior movies were acquired with camera control GUI
    if timestamps_file is not None:
        read_timestamps_from_camera_control(timestamps_file=timestamps_file)

    if tiff_file is not None:
        read_tiff_ci_movie_frames(tiff_file)

    return timestamps_dict, n_frames_dict

# This function can be used alone, in that case un-comment 2 lines below
# yaml_file = "C:/Users/rdard/Documents/test_data/log_file_config.yml"
# analyze_continuous_log(config_file=yaml_file)




