import ast
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('QtAgg')
import yaml

from utils import read_sglx, server_paths, tiff_loading, widefield_utils
from utils.continuous_processing import (extract_timestamps,
                                         plot_continuous_data_dict,
                                         print_info_dict,
                                         read_behavior_avi_movie,
                                         read_binary_continuous_log)
from utils.ephys_converter_misc import extract_ephys_timestamps, read_ephys_binary_data


def analyze_continuous_log(config_file, do_plot=False, plot_start=None, plot_stop=None, camera_filtering=False):
    """
    Extract timestamps from continuous logging data and plot the data if do_plot is True.
    Args:
        config_file:     Path to the yaml config file used to analyse continuous logging and behavior
        do_plot:         If True, plot the continuous logging data
        plot_start:      Start time of the plot in seconds
        plot_stop:       Stop time of the plot in seconds
        camera_filtering:

    Returns:
        timestamps_dict: Dictionary containing timestamps for each data type
        n_frames_dict:   Dictionary containing number of timestamps for each data type

    """

    if __name__ == "__main__":
        print('Standalone analysis')
        with open(config_file, 'r', encoding='utf8') as stream:
            config = yaml.safe_load(stream)
        bin_file = os.path.join(config['root_path'], "log_continuous.bin")
        bin_file_cor = os.path.join(config['root_path'], "log_continuous_cor.bin")
        if os.path.exists(bin_file_cor):
            bin_file = bin_file_cor

        movie_names = config['movie_files_path']
        if movie_names is not None:
            movie_files = [os.path.join(config['root_path'], movie_name) for movie_name in movie_names]
        else:
            movie_files = None
        tiff_file = config['ci_tiff_path']
        mj2_file = config['mj2_file_path']

    else:
        # Load NWB config file
        with open(config_file, 'r', encoding='utf8') as stream:
            config = yaml.safe_load(stream)

        bin_file = server_paths.get_log_continuous_file(config_file)

        if config['session_metadata']['experimenter'] == 'AB':
            # Check if continuous processing is required
            exp_desc = ast.literal_eval(config.get('session_metadata').get('experiment_description'))
            if exp_desc['session_type'] == 'behaviour_only_session':
                if os.path.isfile(bin_file):
                    pass
                else:
                    print("No continuous data found for this session. No timestamps available: using trial table "
                          "information only")
                    return None, None

            movie_files = server_paths.get_session_movie_files(config_file)
        else:
            movie_files = server_paths.get_movie_files(config_file)

        tiff_file = server_paths.get_imaging_file(config_file)

        mj2_file = server_paths.get_widefield_file(config_file)

    # Get relevant continuous processing information
    channels_dict = config['log_continuous_metadata']['channels_dict']
    threshold_dict = config['log_continuous_metadata']['threshold_dict']

    if 'scanimage_dict' in config['log_continuous_metadata']:
        scanimage_dict = config['log_continuous_metadata']['scanimage_dict']
    else:
        scanimage_dict = None

    # Extract session timestamps
    continuous_data_dict = read_binary_continuous_log(bin_file=bin_file,
                                                      channels_dict=channels_dict,
                                                      ni_session_sr=5000,
                                                      t_stop=None)
    if continuous_data_dict is None:
        print("No continuous data found for this session. No timestamps available: using trial table information only")
        return None, None

    if movie_files is None:
        camera_filtering = False

    timestamps_dict, n_frames_dict = extract_timestamps(continuous_data_dict, threshold_dict,
                                                        ni_session_sr=5000,
                                                        scanimage_dict=scanimage_dict,
                                                        filter_cameras=camera_filtering,
                                                        wf_file=mj2_file[0] if mj2_file is not None else None)

    # Optionally plot log_continuous.bin data for inspection, given a start and stop time
    if do_plot:
        plot_continuous_data_dict(continuous_data_dict=continuous_data_dict,
                                  timestamps_dict=timestamps_dict,
                                  ni_session_sr=5000,
                                  t_start=plot_start,
                                  t_stop=plot_stop,
                                  black_background=False)

    if 'ephys_metadata' in config:
        ephys_nidq_meta, ephys_nidq_bin = server_paths.get_raw_ephys_nidq_files(config_file)
        ephys_cont_data_dict = read_ephys_binary_data(ephys_nidq_bin, ephys_nidq_meta)
        ephys_timestamps_dict, ephys_n_frames_dict = extract_ephys_timestamps(config_file=config_file,
                                                                        continuous_data_dict=ephys_cont_data_dict,
                                                                        threshold_dict=threshold_dict,
                                                                        log_timestamps_dict=timestamps_dict,
                                                                        n_frames_dict=n_frames_dict)
        timestamps_dict = ephys_timestamps_dict
        print('Number of timestamps per acquisition (ephys):')
        print_info_dict(ephys_n_frames_dict)

    print("Number of timestamps per acquisition:")
    print_info_dict(n_frames_dict)

    if __name__ == "__main__":
        if movie_files is not None:
            print(f"Check number video filming frames")
            for movie in movie_files:
                read_behavior_avi_movie(movie_file=movie)

        if tiff_file is not None:
            print(f"Tiff file found, reading number of CI frames")
            tiff_loading.get_tiff_movie_shape(tiff_file)

        if mj2_file is not None:
            print(f"Motion JPEG 2000 file found, reading number of widefield frames")
            widefield_utils.read_motion_jpeg_2000_movie(mj2_file=mj2_file)

    return timestamps_dict, n_frames_dict


if __name__ == "__main__":
    # This is a simplified config_file to analyse quickly the continuous logging
    yaml_file = r"\\sv-nas1.rcp.epfl.ch\Petersen-Lab\analysis\Anthony_Renard\data\AR176\AR176_20241215_160714\config_AR176_20241215_160714.yaml"
    analyze_continuous_log(config_file=yaml_file,
                           do_plot=False, plot_start=0, plot_stop=1500, camera_filtering=False)
