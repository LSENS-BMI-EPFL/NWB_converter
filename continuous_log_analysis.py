import yaml
import os
import ast
import pathlib
import matplotlib.pyplot as plt
from utils.continuous_processing import read_binary_continuous_log, \
    plot_continuous_data_dict, extract_timestamps, read_behavior_avi_movie, \
    print_info_dict
from utils import server_paths
from utils import tiff_loading
from utils.ephys_converter_misc import extract_ephys_timestamps
from utils import read_sglx


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
        with open(config_file, 'r', encoding='utf8') as stream:
            config = yaml.safe_load(stream)
        bin_file = os.path.join(config['root_path'], "log_continuous.bin")
        movie_names = config['movie_files_path']
        if movie_names is not None:
            movie_files = [os.path.join(config['root_path'], movie_name) for movie_name in movie_names]
        else:
            movie_files = None
        tiff_file = config['ci_tiff_path']

    else:
        # Load NWB config file
        with open(config_file, 'r', encoding='utf8') as stream:
            config = yaml.safe_load(stream)

        bin_file = server_paths.get_log_continuous_file(config_file)

        if config['session_metadata']['experimenter'] == 'AB':
            # Check if continuous processing is required
            exp_desc = ast.literal_eval(config.get('session_metadata').get('experiment_description'))
            if exp_desc['session_type'] == 'behaviour_only_session':
                return None, None
            movie_files = server_paths.get_session_movie_files(config_file)
        else:
            movie_files = server_paths.get_movie_files(config_file)

        tiff_file = server_paths.get_imaging_file(config_file)

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
                                                        filter_cameras=camera_filtering)

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
        ephys_cont_data_dict = read_sglx.read_ephys_binary_data(ephys_nidq_bin, ephys_nidq_meta)
        ephys_timestamps_dict, n_frames_dict = extract_ephys_timestamps(config_file=config_file,
                                                                        continuous_data_dict=ephys_cont_data_dict,
                                                                        threshold_dict=threshold_dict,
                                                                        log_timestamps_dict=timestamps_dict)
        timestamps_dict = ephys_timestamps_dict

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

    return timestamps_dict, n_frames_dict


if __name__ == "__main__":
    # This is a simplified config_file with only necessary to analyse quickly the continuous logging
    yaml_file = "C:/Users/rdard/Documents/test_data/log_file_config.yml"
    analyze_continuous_log(config_file=yaml_file,
                           do_plot=False, plot_start=800, plot_stop=1000, camera_filtering=False)
