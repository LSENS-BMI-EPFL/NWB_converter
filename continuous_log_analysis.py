import yaml
import os
from utils.continuous_processing import read_binary_continuous_log, \
    plot_continuous_data_dict, extract_timestamps, read_behavior_avi_movie, \
    read_tiff_ci_movie_frames, print_info_dict
from utils import server_paths


def analyze_continuous_log(config_file, do_plot=False, plot_start=None, plot_stop=None, camera_filtering=False):

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
        bin_file = server_paths.get_log_continuous_file(config_file)
        movie_files = server_paths.get_movie_files(config_file)
        tiff_file = server_paths.get_imaging_file(config_file)

    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    channels_dict = config['log_continuous_metadata']['channels_dict']
    threshold_dict = config['log_continuous_metadata']['threshold_dict']
    scanimage_dict = config['log_continuous_metadata']['scanimage_dict']

    continuous_data_dict = read_binary_continuous_log(bin_file=bin_file,
                                                      channels_dict=channels_dict, ni_session_sr=5000, t_stop=None)

    if movie_files is None:
        camera_filtering = False
    timestamps_dict, n_frames_dict = extract_timestamps(continuous_data_dict, threshold_dict, scanimage_dict,
                                                        ni_session_sr=5000, filter_cameras=camera_filtering)
    print("Number of frames per acquisition: ")
    print_info_dict(n_frames_dict)

    if do_plot:
        plot_continuous_data_dict(continuous_data_dict, timestamps_dict, ni_session_sr=5000,
                                  t_start=plot_start, t_stop=plot_stop,
                                  black_background=False)

    if movie_files is not None:
        read_behavior_avi_movie(movie_files=movie_files)

    if tiff_file is not None:
        print(f"Tiff file found, reading number of CI frames")
        read_tiff_ci_movie_frames(tiff_file)

    return timestamps_dict, n_frames_dict


if __name__ == "__main__":
    # This is a simplified config_file with only necessary to analyse quickly the continuous logging
    yaml_file = "C:/Users/rdard/Documents/test_data/log_file_config.yml"
    analyze_continuous_log(config_file=yaml_file,
                           do_plot=False, plot_start=800, plot_stop=1000, camera_filtering=False)

