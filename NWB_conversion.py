from converters.subject_to_nwb import create_nwb_file
from converters.ci_movie_to_nwb import convert_ci_movie
from converters.suite2p_to_nwb import convert_suite2p_data
from converters.nwb_saving import save_nwb_file
from converters.behavior_to_nwb import convert_behavior_data
from continuous_log_processing.continuous_log_analysis import analyze_continuous_log
import os


def convert_data_to_nwb(subject_data_yaml, session_data_yaml, log_yaml_file, two_p_yaml_file, suite2p_folder,
                        behavior_results_file, bhv_mapping_file, output_folder):
    """

    :param subject_data_yaml: Path to the subject data yaml config file containing mouse ID and metadata
    :param session_data_yaml: Path to the session data yaml config file containing session metadata
    :param log_yaml_file: Path to the yaml config file used to analyse continuous logging and behavior
    :param two_p_yaml_file: Path to the yaml config file containing 2P info
    :param suite2p_folder: Path to the suite2p output folder
    :param behavior_results_file: Path to the 'results.csv' behavior file
    :param bhv_mapping_file: Path to the dictionary mapping performance number in behavior file with trial name
    :param output_folder: Path to the folder to save NWB files
    :return: NWB file
    """

    print(f"Start NWB conversion")
    print(f" ")
    print(f"Extract timestamps from continuous log")
    timestamps_dict, n_frames_dict = analyze_continuous_log(config_file=log_yaml_file, do_plot=False,
                                                            plot_start=None, plot_stop=None)

    print(f" ")
    print(f"Open NWB file and add metadata")
    nwb_file = create_nwb_file(subject_data_yaml_file=subject_data_yaml,
                               session_data_yaml_file=session_data_yaml)

    print(f" ")
    print(f"Convert CI movie")
    convert_ci_movie(nwb_file=nwb_file,
                     two_p_yaml_file=two_p_yaml_file,
                     log_yaml_file=log_file,
                     movie_format='link')

    print(f" ")
    print(f"Convert Suite2p data")
    if suite2p_folder is not None:
        convert_suite2p_data(nwb_file=nwb_file,
                             suite2p_folder=suite2p_folder,
                             ci_frame_timestamps=timestamps_dict['galvo_position'][0:-2])
    else:
        print(f"No Suite2p data to add")

    print(f" ")
    print(f"Convert Behavior data")
    convert_behavior_data(nwb_file, timestamps_dict, behavior_results_file, bhv_mapping_file)

    print(f" ")
    print(f"Saving NWB file")
    save_nwb_file(nwb_file=nwb_file, output_folder=output_folder)


# Run the conversion
root_path = "C:/Users/rdard/Documents/NWB_to_create"
mouse_id = "PB124"
day = "whisker_day0"
bhv_session = "PB124_20230404_141456"
subject_yaml = os.path.join(root_path, mouse_id, day, "subject_data.yml")
session_yaml = os.path.join(root_path, mouse_id, day, "session_data.yml")
log_file = os.path.join(root_path, mouse_id, day, f"log_file_config_{bhv_session}.yml")
two_p_yml = os.path.join(root_path, mouse_id, day, "2P_setup.yml")
s2p_folder = os.path.join(root_path, mouse_id, day, "suite2p")
# s2p_folder = None
bhv_file = os.path.join(root_path, mouse_id, day, bhv_session, "results.csv")
bhv_trials_names = os.path.join(root_path, mouse_id, day, "trial_type_mapping.yml")
nwb_output_folder = "C:/Users/rdard/Documents/NWB_files"

convert_data_to_nwb(subject_data_yaml=subject_yaml, session_data_yaml=session_yaml, log_yaml_file=log_file,
                    two_p_yaml_file=two_p_yml, suite2p_folder=s2p_folder, behavior_results_file=bhv_file,
                    bhv_mapping_file=bhv_trials_names, output_folder=nwb_output_folder)
