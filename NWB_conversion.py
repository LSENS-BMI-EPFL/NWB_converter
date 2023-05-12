from converters.subject_to_nwb import create_nwb_file
from converters.ci_movie_to_nwb import convert_ci_movie
from converters.suite2p_to_nwb import convert_suite2p_data
from converters.nwb_saving import save_nwb_file
from converters.behavior_to_nwb import convert_behavior_data
from continuous_log_processing.continuous_log_analysis import analyze_continuous_log


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
    print(f"Extract timestamps from continuous log")
    timestamps_dict, n_frames_dict = analyze_continuous_log(config_file=log_yaml_file, do_plot=False,
                                                            plot_start=None, plot_stop=None)

    print(f"Open NWB file and add metadata")
    nwb_file = create_nwb_file(subject_data_yaml_file=subject_data_yaml,
                               session_data_yaml_file=session_data_yaml)

    print(f"Convert CI movie")
    convert_ci_movie(nwb_file=nwb_file,
                     two_p_yaml_file=two_p_yaml_file,
                     log_yaml_file=log_file,
                     movie_format='link')

    print(f"Convert Suite2p data")
    convert_suite2p_data(nwb_file=nwb_file,
                         suite2p_folder=suite2p_folder,
                         ci_frame_timestamps=timestamps_dict['galvo_position'][0:-2])

    print(f"Convert Behavior data")
    convert_behavior_data(nwb_file, timestamps_dict, behavior_results_file, bhv_mapping_file)

    print(f"Saving NWB file")
    save_nwb_file(nwb_file=nwb_file, output_folder=output_folder)


subject_yaml = "C:/Users/rdard/Documents/NWB_to_create/PB124/whisker_day0/subject_data.yml"
session_yaml = "C:/Users/rdard/Documents/NWB_to_create/PB124/whisker_day0/session_data.yml"
log_file = "C:/Users/rdard/Documents/NWB_to_create/PB124/whisker_day0/log_file_config_PB124_wd0.yml"
two_p_yml = "C:/Users/rdard/Documents/NWB_to_create/PB124/whisker_day0/2P_setup.yml"
s2p_folder = "C:/Users/rdard/Documents/NWB_to_create/PB124/whisker_day0/suite2p"
bhv_file = "C:/Users/rdard/Documents/NWB_to_create/PB124/whisker_day0/PB124_20230404_141456/results.csv"
bhv_trials_names = "C:/Users/rdard/Documents/NWB_to_create/PB124/whisker_day0/trial_type_mapping.yml"
nwb_output_folder = "C:/Users/rdard/Documents/NWB_files"

convert_data_to_nwb(subject_data_yaml=subject_yaml, session_data_yaml=session_yaml, log_yaml_file=log_file,
                    two_p_yaml_file=two_p_yml, suite2p_folder=s2p_folder, behavior_results_file=bhv_file,
                    bhv_mapping_file=bhv_trials_names, output_folder=nwb_output_folder)
