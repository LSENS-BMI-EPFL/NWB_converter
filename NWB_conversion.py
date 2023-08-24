"""_summary_
"""
import os
import yaml
from converters.subject_to_nwb import create_nwb_file
from converters.ci_movie_to_nwb import convert_ci_movie
from converters.suite2p_to_nwb import convert_suite2p_data
from converters.nwb_saving import save_nwb_file
from converters.behavior_to_nwb import convert_behavior_data
from continuous_log_analysis import analyze_continuous_log
from utils.behavior_converter_misc import find_training_days
from utils.server_paths import get_subject_data_folder, get_subject_analysis_folder, get_nwb_folder
from utils.ecephys_utils import get_ephys_sync_timestamps


def convert_data_to_nwb(config_file, output_folder):
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

    # Read config file to know what data to convert.
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)

    print("Start NWB conversion")
    print(" ")
    print("Extract timestamps")

    timestamps_dict, n_frames_dict = analyze_continuous_log(config_file=config_file,
                                                            do_plot=False, plot_start=None,
                                                            plot_stop=None, camera_filtering=False)

    print(" ")
    print("Open NWB file and add metadata")
    nwb_file = create_nwb_file(config_file=config_file)

    # # TODO: update/remove the link to motion corrected ci tiff.
    if config_dict.get("2P_metadata") is not None:
        print(" ")
        print("Convert CI movie")
        convert_ci_movie(nwb_file=nwb_file, config_file=config_file,
                         movie_format='link', ci_frame_timestamps=timestamps_dict['galvo_position'])

        # # TODO: find suite2P folder with config file.
        print(" ")
        print("Convert Suite2p data")
        convert_suite2p_data(nwb_file=nwb_file,
                             config_file=config_file,
                             ci_frame_timestamps=timestamps_dict['galvo_position'])

    if config_dict.get("ephys_metadata") is not None:
        print(" ")
        print("Convert ephys spike data")

    print(" ")
    print("Convert Behavior data")
    convert_behavior_data(nwb_file, timestamps_dict, config_file)

    print(" ")
    print("Saving NWB file")
    save_nwb_file(nwb_file=nwb_file, output_folder=output_folder)

    return


if __name__ == '__main__':
    # Run the conversion
    # mouse_ids = ['RD001', 'RD002', 'RD003', 'RD004', 'RD005', 'RD006']
    # mouse_ids = ['RD001', 'RD003', 'RD005']
    # mouse_ids = ['RD002', 'RD004', 'RD006']
    # mouse_ids = ['RD002', 'RD004']
    mouse_ids = ['AB082']

    for mouse_id in mouse_ids:
        data_folder = get_subject_data_folder(mouse_id)
        analysis_folder = get_subject_analysis_folder(mouse_id)
        nwb_folder = get_nwb_folder(mouse_id)
        # nwb_folder = 'C:/Users/rdard/Documents/NWB_files/Anthony'

        # Find session list and session description.
        training_days = find_training_days(mouse_id, data_folder)

        # Create NWB by looping over sessions.
        for isession, iday in training_days:
            # Filter sessions to do :
            session_to_do = ["PB124_20230404_141456"]
            # session_to_do = ["RD001_20230624_123913", "RD003_20230624_134719", "RD005_20230624_145511"]
            # if isession not in session_to_do:
            #    continue

            # date_to_do = "20230629"
            # if date_to_do not in isession:
            #     continue

            # Find yaml config file and behavior results for this session.
            config_yaml = os.path.join(analysis_folder, isession, f"config_{isession}.yaml")
            # Make conversion.
            print(f" ------------------ ")
            print(f"Session: {isession}")
            convert_data_to_nwb(config_file=config_yaml, output_folder=nwb_folder)
