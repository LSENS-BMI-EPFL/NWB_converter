"""_summary_
"""
# Imports
import os
import yaml
from converters.subject_to_nwb import create_nwb_file
from converters.ci_movie_to_nwb import convert_ci_movie
from converters.suite2p_to_nwb import convert_suite2p_data
from converters.nwb_saving import save_nwb_file
from converters.behavior_to_nwb import convert_behavior_data
from converters.images_to_nwb import convert_images_data
from converters.ephys_to_nwb import convert_ephys_recording
from continuous_log_analysis import analyze_continuous_log
from utils.behavior_converter_misc import find_training_days
from utils.server_paths import get_subject_data_folder, get_subject_analysis_folder, get_nwb_folder


def convert_data_to_nwb(config_file, output_folder):
    """
    :param config_file: Path to the yaml config file containing mouse ID and metadata for the session to convert
    :param output_folder: Path to the folder to save NWB files
    :return: NWB file
    """

    # Read config file to know what data to convert.
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)

    print(" ")
    print("Start NWB conversion")

    print(" ")
    print("Extract timestamps")

    timestamps_dict, n_frames_dict = analyze_continuous_log(config_file=config_file,
                                                            do_plot=False, plot_start=None,
                                                            plot_stop=None, camera_filtering=False)

    print(" ")
    print("Open NWB file and add metadata")
    nwb_file = create_nwb_file(config_file=config_file)


    print(" ")
    print("Convert behavior data")
    convert_behavior_data(nwb_file=nwb_file, timestamps_dict=timestamps_dict, config_file=config_file)


    # # TODO: update/remove the link to motion corrected ci tiff.
    if config_dict.get("two_photon_metadata") is not None:
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
        print("Convert extracellular electrophysiology data")

        convert_ephys_recording(nwb_file=nwb_file,
                             config_file=config_file)


    print(" ")
    print("Saving NWB file")
    save_nwb_file(nwb_file=nwb_file, output_folder=output_folder)

    return


if __name__ == '__main__':

    # Run the conversion
    # mouse_ids = ['PB164', 'PB165', 'PB166', 'PB167', 'PB168']
    mouse_ids = ['PB000']

    for mouse_id in mouse_ids:
        data_folder = get_subject_data_folder(mouse_id)
        analysis_folder = get_subject_analysis_folder(mouse_id)
        nwb_folder = get_nwb_folder(mouse_id)

        # Find session list and session description.
        training_days = find_training_days(mouse_id, data_folder)

        # Create NWB by looping over sessions.
        for isession, iday in training_days:

            print(isession, iday)
            # Filter sessions to do :
            # session_to_do = ["RD001_20230624_123913", "RD003_20230624_134719", "RD005_20230624_145511"]

            if 'calibration' in isession:
                continue
            session_to_do = ["PB000_20230922_173524"]
            if isession not in session_to_do:
                continue
            #
            # date_to_do = "20231025"
            # if date_to_do not in isession:
            #     continue

            # Find yaml config file and behavior results for this session.
            config_yaml = os.path.join(analysis_folder, isession, f"config_{isession}.yaml")

            # Make conversion.
            print(f" ------------------ ")
            print(f"Session: {isession}")
            convert_data_to_nwb(config_file=config_yaml, output_folder=nwb_folder)
