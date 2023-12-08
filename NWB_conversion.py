"""_summary_
"""
# Imports
import datetime
import os

import yaml

import utils.gf_utils as utils_gf
from continuous_log_analysis import analyze_continuous_log
from converters.behavior_to_nwb import convert_behavior_data
from converters.ci_movie_to_nwb import convert_ci_movie
from converters.ephys_to_nwb import convert_ephys_recording
from converters.images_to_nwb import convert_images_data
from converters.nwb_saving import save_nwb_file
from converters.subject_to_nwb import create_nwb_file
from converters.suite2p_to_nwb import convert_suite2p_data
from utils.behavior_converter_misc import find_training_days
from utils.server_paths import (get_nwb_folder, get_subject_analysis_folder,
                                get_subject_data_folder)


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

    if config_dict['session_metadata']['experimenter'] != 'GF':
        timestamps_dict, n_frames_dict = analyze_continuous_log(config_file=config_file,
                                                                do_plot=False, plot_start=None,
                                                                plot_stop=None, camera_filtering=False)
    else:
        timestamps_dict, n_frames_dict = utils_gf.infer_timestamps_dict(config_file=config_file)

    print(" ")
    print("Open NWB file and add metadata")
    nwb_file = create_nwb_file(config_file=config_file)

    print(" ")
    print("Convert behavior data")
    convert_behavior_data(nwb_file=nwb_file, timestamps_dict=timestamps_dict, config_file=config_file)

    if config_dict.get("two_photon_metadata") is not None:
        print(" ")
        print("Convert CI movie")
        convert_ci_movie(nwb_file=nwb_file, config_file=config_file,
                         movie_format='link', ci_frame_timestamps=timestamps_dict['galvo_position'])

        print(" ")
        print("Convert Suite2p data")
        convert_suite2p_data(nwb_file=nwb_file,
                             config_file=config_file,
                             ci_frame_timestamps=timestamps_dict['galvo_position'])

    if config_dict.get("ephys_metadata") is not None and config_dict.get("ephys_metadata").get("processed") == "true":
        print(" ")
        print("Convert extracellular electrophysiology data")

        convert_ephys_recording(nwb_file=nwb_file,
                                config_file=config_file)

    print(" ")
    print("Saving NWB file")
    save_nwb_file(nwb_file=nwb_file, output_folder=output_folder, with_time_string=True)

    return


if __name__ == '__main__':

    # Run the conversion
    # mouse_ids = ['RD001', 'RD002', 'RD003', 'RD004', 'RD005', 'RD006']
    # mouse_ids = ['RD013', 'RD014', 'RD015', 'RD016', 'RD017']
    # mouse_ids = ['RD025', 'RD026']
    mouse_ids = ['RD027', 'RD029', 'RD030', 'RD031', 'RD032']
    # mouse_ids = ['RD030']
    # mouse_ids = ['RD033', 'RD034', 'RD035', 'RD036']
    experimenter = 'RD'

    if experimenter == 'GF':
        # Read excel database.
        db_folder = 'C:\\Users\\aprenard\\recherches\\fast-learning\\docs'
        db_name = 'sessions_GF.xlsx'
        db = utils_gf.read_excel_database(db_folder, db_name)

    for mouse_id in mouse_ids:
        data_folder = get_subject_data_folder(mouse_id)
        analysis_folder = get_subject_analysis_folder(mouse_id)
        # nwb_folder = get_nwb_folder(mouse_id)
        nwb_folder = "C:/Users/rdard/Documents/NWB_files"

        # Find session list and session description.
        if experimenter == 'GF':
            training_days = db.loc[db.subject_id == mouse_id, 'session_day'].to_list()
            training_days = utils_gf.format_session_day_GF(mouse_id, training_days)
            sessions = db.loc[db.subject_id == mouse_id, 'session_id'].to_list()
            training_days = list(zip(sessions, training_days))
        else:
            training_days = find_training_days(mouse_id, data_folder)

        # Create NWB by looping over sessions.
        for isession, iday in training_days:

            # # Filter sessions to do :
            # session_to_do = ["GF307_20112020_082942"]
            # if isession not in session_to_do:
            #    continue

            date_to_do = "20231129"
            if date_to_do not in isession:
                continue

            # session_date = isession.split('_')[1]
            # session_date = datetime.datetime.strptime(session_date, "%Y%m%d")

            # if last_done_day is not None:
            #     if session_date <= datetime.datetime.strptime(last_done_day, "%Y%m%d"):
            #         continue
            # Find yaml config file and behavior results for this session.
            config_yaml = os.path.join(analysis_folder, isession, f"config_{isession}.yaml")

            # Make conversion.
            print(f" ------------------ ")
            print(f"Session: {isession}")
            convert_data_to_nwb(config_file=config_yaml, output_folder=nwb_folder)

