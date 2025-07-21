"""_summary_
"""
import datetime
import os
import platform
from pathlib import Path
import numpy as np

import yaml
import json

import utils.utils_gf as utils_gf
from continuous_log_analysis import analyze_continuous_log
from converters.behavior_to_nwb import convert_behavior_data
from converters.ci_movie_to_nwb import convert_ci_movie
from converters.ephys_to_nwb import convert_ephys_recording
from converters.nwb_saving import save_nwb_file
from converters.subject_to_nwb import create_nwb_file
from converters.suite2p_to_nwb import convert_suite2p_data
from converters.widefield_to_nwb import convert_widefield_recording
from converters.DLC_to_nwb import convert_dlc_data
from converters.facemap_to_nwb import convert_facemap_data
from utils.behavior_converter_misc import find_training_days
from utils.server_paths import (get_nwb_folder, get_subject_analysis_folder, get_experimenter_analysis_folder,
                                get_subject_data_folder, get_dlc_file_path, get_facemap_file_path)


def convert_data_to_nwb(config_file, output_folder, with_time_string=True, experimenter=None):
    """
    :param config_file: Path to the yaml config file containing mouse ID and metadata for the session to convert
    :param output_folder: Path to the folder to save NWB files
    :param experimenter: (Optional) experimenter initials, provide if experimenter and mouse initials are different
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
        timestamps_dict, _ = analyze_continuous_log(config_file=config_file,
                                                    do_plot=False, plot_start=1,
                                                    plot_stop=100, camera_filtering=False,
                                                    experimenter=experimenter)
    else:
        timestamps_dict, _ = utils_gf.infer_timestamps_dict(
            config_file=config_file)

    print(" ")
    print("Open NWB file and add metadata")
    nwb_file = create_nwb_file(config_file=config_file)

    print(" ")
    print("Convert behavior data")
    convert_behavior_data(
        nwb_file=nwb_file, timestamps_dict=timestamps_dict, config_file=config_file)

    if config_dict.get("two_photon_metadata") is not None:
        print(" ")
        print("Convert CI movie")
        convert_ci_movie(nwb_file=nwb_file, config_file=config_file, movie_format='link',
                         add_movie_data_or_link=True, ci_frame_timestamps=timestamps_dict['galvo_position'])

        print(" ")
        print("Convert Suite2p data")
        convert_suite2p_data(nwb_file=nwb_file,
                             config_file=config_file,
                             ci_frame_timestamps=timestamps_dict['galvo_position'])

    if config_dict.get("ephys_metadata") is not None:
        if config_dict.get("ephys_metadata").get("processed") == 1:
             print(" ")
             print("Convert extracellular electrophysiology data")
             convert_ephys_recording(nwb_file=nwb_file,
                                     config_file=config_file,
                                     experimenter=experimenter,
                                     )

    # Check we are on WF computer
    platform_info = platform.uname()
    computer_name = platform_info.node
    wf_computers = ['SV-07-082', 'SV-07-097']  # Add name of WF preprocessing computers here
    if computer_name in wf_computers and config_dict.get("widefield_metadata") is not None:
        print(" ")
        print("Convert widefield data")
        convert_widefield_recording(nwb_file=nwb_file,
                                    config_file=config_file,
                                    wf_frame_timestamps=timestamps_dict["widefield"])

    if config_dict.get('behaviour_metadata')['camera_flag'] == 1:
        dlc_file = get_dlc_file_path(config_file)
        if dlc_file is not None:
            print(" ")
            print("Convert DeepLabCut data")
            convert_dlc_data(nwb_file=nwb_file,
                             config_file=config_file,
                             video_timestamps={k: timestamps_dict[k] for k in ("cam1", "cam2")})

        facemap_file = get_facemap_file_path(config_file)
        if facemap_file is not None:
            print(" ")
            print("Convert Facemap data")
            convert_facemap_data(nwb_file=nwb_file,
                                 config_file=config_file,
                                 video_timestamps={k: timestamps_dict[k] for k in ("cam1", "cam2")})

    print(" ")
    print("Saving NWB file")
    save_nwb_file(nwb_file=nwb_file, output_folder=output_folder, with_time_string=with_time_string)

    return


if __name__ == '__main__':

    # Run the conversion
    mouse_ids = [
        # 'PB191', 
        # 'PB192', 
        # 'PB195', 
        # 'PB196', 
        # 'PB201',
        # 'RD076',
        # 'RD077',
        # 'RD072',
        # 'JL002',
        'PB191',
        ]

    sessions_to_do = [
        # 'PB191_20241210_110601',
        # 'PB192_20241211_113347',
        # 'PB195_20241214_114410',
        # 'PB196_20241217_144715',
        # 'PB201_20241212_192123',
        # 'RD076_20250214_125235',
        # 'RD072_20250305_131521',
        # 'RD077_20250219_183425',
        # 'RD077_20250221_102024',
        # 'JL002_20250507_135553',
        'PB191_20241210_110601',
    ]


    session_not_to_do = []
    experimenter = 'JL'    
    experimenter_full = 'Jules_Lebert'
    # last_done_day = '20240506'
    last_done_day = None
    skip_existing_files = False


    if experimenter == 'GF':
        # Read excel database.
        db_folder = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Anthony_Renard'
        db_name = 'sessions_GF.xlsx'
        db = utils_gf.read_excel_database(db_folder, db_name)

        mouse_ids = db['subject_id'].unique()
    
    for mouse_id in mouse_ids:
        data_folder = get_subject_data_folder(mouse_id)
        if os.path.exists(data_folder):
            pass
        else:
            print(f"No mouse data folder for {mouse_id}.")
            continue
        analysis_folder = get_subject_analysis_folder(mouse_id, experimenter=experimenter_full)
        nwb_folder = get_nwb_folder(mouse_id, experimenter=experimenter_full)

        sessions_done = Path(nwb_folder).glob('*.nwb')
        sessions_done = [f.stem for f in sessions_done]

        if experimenter == 'AB' and mouse_id.startswith('PB'):
            nwb_folder = get_experimenter_analysis_folder('AB')
            nwb_folder = os.path.join(nwb_folder, 'NWB')

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

            # Filter session ID to do.
            #session_to_do = ['AB105_20240314_115206']
            if isession not in sessions_to_do:
               continue

            if skip_existing_files:
                session_not_to_do = session_not_to_do + sessions_done
            if isession in session_not_to_do:
                print(f'Skipping {isession}')
                continue

            # Filter by date.
            #date_to_do = '20240217'
            #if date_to_do not in isession:
            #  continue

            # Filter by time since date.
            session_date = isession.split('_')[1]
            session_date = datetime.datetime.strptime(session_date, "%Y%m%d")
            if last_done_day is not None:
                if session_date <= datetime.datetime.strptime(last_done_day, "%Y%m%d"):
                     continue
                else:
                    print('Converting', isession)


            # Filter by session type.
            last_session_type = training_days[-1][1]
            if experimenter == 'AB' and iday not in ['whisker_0']:
                continue
            elif experimenter == 'PB' and iday!=last_session_type:
                continue
            print('Converting', isession)


            # Find yaml config file and behavior results for this session.
            config_yaml = os.path.join(analysis_folder, isession, f"config_{isession}.yaml")

            # Make conversion.
            print(" ------------------ ")
            print(f"Session: {isession}")
            convert_data_to_nwb(config_file=config_yaml,
                                output_folder=nwb_folder,
                                with_time_string=False,
                                experimenter=experimenter_full)
