"""_summary_
"""
import datetime
import os
import platform
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed

import yaml
import json

import utils.utils_gf as utils_gf
from continuous_log_analysis import analyze_continuous_log
from converters.behavior_to_nwb import convert_behavior_data
from converters.ci_movie_to_nwb import convert_ci_movie
from converters.ephys_to_nwb_test import convert_ephys_recording #todo: finish
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
        config_dict = yaml.load(stream, Loader=yaml.UnsafeLoader)
        #config_dict = yaml.safe_load(stream)

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
    convert_behavior_data(nwb_file=nwb_file, timestamps_dict=timestamps_dict, config_file=config_file)

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
             add_recordings=False
             if add_recordings:
                 output_folder = Path(output_folder) / 'nwb_recordings'
             convert_ephys_recording(nwb_file=nwb_file,
                                     config_file=config_file,
                                     add_recordings=add_recordings,
                                     experimenter=experimenter)

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

def convert_single_session(config_yaml, nwb_folder, experimenter_full, isession):
    """Wrapper function to convert a single session to NWB format."""
    print(" ------------------ ")
    print(f"Session: {isession}")
    
    try:
        convert_data_to_nwb(config_file=config_yaml,
                           output_folder=nwb_folder,
                           with_time_string=False,
                           experimenter=experimenter_full)
        return f"Successfully processed {isession}"
    except Exception as e:
        print(f"Error processing {isession}: {str(e)}")
        return f"Error processing {isession}: {str(e)}"
if __name__ == '__main__':

    # Run the conversion
    mouse_ids_mh = [
        'MH004',
        'MH006',
        'MH007',
        'MH008',
        'MH009',
        'MH010',
        'MH011',
        'MH013',
        'MH014',
        'MH015',
        'MH016',
        'MH017',
        'MH018',
        'MH019',
        'MH020',
        'MH021',
        'MH022',
        'MH023',
        'MH025',
        'MH026',
        'MH027',
        'MH028',
        'MH029',
        'MH030',
        #'MH031',
        'MH032',
        'MH034',
        'MH035',
        'MH036',
        'MH037',
        #'MH038',
        'MH039',
        'MH062',
        'MH064',
        'MH065',
        'MH068',
        'MH069',
        'MH070',
    ]
    mouse_ids = [
        'AB077',
        'AB079',
        'AB080',
        'AB082',
        'AB085',
        'AB086',
        'AB087',
        'AB091',
        'AB092',
        'AB093',
        'AB094',
        'AB095',
        'AB102',
        'AB104',
        'AB107',
        'AB116',
        'AB117',
        'AB119',
        'AB120',
        'AB121',
        'AB122',
        'AB123',
        'AB124',
        'AB125',
        'AB126',
        'AB127',
        'AB128',
        'AB129',
        'AB130',
        'AB131',
        'AB132',
        'AB133',
        'AB134',
        'AB135',
        'AB136',
        'AB137',
        'AB138',
        'AB139',
        'AB140',
        'AB141',
        'AB142',
        'AB143',
        'AB144',
        'AB145',
        'AB147',
        'AB149',
        'AB150',
        'AB151',
        'AB152',
        'AB153',
        'AB154',
        'AB155',
        'AB156',
        'AB157',
        'AB158',
        'AB159',
        'AB161',
        'AB162',
        'AB163',
        'AB164',
    ]
    mouse_ids = mouse_ids + mouse_ids_mh
    mouse_ids = ['MH035']


    session_not_to_do = ['MH007_20250128_110740', 'MH007_20250128_110814']

    experimenter = 'AB'
    experimenter_full = 'Axel_Bisi'
    # last_done_day = '20240506'
    last_done_day = None
    skip_existing_files = False # Overwrite if False
    n_jobs = 30
    sessions_to_convert = []

    for mouse_id in mouse_ids:
        data_folder = get_subject_data_folder(mouse_id)
        if os.path.exists(data_folder):
            pass
        else:
            print(f"No mouse data folder for {mouse_id}.")
            continue
        analysis_folder = get_subject_analysis_folder(mouse_id, experimenter=experimenter_full)
        nwb_folder = get_nwb_folder(mouse_id, experimenter=experimenter_full)
        nwb_folder = r"M:\analysis\Axel_Bisi\NWB_new"

        sessions_done = Path(nwb_folder).glob('*.nwb')
        sessions_done = [f.stem for f in sessions_done]

        training_days = find_training_days(mouse_id, data_folder)

        # Create NWB by looping over sessions.
        for isession, iday in training_days:

            # Filter session ID to do.
            #if isession not in sessions_to_do:
            #    continue

           # if skip_existing_files:
           #     session_not_to_do = session_not_to_do + sessions_done
            #if isession in session_not_to_do:
            #    print(f'Skipping {isession}')
            #    continue

            # Filter by date.
            session_date = isession.split('_')[1]
            session_date = datetime.datetime.strptime(session_date, "%Y%m%d")

            #if last_done_day is not None:
            #    if session_date <= datetime.datetime.strptime(last_done_day, "%Y%m%d"):
            #        continue
            #    else:
            #        print('Converting', isession)

            # Filter by session type.
            last_session_type = training_days[-1][1]

            #if experimenter == 'AB' and iday not in ['whisker_0']:
            #    continue
            #if experimenter == 'MH' and iday not in ['whisker_0']: # not in ['whisker_0', 'whisker_+1', 'whisker_+2', 'whisker_+3', 'whisker_+4']:
            #    continue
            #elif experimenter == 'PB' and iday!=last_session_type:
            #    continue

            print('Converting', isession)

            # Find yaml config file and behavior results for this session.
            config_yaml = os.path.join(analysis_folder, isession, f"config_{isession}.yaml")
            
            # Add to list of sessions to convert
            sessions_to_convert.append((config_yaml, nwb_folder, experimenter_full, isession))

    # Now run all conversions in parallel
    print(f"Converting {len(sessions_to_convert)} sessions in parallel...")

    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(convert_single_session)(config_yaml, nwb_folder, experimenter_full, isession)
        for config_yaml, nwb_folder, experimenter_full, isession in sessions_to_convert
    )

    # Print results
    successful = [r for r in results if r.startswith("Successfully")]
    errors = [r for r in results if r.startswith("Error")]

    print(f"\nProcessing complete:")
    print(f"  - Successfully processed: {len(successful)} sessions")
    print(f"  - Errors: {len(errors)} sessions")

    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  {error}")