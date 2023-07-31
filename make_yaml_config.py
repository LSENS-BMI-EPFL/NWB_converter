"""_summary_
"""
import os
import datetime
import json
import yaml
import pandas as pd
from utils.behavior_converter_misc import find_training_days
from utils.server_paths import get_subject_data_folder, get_subject_analysis_folder


def make_yaml_config(subject_id, session_id, session_description, input_folder, output_folder,
                     mouse_line='C57BL/6', gmo=True):
    """_summary_

    Args:
        subject_id (_type_): _description_
        session_id (_type_): _description_
        session_description (_type_): _description_
        mouse_line (str, optional): _description_. Defaults to 'C57BL/6'.
        gmo (bool, optional): _description_. Defaults to True.
        server_mount (_type_, optional): _description_. Defaults to None.
    """

    # Subject metadata.
    # #################

    # Select most recent metadata export from SLIMS folder.
    slims_csv = sorted(os.listdir(os.path.join(input_folder, 'SLIMS')))[-1]
    slims_csv_path = os.path.join(input_folder, 'SLIMS', slims_csv)
    slims = pd.read_csv(slims_csv_path, sep=';', engine='python')
    slims = slims.loc[slims.cntn_cf_mouseName==mouse_id]

    subject_metadata = {
        'description': subject_id,
        'age__reference': 'birth',
        'strain': mouse_line,
        'sex': slims['cntn_cf_sex'].values[0].capitalize(),
        'species': 'Mus musculus',
        'subject_id': slims['cntn_cf_mouseName'].values[0],
        }

    # Compute age in days at session time.
    session_date = session_id.split('_')[1]
    session_date = datetime.datetime.strptime(session_date, "%Y%m%d")
    birth_date = slims['cntn_cf_dateofbirth'].values[0]
    birth_date = datetime.datetime.strptime(birth_date, "%d/%m/%Y")
    days = (session_date - birth_date).days
    subject_metadata['age'] = f"P{days}D"
    subject_metadata['date_of_birth'] = datetime.datetime.strftime(birth_date, '%m/%d/%Y')

    # Add strain from Slims if not WT mouse.
    # Wether mouse is WT in not in the default Slims metadata, so use 'gmo' parameter.
    # If you want to check if the mouse has the mutation with Slims, export the mutation column.
    if gmo:
        subject_metadata['genotype'] = slims['cntn_cf_strain'].values[0]
    else:
        subject_metadata['genotype'] = 'WT'
        subject_metadata['strain'] = slims['cntn_cf_strain'].values[0]

    # Add weight at the beginning of the session from json config file.
    json_path = os.path.join(input_folder, 'Training', session_id, 'session_config.json')
    with open(json_path, 'r') as f:
        json_config = json.load(f)
    if 'mouse_weight_before' in json_config:
        subject_metadata['weight'] = json_config['mouse_weight_before']
    else:
        subject_metadata['weight'] = 'na'


    # Session metadata.
    # #################

    # session data
    session_metadata = {
        'identifier': session_id,  # key to name the NWB file
        'session_id': session_id,
        'session_start_time': session_id.split('_')[1] + ' ' + session_id.split('_')[2],
        'session_description': session_description,
        'experimenter': subject_id[:2],
        'institution': 'EPFL',
        'lab': 'LSENS',
        'experiment_description': 'na',
        'keywords': [],
        'notes': 'na',
        'pharmacology': 'na',
        'protocol': 'na',
        'related_publications': 'na',
        'source_script': 'na',
        'source_script_file_name': 'na',
        'surgery': 'na',
        'virus': 'na',
        'stimulus_notes': 'na',
        'slices': 'na',
        }


    # Log continuous metadata.
    # ########################

    context_channel_date = "20230524"  # one day before first session with added channel odd number // 24 even
    context_channel_date = datetime.datetime.strptime(context_channel_date, "%Y%m%d")

    scanimage_dict = {
        'theoretical_ci_sampling_rate': 30,
        'zoom': 3
        }

    if session_date <= context_channel_date:
        channels_dict = {
            'trial_TTL': 2,
            'lick_trace': 0,
            'galvo_position': 1,
            'cam1': 3,
            'cam2': 4,
            }

        threshold_dict = {
            'trial_TTL': 4,
            'cam1': 2,
            'cam2': 2,
            'galvo_position': {
                '1': 2,
                '2': 1.2,
                '2.5': 1.3,
                '3': 0.9,
            },
        }
    else:
        channels_dict = {
            'trial_TTL': 2,
            'lick_trace': 0,
            'galvo_position': 1,
            'cam1': 3,
            'cam2': 4,
            'context': 5
        }
        threshold_dict = {
            'trial_TTL': 4,
            'cam1': 2,
            'cam2': 2,
            'galvo_position': {
                '1': 2,
                '2': 1.2,
                '2.5': 1.3,
                '3': 0.9,
                },
            'context': 4
            }

    log_continuous_metadata = {
        'scanimage_dict': scanimage_dict,
        'channels_dict': channels_dict,
        'threshold_dict': threshold_dict,
        }


    # Trial outcome mapping.
    # ######################

    trial_map = {
        0: 'whisker_miss',
        1: 'auditory_miss',
        2: 'whisker_hit',
        3: 'auditory_hit',
        4: 'correct_rejection',
        5: 'false_alarm',
        6: 'early_lick',
        }


    # 2P imaging metadata.
    # ####################

    two_photon_metadata = {
        'device': '2P microscope setup 1',
        'emission_lambda': 510.0,
        'excitation_lambda': 940.0,
        'image_plane_location': 'S1_L2/3',
        'indicator': 'GCaMP8m',
    }


    # Write to yaml file.
    # ###################

    main_dict = {
        'subject_metadata': subject_metadata,
        'session_metadata': session_metadata,
        'log_continuous_metadata': log_continuous_metadata,
        'trial_map': trial_map,
        '2P_metadata': two_photon_metadata,
    }

    analysis_session_folder = os.path.join(output_folder, session_id)
    if not os.path.exists(analysis_session_folder):
        os.makedirs(analysis_session_folder)
    with open(os.path.join(analysis_session_folder, f"config_{session_id}.yaml"), 'w') as stream:
        yaml.dump(main_dict, stream, default_flow_style=False, explicit_start=True)


if __name__ == '__main__':
    # mouse_ids = ['RD001', 'RD002', 'RD003', 'RD004', 'RD005', 'RD006']
    # mouse_ids = ['RD001', 'RD003', 'RD005']
    # mouse_ids = ['RD002', 'RD004', 'RD006']
    # mouse_ids = ['RD002', 'RD004']
    mouse_ids = ['AB077']
    #last_done_day = "20230601"

    for mouse_id in mouse_ids:

        # Find data and analysis folders on server for that mouse.
        data_folder = get_subject_data_folder(mouse_id)
        analysis_folder = get_subject_analysis_folder(mouse_id)
        if not os.path.exists(analysis_folder):
            os.makedirs(analysis_folder)

        # Make config files.
        training_days = find_training_days(mouse_id, data_folder)
        for session_id, day in training_days:
            session_date = session_id.split('_')[1]
            session_date = datetime.datetime.strptime(session_date, "%Y%m%d")
            # if session_date <= datetime.datetime.strptime(last_done_day, "%Y%m%d"):
            #     continue
            #sessions_to_do = ["PB124_20230404_141456"]
            #if session_id not in sessions_to_do:
            #    continue
            #else:
            make_yaml_config(mouse_id, session_id, day, data_folder, analysis_folder,
                                mouse_line='C57BL/6', gmo=False)
