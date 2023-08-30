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
    slims = slims.loc[slims.cntn_cf_mouseName == mouse_id]

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
    # Whether mouse is WT in not in the default Slims metadata, so use 'gmo' parameter.
    # If you want to check if the mouse has the mutation with Slims, export the mutation column.
    if gmo:
        subject_metadata['genotype'] = slims['cntn_cf_strain'].values[0]
    else:
        subject_metadata['genotype'] = 'WT'

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

    log_continuous_metadata = {}

    # Add logged channels and thresholds (Volt) for edge detections.
    channels_dict, threshold_dict = create_channels_threshold_dict(experimenter=subject_id[:2],
                                                                   json_config=json_config)
    if json_config['twophoton_session']==1:
        scanimage_dict = {
            'theoretical_ci_sampling_rate': 30,
            'zoom': 3
        }
        log_continuous_metadata.update({'scanimage_dict': scanimage_dict})


    # Add to general dictionary.
    log_continuous_metadata.update({'channels_dict': channels_dict})
    log_continuous_metadata.update({'threshold_dict': threshold_dict})

    # Behaviour metadata. #TODO: this should also be experimenter-dependent and a function of the json config file.
    # ###################

    behaviour_metadata = {
        'path_to_config_file': json_path,
        'behaviour_type': json_config['behaviour_type'],
        'trial_table': 'simple', # for raw NWB trial data, 'full' or 'simple'
        'camera_flag': json_config['camera_flag'],
    }
    if 'camera_exposure_time' in json_config.keys():
        behaviour_metadata.update({'camera_exposure_time': json_config['camera_exposure_time']})

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

    # 2P imaging metadata. #TODO: make this from external experimenter-dependent excel file
    # ####################

    two_photon_metadata = {
        'device': '2P microscope setup 1',
        'emission_lambda': 510.0,
        'excitation_lambda': 940.0,
        'image_plane_location': 'S1_L2/3',
        'indicator': 'GCaMP8m',
    }

    # Extracell. ephys. metadata. #TODO: make this from external experimenter-dependent excel file
    # ####################

    ephys_metadata = {
        'device': 'Neuropixels setup 1 AI3209',
    }

    # Write to yaml file.
    # ###################

    main_dict = {
        'subject_metadata': subject_metadata,
        'session_metadata': session_metadata,
        'log_continuous_metadata': log_continuous_metadata,
        'behaviour_metadata': behaviour_metadata,
        'trial_map': trial_map,
    }

    # Depending on session type, add relevant dictionary
    if json_config['twophoton_session']:
        main_dict.update({'two_photon_metadata': two_photon_metadata})

    elif json_config['ephys_session']:
        main_dict.update({'ephys_metadata': ephys_metadata})

    main_dict.update({'behaviour_metadata': behaviour_metadata})

    analysis_session_folder = os.path.join(output_folder, session_id)
    if not os.path.exists(analysis_session_folder):
        os.makedirs(analysis_session_folder)
    with open(os.path.join(analysis_session_folder, f"config_{session_id}.yaml"), 'w') as stream:
        yaml.dump(main_dict, stream, default_flow_style=False, explicit_start=True)

    return

def create_channels_threshold_dict(experimenter, json_config):
    """
    Make log_continuous channels & thresholds dictionary for a given experimenter and session.
    Args:
        experimenter: experimenter initials
        json_config: session config dictionary from session_config.json file
    Returns:

    """
    if experimenter in ['AB']:
        lick_threshold = json_config['lick_threshold']
        channels_dict = {
            'trial_TTL': 2,
            'lick_trace': 0,
            'cam1': 3,
            'cam2': 4,
            'empty_1': 1,
            'empty_2': 6
        }
        threshold_dict = {
            'trial_TTL': 4,
            'lick_trace': lick_threshold,
            'cam1': 2,
            'cam2': 2,
            'empty_1': 0,
            'empty_2': 0
        }

    elif experimenter in ['RD', 'AR'] or json_config['mouse_name']=='PB124':
        channels_dict = {
            'trial_TTL': 2,
            'lick_trace': 0,
            'galvo_position': 1,
            'cam1': 3,
            'cam2': 4,
            #'context': 5,
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

        context_channel_date = "20230524"  # one day before first session with added channel odd number // 24 even
        context_channel_date = datetime.datetime.strptime(context_channel_date, "%Y%m%d")
        session_date = datetime.datetime.strptime(json_config['date'], "%Y%m%d")
        if session_date > context_channel_date:
            channels_dict.update({'context': 5})
            threshold_dict.update({'context': 4})



    # elif experimenter in ['PB'] and json_config['mouse_name']!='PB124':
    # ...

    return channels_dict, threshold_dict


if __name__ == '__main__':
    # Select mouse IDs.
    # mouse_ids = ['RD001', 'RD002', 'RD003', 'RD004', 'RD005', 'RD006']
    mouse_ids = [50,51,52,54,56,58,59,68,72,73,75,76,77,78,79,80,81,82,83]
    mouse_ids = ['AB0{}'.format(i) for i in mouse_ids]
    mouse_ids = ['RD004']
    # last_done_day = "20230601"

    for mouse_id in mouse_ids:

        # Find data and analysis folders on server for that mouse.
        data_folder = get_subject_data_folder(mouse_id)
        analysis_folder = get_subject_analysis_folder(mouse_id)
        analysis_folder = analysis_folder.replace('Robin_Dard', 'Axel_Bisi') #TODO: delete
        if not os.path.exists(analysis_folder):
            os.makedirs(analysis_folder)

        # Make config files.
        training_days = find_training_days(mouse_id, data_folder)
        for session_id, day in training_days:
            session_date = session_id.split('_')[1]
            session_date = datetime.datetime.strptime(session_date, "%Y%m%d")
            # if session_date <= datetime.datetime.strptime(last_done_day, "%Y%m%d"):
            #     continue
            # sessions_to_do = ["PB124_20230404_141456"]
            # if session_id not in sessions_to_do:
            #    continue
            # else:
            make_yaml_config(mouse_id, session_id, day, data_folder, analysis_folder,
                             mouse_line='C57BL/6', gmo=False)
