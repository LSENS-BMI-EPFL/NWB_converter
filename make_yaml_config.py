"""_summary_
"""
import datetime
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml

from utils.behavior_converter_misc import find_training_days
from utils.server_paths import (
    get_ref_weight_folder,
    get_subject_analysis_folder,
    get_subject_data_folder,
    get_subject_mouse_number, 
    get_experimenter_analysis_folder,
    get_analysis_root,
    EXPERIMENTER_MAP,
    )
from metadata_to_yaml import add_metadata_to_config

# Update your keywords
GENERAL_KEYWORDS = ['neurophysiology', 'behaviour', 'mouse']
KEYWORD_MAP = {
    'AR': ['optogenetics', 'widefield', 'two_photon', 'calcium_imaging', 'barrel_cortex'],
    'RD': [''],
    'AB': ['electrophysiology', 'neuropixels'],
    'MH': ['electrophysiology', 'neuropixels'],
    'MP': [''],
    'PB': ['behaviour', 'optogenetics', 'widefield', 'two_photon'],
    'MM': [''],
    'LS': [''],
    'GF': ['optogenetics', 'widefield', 'two_photon', 'calcium_imaging', 'barrel_cortex'],
    'MI': ['optogenetics', 'widefield', 'two_photon', 'calcium_imaging', 'barrel_cortex'],
    'JL': ['electrophysiology', 'neuropixels'],
}


def make_yaml_config(subject_id, session_id, session_description, input_folder, output_folder,
                     mouse_line='C57BL/6', gmo=True, experimenter=None):
    """_summary_

    Args:
        subject_id (_type_): _description_
        session_id (_type_): _description_
        session_description (_type_): _description_
        mouse_line (str, optional): _description_. Defaults to 'C57BL/6'.
        gmo (bool, optional): _description_. Defaults to True.
    """

    # Subject metadata.
    # #################

    # Get mouse number and experimenter initials from subject ID.
    if experimenter is None:
        _, experimenter = get_subject_mouse_number(subject_id)

    # Select most recent metadata export from SLIMS folder.
    try:
        slims_csv = sorted(os.listdir(os.path.join(input_folder, 'SLIMS')))[
            0]  # post-euthanasia SLIMS file has more information
        if experimenter == 'JL':
            slims_csv = sorted(os.listdir(os.path.join(input_folder, 'SLIMS')))[-1]
        slims_csv_path = os.path.join(input_folder, 'SLIMS', slims_csv)
        slims = pd.read_csv(slims_csv_path, sep=';', engine='python')
    except IndexError:
        print('Error: SLIMS folder may be empty. Export SLIMS info .csv file.')
        return
    except UnicodeDecodeError:
        print('Error: SLIMS file may not be in a .csv file. Please export it again from SLIMS as .csv.')
        return
    except FileNotFoundError:
        print('Error: SLIMS file not found. Export SLIMS info .csv file.')
        return

    slims = slims.loc[slims.cntn_cf_mouseName == subject_id]

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
    sess_folder = os.path.join(input_folder, 'Training', session_id)
    session_config_file = 'session_config_corrected.json' if 'session_config_corrected.json' in os.listdir(sess_folder) else 'session_config.json' # seek any corrected file if it exists, default to GUI output otherwise
    session_config_json_path = os.path.join(sess_folder, session_config_file)
    with open(session_config_json_path, 'r') as f:
        json_config = json.load(f)

    # Get mouse session weight
    if 'mouse_weight_before' in json_config:
        subject_metadata['weight'] = json_config['mouse_weight_before']
    else:
        subject_metadata['weight'] = 'na'

    # Get mouse reference weight
    if subject_id in ['PB{}'.format(i) for i in range(191, 202)]:
        ref_weight_path = get_ref_weight_folder('AB')
    else:
        ref_weight_path = get_ref_weight_folder(experimenter=experimenter)
    ref_weight_csv_path = os.path.join(ref_weight_path, 'mouse_reference_weight.xlsx')
    if not os.path.exists(ref_weight_csv_path):
        print(f'Error: reference weight file not found for {experimenter}. Please create it in analysis_folder/mice_info.')
        ref_weight = np.nan
    else:
        ref_weight_df = pd.read_excel(ref_weight_csv_path)
        # Make sure subject is in the reference weight file
        if subject_id not in ref_weight_df['mouse_id'].values:
            print(f'Error: subject {subject_id} not found in reference weight file for {subject_id}. Please add it.')
            ref_weight = np.nan
        else:
            ref_weight_cols = [col for col in ref_weight_df.columns if 'weight' in col]
            if len(ref_weight_cols) > 1:
                print(
                    f'NotImplementedError: more than one reference weight column found for {experimenter}. Please check.')
                ref_weight = np.nan
            else:
                ref_weight = ref_weight_df.loc[ref_weight_df['mouse_id'] == subject_id, ref_weight_cols[0]].values[0]
                assert isinstance(ref_weight,
                                  float), f'Error: reference weight for {subject_id} is not a float. Please check.'

    # Generating session metadata dictionary as experimenter_description #TODO: modularize
    session_type_flags = [sess_key_flag for sess_key_flag in json_config.keys() if 'session' in sess_key_flag]
    session_type_flags.remove('session_time')
    session_type_flags.remove('dummy_session_flag')
    session_type_ticked = [sess_key_flag for sess_key_flag in session_type_flags if json_config[sess_key_flag] == True]
    if session_type_ticked:
        session_type_prefixes = [sess_key_flag.split('_')[0] for sess_key_flag in session_type_ticked]
        session_type_prefixes.append('session')
        session_type = '_'.join(session_type_prefixes)  # e.g. 'twophoton_session', or 'wf_opto_session', etc.
    else:
        session_type = 'behaviour_only_session'

    session_experiment_metadata = {
        "reference_weight": str(ref_weight),  # reference weight before water-restriction
        "session_type": session_type,
        "wh_reward": json_config['wh_reward'],
        "aud_reward": json_config['aud_reward'],
        "reward_proba": json_config['reward_proba'],
        "lick_threshold": json_config['lick_threshold'],
        "no_stim_weight": json_config['no_stim_weight'],
        "wh_stim_weight": sum([v for k, v in json_config.items() if 'wh_stim_weight' in k]),
        "aud_stim_weight": sum([v for k, v in json_config.items() if 'aud_stim_weight' in k]),
        "camera_flag": json_config['camera_flag'],
        "camera_freq": json_config['camera_freq'],
        "camera_exposure_time":
            [json_config['camera_exposure_time'] if 'camera_exposure_time' in json_config.keys() else 'na'][0],
        "camera_start_delay":
            [json_config['camera_start_delay'] if 'camera_start_delay' in json_config.keys() else 'na'][0],
        "artifact_window": json_config['artifact_window'],

    }

    # Session metadata.
    # #################

    # session data
    session_metadata = {
        'identifier': session_id,  # key to name the NWB file
        'session_id': session_id,
        'session_start_time': session_id.split('_')[1] + ' ' + session_id.split('_')[2],
        'session_description': session_description,
        'experimenter': experimenter,
        'institution': 'Ecole Polytechnique Federale de Lausanne',
        'lab': 'Laboratory of Sensory Processing',
        'experiment_description': str(session_experiment_metadata),
        'keywords': GENERAL_KEYWORDS + KEYWORD_MAP[experimenter],
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
    channels_dict, threshold_dict = create_channels_threshold_dict(experimenter=experimenter,
                                                                   json_config=json_config,
                                                                   session_type=session_type)
    if json_config['twophoton_session'] == 1:
        scanimage_dict = {
            'theoretical_ci_sampling_rate': 30,
            'zoom': 3
        }
        log_continuous_metadata.update({'scanimage_dict': scanimage_dict})

    # Add to general dictionary.
    log_continuous_metadata.update({'channels_dict': channels_dict})
    log_continuous_metadata.update({'threshold_dict': threshold_dict})

    # Behaviour metadata.
    # ###################

    behaviour_metadata = create_behaviour_metadata(experimenter=experimenter,
                                                   path_to_json_config=session_config_json_path)

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
    if experimenter in ['AB', 'MH']:
        trial_map[6] = 'association'

    # 2P imaging metadata.
    # ####################

    two_photon_metadata = create_two_photon_metadata(experimenter)

    # Extracell. ephys. metadata.
    # ####################
    if experimenter in ['AB', 'PB', 'MH', 'RD', 'JL']:
        ephys_metadata = create_ephys_metadata(subject_id=subject_id, experimenter=experimenter, session_date=session_date)

    # Write to yaml file.
    # ###################

    main_dict = {
        'subject_metadata': subject_metadata,
        'session_metadata': session_metadata,
        'log_continuous_metadata': log_continuous_metadata,
        'behaviour_metadata': behaviour_metadata,
        'trial_map': trial_map,
    }

    # Depending on session type, add relevant metadata.
    # #################################################

    if json_config['twophoton_session']:
        main_dict.update({'two_photon_metadata': two_photon_metadata})

    elif json_config['ephys_session'] and ephys_metadata['processed'] == 1:
        main_dict.update({'ephys_metadata': ephys_metadata})

    elif json_config['wf_session']:
        widefield_metadata = create_wf_metadata(config_path=os.path.join(input_folder, 'Training', session_id))
        main_dict.update({'widefield_metadata': widefield_metadata})


    main_dict.update({'behaviour_metadata': behaviour_metadata})

    analysis_session_folder = os.path.join(output_folder, session_id)
    if not os.path.exists(analysis_session_folder):
        os.makedirs(analysis_session_folder)
    with open(os.path.join(analysis_session_folder, f"config_{session_id}.yaml"), 'w') as stream:
        yaml.dump(main_dict, stream, default_flow_style=False, explicit_start=True)

    return


def create_channels_threshold_dict(experimenter, json_config, session_type):
    """
    Make log_continuous channels & thresholds dictionary for a given experimenter and session.
    This relates to logging by the behavioural control software.
    Args:
        experimenter: experimenter initials
        json_config: session config dictionary from session_config.json file
        session_type: session type (helpful to know if it is behavior only)
    Returns:

    """
    channels_dict, threshold_dict = {}, {}

    if experimenter in ['AB'] or json_config['ephys_session']: # note: this differs from SpikeGLX ephys setup
        lick_threshold = json_config['lick_threshold']
        channels_dict = {
            'trial_TTL': 2,
            'lick_trace': 0,
            'cam1': 3,
            'cam2': 4,
            'empty_1': 1,
            #'empty_2': 6
        }
        threshold_dict = {
            'trial_TTL': 4,
            'lick_trace': lick_threshold,
            'cam1': 2,
            'cam2': 2,
            'empty_1': 0,
            #'empty_2': 0
        }

        #if json_config['mouse_name'] == 'AB068':  # before context logging
        #    channels_dict.pop('empty_2')
        #    threshold_dict.pop('empty_2')

    elif json_config['twophoton_session'] or (session_type in ['pharma_session', 'behaviour_only_session']):
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

    elif json_config['wf_session'] or (session_type in ['opto_session', 'wf_opto_session']):
        channels_dict = {
            'trial_TTL': 2,
            'lick_trace': 0,
            'widefield': 1,
            'cam1': 3,
            'cam2': 4,
        }
        threshold_dict = {
            'trial_TTL': 4,
            'cam1': 2,
            'cam2': 2,
            'widefield': 2
        }

    # Add context channel and threshold
    context_channel_date = "20230524"  # one day before first session with added channel odd number // 24 even
    context_channel_date = datetime.datetime.strptime(context_channel_date, "%Y%m%d")
    session_date = datetime.datetime.strptime(json_config['date'], "%Y%m%d")
    if session_date > context_channel_date:
        channels_dict.update({'context': 5})
        threshold_dict.update({'context': 4})


    # elif experimenter in ['PB'] and json_config['mouse_name']!='PB124':
    # ...

    return channels_dict, threshold_dict


def create_behaviour_metadata(experimenter, path_to_json_config):
    """
    Make behaviour metadata dictionary.
    Args:
        experimenter: experimenter initials
        path_to_json_config:

    Returns:

    """

    with open(path_to_json_config, 'r') as f:
        json_config = json.load(f)

    # Default behaviour metadata
    if json_config['behaviour_type'] == 'whisker_off':
        json_config['behaviour_type'] = 'whisker_off_1' # ensures correct parsing later on

    behaviour_metadata = {
        'path_to_config_file': path_to_json_config,
        'behaviour_type': json_config['behaviour_type'],
        'trial_table': 'standard',  # for raw NWB trial data, 'standard', 'simple'
        'camera_flag': json_config['camera_flag'],
    }
    # Add camera exposure time if present in json config (was not logged before a certain date)
    if 'camera_exposure_time' in json_config.keys():
        behaviour_metadata.update({'camera_exposure_time': json_config['camera_exposure_time']})

    # Experimenter specific behaviour metadata - change setup per mouse
    if experimenter in ['AB']:
        behaviour_metadata.update({'trial_table': 'standard',
                                  'setup': 'Neuropixels setup 1 AI3209'})
    elif experimenter in ['MH']:
        behaviour_metadata.update({'trial_table': 'standard',
                                  'setup': 'Neuropixels setup 2 AI3209'})
    return behaviour_metadata


def create_ephys_metadata(subject_id, experimenter, session_date):
    """
    Make ephys metadata dictionary.
    Args:
        subject_id:

    Returns:

    """

    experimenter_full = EXPERIMENTER_MAP[experimenter]
    mouse_number, initials = get_subject_mouse_number(subject_id)
    mouse_initials = initials[:2]
    path_to_probe_info = Path(get_analysis_root()) / experimenter_full / 'mice_info' / 'probe_insertion_info.xlsx'
    df_probe_info = pd.read_excel(path_to_probe_info)
    mouse_rows = df_probe_info[df_probe_info['mouse_name'] == subject_id]
    mouse_rows = df_probe_info[df_probe_info['mouse_name'] == subject_id]

    if 'date' in df_probe_info.columns and mouse_rows['date'].notnull().any():
        # Ensure 'date' is datetime
        df_probe_info['date'] = pd.to_datetime(df_probe_info['date'], errors='coerce')
        setup = df_probe_info.loc[
            (df_probe_info['date'].dt.date == session_date.date()) & (df_probe_info['mouse_name'] == subject_id), 'Setup'
        ]
        setup = setup.iloc[0] if not setup.empty else None
    else:
        setup = df_probe_info.loc[df_probe_info['mouse_name'] == subject_id, 'Setup']
        setup = setup.iloc[0] if not setup.empty else None

    if setup is None:
        warnings.warn(f"Setup information not found for subject {subject_id} on date {session_date.strftime('%Y-%m-%d')}.")

    setup = f'Neuropixels {setup}'


    path_to_atlas_dict = {
        'AB': r'C:\Users\bisi\.brainglobe\allen_mouse_bluebrain_barrels_10um_v1.0',
    }

    path_to_atlas = path_to_atlas_dict.get(experimenter, None)
    if path_to_atlas is None:
        warnings.warn(f'No path to atlas set for {experimenter}')

    # Set SpikeGLX-NIDQ channel mapping, experiment-dependent
    # Note: if the setup is changed, the channel mapping should be updated
    # TODO: use it later on in preprocessing
    ephys_channels_dict_map = {
        'Neuropixels setup 1 AI3209': {
            'default': {
                0: 'sync',
                1: 'trial_TTL',
                2: 'auditory_stim',
                3: 'whisker_stim',
                4: 'valve',
                5: 'cam1',
                6: 'cam2',
                7: 'lick_trace'
            },
            'context': {
                0: 'sync',
                1: 'trial_TTL',
                2: 'auditory_stim',
                3: 'whisker_stim',
                4: 'context_transition',
                5: 'cam1',
                6: 'cam2',
                7: 'lick_trace'
            },
        },
        'Neuropixels setup 2 AI3209': {
            'default':{
                0: 'sync',
                1: 'trial_TTL',
                2: 'whisker_stim',
                3: 'lick_trace',
                4: 'valve',
                5: 'cam1',
                6: 'cam2',
                7: 'auditory_stim'
            },
            'context':{
                0: 'sync',
                1: 'trial_TTL',
                2: 'whisker_stim',
                3: 'lick_trace',
                4: 'context_transition',
                5: 'cam1',
                6: 'cam2',
                7: 'auditory_stim'
            },
        },
    }
    channel_map = 'default'
    if initials in ['JL', 'PB', 'RD']:
        channel_map = 'context'

    ephys_channels_dict = ephys_channels_dict_map[setup][channel_map]

    # Set which mice have processed neural data
    if mouse_initials == 'AB' and int(mouse_number) not in [159, 163, 164]:
        processed = 0
    elif subject_id in ['AB077', 'AB135', 'AB137']:
        processed = 0
    elif mouse_initials == 'PB':
        processed = 0
    else:
        processed = 1

    ephys_metadata = {
        'setup': setup,
        'ephys_channels_dict': ephys_channels_dict,
        'unit_table': 'standard',  # 'simple' or 'standard'
        'processed': processed, # 0 or 1
        'path_to_probe_info': str(path_to_probe_info),
        'path_to_atlas': path_to_atlas,
    }
    return ephys_metadata


def create_two_photon_metadata(experimenter):
    # TODO make it dependant on excel file for imaging params
    if experimenter == 'AR':
        setup_id = 1
    else:
        setup_id = 3
    two_photon_metadata = {
        'device': f'2P microscope setup {setup_id}',
        'emission_lambda': 510.0,
        'excitation_lambda': 940.0,
        'image_plane_location': 'S1_L2/3',
        'indicator': 'GCaMP8m',
    }
    return two_photon_metadata


def create_wf_metadata(config_path):
    """Create metadata structure specific for widefield imaging
    Args:
        subject_id

    Returns:
        """

    wf_config_json_path = os.path.join(config_path, 'wf_config.json')

    with open(wf_config_json_path, 'r') as f:
        widefield_metadata = json.load(f)

    for item in ['CameraRoot', 'CameraPathConfig', 'CameraPathTemplateConfig', 'n_frames_to_grab', 'savedir']:
        del(widefield_metadata[item])

    return widefield_metadata


if __name__ == '__main__':
    # Select mouse IDs.
    experimenter = 'JL'
    experimenter_full = 'Jules_Lebert'
    mouse_ids = [
        'PB191', 
        'JL007',
    ]
    sessions_to_do = [
        'JL007_20250603_150143',
        'PB191_20241210_110601',
    ]
    #last_done_day = '20241210'

    for mouse_id in mouse_ids:

        # Find data and analysis folders on server for that mouse.
        data_folder = get_subject_data_folder(mouse_id)

        if os.path.exists(data_folder):
            pass
        else:
            print(f"No mouse data folder for {mouse_id}.")
            continue
        analysis_folder = get_subject_analysis_folder(mouse_id, experimenter=experimenter_full)

        # Make config files.
        training_days = find_training_days(mouse_id, data_folder)

        for session_id, day in training_days:

            session_date = session_id.split('_')[1]
            session_date = datetime.datetime.strptime(session_date, "%Y%m%d")
            #if last_done_day is not None:
            #     if session_date <= datetime.datetime.strptime(last_done_day, "%Y%m%d"):
            #         continue#

            if session_id not in sessions_to_do:
                continue

            if experimenter == 'AB' and day != 'whisker_0':
                continue

            #if experimenter == 'AB' and 'whisker' not in day:
            #    continue


            make_yaml_config(mouse_id, session_id, day, data_folder, analysis_folder,
                             mouse_line='C57BL/6', gmo=False, experimenter=experimenter)

            #add_metadata_to_config(mouse_id, session_id, experimenter)

