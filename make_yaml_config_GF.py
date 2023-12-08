"""_summary_
"""
import datetime
import json
import os

import numpy as np
import pandas as pd
import yaml

import utils.gf_utils as utils_gf
from utils.server_paths import (get_ref_weight_folder,
                                get_subject_analysis_folder,
                                get_subject_data_folder,
                                get_subject_mouse_number)

# Update your keywords
GENERAL_KEYWORDS = ['neurophysiology', 'behaviour', 'mouse']
KEYWORD_MAP = {
    'AR': ['optogenetics', 'widefield', 'two_photon', 'calcium_imaging', 'barrel_cortex'],
    'RD': [''],
    'AB': ['electrophysiology', 'neuropixels'],
    'MP': [''],
    'PB': ['behaviour', 'optogenetics', 'widefield', 'two_photon'],
    'MM': [''],
    'LS': [''],
    'GF': ['optogenetics', 'widefield', 'two_photon', 'calcium_imaging', 'barrel_cortex'],
    'MI': ['optogenetics', 'widefield', 'two_photon', 'calcium_imaging', 'barrel_cortex']
}


def make_yaml_config_GF(subject_id, session_id, session_description, input_folder, output_folder,
                     database, mouse_line='C57BL/6', gmo=True):
    """_summary_

    Args:
        subject_id (_type_): _description_
        session_id (_type_): _description_
        session_description (_type_): _description_
        mouse_line (str, optional): _description_. Defaults to 'C57BL/6'.
        gmo (bool, optional): _description_. Defaults to True.
    """

    print(f'Creating yaml config file for session {session_id}.')

    # Subject metadata.
    # #################

    # Get mouse number and experimenter initials from subject ID.
    _, experimenter = get_subject_mouse_number(subject_id)

    if experimenter in ['GF']:
        slims_csv_path = '\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab\\analysis\\Georgios_Foustoukos\\FoustoukosData\\MetaData\\MiceMetaData.csv'
        slims = pd.read_csv(slims_csv_path, sep=';', engine='python')
    else:
        # Select most recent metadata export from SLIMS folder.
        try:
            slims_csv = sorted(os.listdir(os.path.join(input_folder, 'SLIMS')))[-1]  # post-euthanasia SLIMS file has more information
            slims_csv_path = os.path.join(input_folder, 'SLIMS', slims_csv)
            slims = pd.read_csv(slims_csv_path, sep=';', engine='python')
        except IndexError:
            print('Error: SLIMS folder may be empty. Export SLIMS info .csv file.', end='\r')
            return
        except UnicodeDecodeError:
            print('Error: SLIMS file may not be in a .csv file. Please export it again from SLIMS as .csv.', end='\r')
            return
        except FileNotFoundError:
            print('Error: SLIMS file not found. Export SLIMS info .csv file.', end='\r')
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
    if len(session_date) > 8:
        session_date = session_date[:8]
    session_date = datetime.datetime.strptime(session_date, "%d%m%Y")
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

    # # Add weight at the beginning of the session from json config file.
    # session_config_json_path = os.path.join(input_folder, 'Training', session_id, 'session_config.json')
    # with open(session_config_json_path, 'r') as f:
    #     json_config = json.load(f)

    # # Get mouse session weight
    # if 'mouse_weight_before' in json_config:
    #     subject_metadata['weight'] = json_config['mouse_weight_before']
    # else:
    #     subject_metadata['weight'] = 'na'

    subject_metadata['weight'] = 'na'

    # Get mouse reference weight
    ref_weight_path = get_ref_weight_folder(experimenter=experimenter)
    ref_weight_csv_path = os.path.join(ref_weight_path, 'mouse_reference_weight.xlsx')
    if not os.path.exists(ref_weight_csv_path):
        print(f'Error: reference weight file not found for {experimenter}. Please create it.')
        ref_weight = np.nan
    else:
        ref_weight_df = pd.read_excel(ref_weight_csv_path)
        # Make sure subject is in the reference weight file
        if subject_id not in ref_weight_df.mouse_name.values:
            print(f'Error: subject {subject_id} not found in reference weight file for {subject_id}. Please add it.')
            ref_weight = np.nan
        else:
            ref_weight_cols = [col for col in ref_weight_df.columns if 'weight' in col]
            if len(ref_weight_cols) > 1:
                print(
                    f'NotImplementedError: more than one reference weight column found for {experimenter}. Please check.')
                ref_weight = np.nan
            else:
                ref_weight = ref_weight_df.loc[ref_weight_df.mouse_name == subject_id, ref_weight_cols[0]].values[0]
                assert isinstance(ref_weight,
                                  float), f'Error: reference weight for {subject_id} is not a float. Please check.'

    # # Generating session metadata dictionary as experimenter_description
    # session_type_flags = [sess_key_flag for sess_key_flag in json_config.keys() if 'session' in sess_key_flag]
    # session_type_flags.remove('session_time')
    # session_type_flags.remove('dummy_session_flag')
    # session_type_ticked = [sess_key_flag for sess_key_flag in session_type_flags if json_config[sess_key_flag] == True]
    # if session_type_ticked:
    #     session_type_prefixes = [sess_key_flag.split('_')[0] for sess_key_flag in session_type_ticked]
    #     session_type_prefixes.append('session')
    #     session_type = '_'.join(session_type_prefixes)  # e.g. 'twophoton_session', or 'wf_opto_session', etc.
    # else:
    #     session_type = 'behaviour_only_session'

    # Get session_type from database.
    if database.loc[database.session_id==session_id, '2P_calcium_imaging'].values[0]:
        session_type = 'twophoton_session'
    elif database.loc[database.session_id==session_id, 'optogenetic'].values[0]:
        session_type = 'opto_session'
    elif database.loc[database.session_id==session_id, 'pharmacology'].values[0]:
        session_type = 'pharma_session'
    else:
        session_type = 'behaviour_only_session'

    # Read json performance file.
    perf_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Anthony_Renard',
                             'data', subject_id, 'Recordings', 'BehaviourData', session_id, 'performanceResults.json')
    with open(perf_path, 'r') as f:
        perf_json = json.load(f)
    perf_df = pd.DataFrame(perf_json['results'], columns=perf_json['headers'])

    # Check if R+ or R- mouse.
    if (perf_df.whrew==1).sum() > 0:
        wh_reward = 1
    else:
        wh_reward = 0

    # Infer stimuli proportions from session day.
    session_day = database.loc[database.session_id==session_id, 'session_day'].values[0]
    if '-' in session_day:  # Audiotry session, otherwise there are whisker trials.
        wh_stim_weight = 0
        aud_stim_weight = 10
    else:
        wh_stim_weight = 7
        aud_stim_weight = 3

    filming_path = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Anthony_Renard',
                                'data', subject_id, 'Recordings', 'FilmingData')
    if os.path.exists(filming_path):
        camera_flag = 1
    else:
        camera_flag = 0

    session_experiment_metadata = {
        'reference_weight': 'na',  # reference weight before water-restriction
        'session_type': session_type,
        'wh_reward': wh_reward,
        'aud_reward': 1,
        'reward_proba': 1,
        'lick_threshold': 'na',
        'no_stim_weight': 10,
        'wh_stim_weight': wh_stim_weight,
        'aud_stim_weight': aud_stim_weight,
        'camera_flag': camera_flag,
        'camera_freq': 100,
        'camera_exposure_time': 2,
        'camera_start_delay': 'na',
        'artifact_window': 100,
    }

    # Session metadata.
    # #################

    start_date = session_id.split('_')[1]
    start_date = datetime.datetime.strptime(start_date, '%d%m%Y').strftime('%Y%m%d')
    start_time = session_id.split('_')[2]

    # Find is there is pharmacolyg, optogentic or chemogenetic.
    if database.loc[database.session_id==session_id, 'pharmacology'].values[0]:
        pharma_day = database.loc[database.session_id==session_id, 'pharma_day'].values[0]
        pharma_inactivation = database.loc[database.session_id==session_id, 'pharma_inactivation_type'].values[0]
        pharma_area = database.loc[database.session_id==session_id, 'pharma_area'].values[0]
        pharma = {'pharma_day': pharma_day,
                  'pharma_inactivation': pharma_inactivation,
                  'pharma_area': pharma_area}
    else:
        pharma = 'na'
    if database.loc[database.session_id==session_id, 'optogenetic'].values[0]:
        opto_day = database.loc[database.session_id==session_id, 'opto_day'].values[0]
        opto_inactivation = database.loc[database.session_id==session_id, 'opto_inactivation_type'].values[0]
        opto_area = database.loc[database.session_id==session_id, 'opto_area'].values[0]
        opto = {'opto_day': opto_day,
                  'opto_inactivation': opto_inactivation,
                  'opto_area': opto_area}
    else:
        opto = 'na'

    # session data

    session_metadata = {
        'identifier': session_id,  # key to name the NWB file
        'session_id': session_id,
        'session_start_time': f'{start_date} {start_time}',
        'session_description': session_description,
        'experimenter': experimenter,
        'institution': 'Ecole Polytechnique Federale de Lausanne',
        'lab': 'Laboratory of Sensory Processing',
        'experiment_description': str(session_experiment_metadata),
        'keywords': GENERAL_KEYWORDS + KEYWORD_MAP[experimenter],
        'notes': 'na',
        'pharmacology': str(pharma),
        'optogenetic': str(opto),
        'chemogenetic': 'na',
        'protocol': 'na',
        'related_publications': 'na',
        'source_script': 'na',
        'source_script_file_name': 'na',
        'surgery': 'na',
        'virus': 'na',
        'stimulus_notes': 'na',
        'slices': 'na',
    }


    # # Log continuous metadata.
    # # ########################

    # log_continuous_metadata = {}

    # # Add logged channels and thresholds (Volt) for edge detections.
    # channels_dict, threshold_dict = create_channels_threshold_dict(experimenter=experimenter,
    #                                                                json_config=json_config)
    # if json_config['twophoton_session'] == 1:
    #     scanimage_dict = {
    #         'theoretical_ci_sampling_rate': 30,
    #         'zoom': 3
    #     }
    #     log_continuous_metadata.update({'scanimage_dict': scanimage_dict})

    # # Add to general dictionary.
    # log_continuous_metadata.update({'channels_dict': channels_dict})
    # log_continuous_metadata.update({'threshold_dict': threshold_dict})


    # # Behaviour metadata. #TODO: this could also be experimenter-dependent and a function of the json config file.
    # # ###################

    # behaviour_metadata = create_behaviour_metadata(experimenter=experimenter,
    #                                                path_to_json_config=session_config_json_path)
    behaviour_metadata = {
        'path_to_config_file': 'na',
        'behaviour_type': 'na',
        'trial_table': 'standard',  # for raw NWB trial data, 'standard', 'simple'
        'camera_flag': 'na',
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
        'indicator': 'GCaMP6f',
    }


    # Optogenetic metadata.
    # #####################

    # Pharmacology metadata.
    # ######################



    # Write to yaml file.
    # ###################

    main_dict = {
        'subject_metadata': subject_metadata,
        'session_metadata': session_metadata,
        # 'log_continuous_metadata': log_continuous_metadata,
        'behaviour_metadata': behaviour_metadata,
        'trial_map': trial_map,
    }

    # Depending on session type, add relevant dictionary
    if 'twophoton' in session_type:
        main_dict.update({'two_photon_metadata': two_photon_metadata})

    analysis_session_folder = os.path.join(output_folder, session_id)
    if not os.path.exists(analysis_session_folder):
        os.makedirs(analysis_session_folder)
    with open(os.path.join(analysis_session_folder, f"config_{session_id}.yaml"), 'w') as stream:
        yaml.dump(main_dict, stream, default_flow_style=False, explicit_start=True)

    return


if __name__ == '__main__':

    # last_done_day = "20231102"
    last_done_day = None
    gmo = True

    # Read excel database.
    db_folder = 'C:\\Users\\aprenard\\recherches\\fast-learning\\docs'
    db_name = 'sessions_GF.xlsx'
    db = utils_gf.read_excel_database(db_folder, db_name)

    # Select mouse IDs.
    mouse_ids = db.subject_id.unique()

    for mouse_id in mouse_ids:
        # Data folder in GF analysis.
        data_folder = get_subject_data_folder(mouse_id)
        # AR analysis folder where yaml and NWB files will be stored.
        analysis_folder = get_subject_analysis_folder(mouse_id)

        # Find training day for that mouse.
        training_days = db.loc[db.subject_id==mouse_id, 'session_day'].to_list()
        training_days = utils_gf.format_session_day_GF(mouse_id, training_days)
        sessions = db.loc[db.subject_id==mouse_id, 'session_id'].to_list()

        for session_id, day in list(zip(sessions, training_days)):
            session_date = session_id.split('_')[1]
            # Some of GF sessions for particle test have a suffix letter.
            if len(session_date) > 8:
                session_date = session_date[:8]
            # Reorder date as YYYYMMDD.
            session_date = datetime.datetime.strptime(session_date, "%d%m%Y")

            if last_done_day is not None:
                if session_date <= datetime.datetime.strptime(last_done_day, "%Y%m%d"):
                    continue

            make_yaml_config_GF(mouse_id, session_id, day, data_folder, analysis_folder,
                             mouse_line='C57BL/6', gmo=gmo, database=db)
