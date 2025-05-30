import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

import utils.utils_gf as utils_gf
from utils import server_paths


def find_training_days(subject_id, input_folder):

    """
    Find days of behavioural training, excluding other test/dummy sessions.
    Make labels for each session based on the type of behaviour for NWB files and analysis.
    Args:
        subject_id:
        input_folder:

    Returns: list of tuples (session_id, behaviour_type)

    """
    print('Finding training days for subject {}:'.format(subject_id))
    sessions_list = os.listdir(os.path.join(input_folder, 'Training'))
    sessions_list = [s for s in sessions_list if os.path.isdir(os.path.join(input_folder, 'Training', s))]

    # Ordering in time with lexicographic ordering assumes %Y%m%d data format in session id
    sessions_list = sorted(sessions_list)

    # ------------------------------------------------
    # Get session type (e.g. auditory or whisker day)
    # ------------------------------------------------
    behavior_type = []
    sessions_to_remove = []
    for isession in sessions_list:
        if 'calibration' in isession:
            continue
        session_folder_path = os.path.join(input_folder, 'Training', isession)
        json_file = 'session_config_corrected.json' if 'session_config_corrected.json' in os.listdir(session_folder_path) else 'session_config.json' # seek any corrected file if it exists, default to GUI output otherwise
        json_path = os.path.join(session_folder_path, json_file)
        with open(json_path, 'r') as f:
            json_config = json.load(f)

        if isession == 'AB131_20240904_123728':
            json_config['behaviour_type'] = 'auditory'

        # Correct typos in behaviour names
        if json_config['behaviour_type'] == 'whisker_off':
            json_config['behaviour_type'] = 'whisker_off_1' # ensures correct string parsing

        if json_config['behaviour_type'] in ['fl', 'free licking']:
            json_config['behaviour_type'] = 'free_licking'

        if json_config['dummy_session_flag']:
            print('Ignoring dummy session found: {}, {}'.format(isession, json_config['behaviour_type']))
            sessions_to_remove.append(isession)
        else:
            # Add to list of behaviours
            behavior_type.append(json_config['behaviour_type'])

    print('Found the following sessions behaviors from raw data: {}'.format(behavior_type))
    sessions_list = [s for s in sessions_list if s not in sessions_to_remove]

    # ---------------------------------------
    # Count sessions based on list of laboratory-defined behaviours
    # Note: after agreement with rest of lab, add your "behaviour_type" label in this section
    # ---------------------------------------
    pretraining_behaviours = ['free_licking',
                              'auditory']
    n_aud = len([s for s in behavior_type if s in pretraining_behaviours])

    whisker_behaviours = ['whisker',
                          'whisker_psy',
                          'context',
                          'whisker_context']
    n_wh = len([s for s in behavior_type if s in whisker_behaviours])

    control_behaviours = ['whisker_on_1',
                          'whisker_off_1',
                          'whisker_on_2']
    n_ctrl = len([s for s in behavior_type if s in control_behaviours])

    # ------------------------------
    # Format behaviour label for NWB
    # ------------------------------
    if subject_id[0:2] == 'AB':
        # List of boolean if consecutive dates are increasing
        dates = [datetime.strptime(s.split('_')[1], '%Y%m%d') for s in sessions_list]
        increasing_dates = [dates[i] > dates[i - 1] for i in range(1, len(dates))]
        increasing_dates = [True] + increasing_dates  # first date is always increasing

        # If whisker sessions, label sessions aligned to first whisker day
        whisker_session_ids = [i for i, s in enumerate(behavior_type) if s in whisker_behaviours]
        if whisker_session_ids:
            label = list(range(-n_aud, 0)) + list(range(0, n_wh + n_ctrl))

            # Case of several sesssions per day: update labels if dates are not increasing
            for i in range(len(label)):
                # If date is not increasing, substract all labels by 1
                if not increasing_dates[i] and label[i] >= 0:
                    # Then update all subsequent labels by -1
                    label[i:] = [l-1 for l in label[i:]]

            label = [f"+{d}" if d > 0 else str(d) for d in label]
            behavior_type = [f"{t}_{l}" for t, l in zip(behavior_type, label)]

        # Else, label sessions in chronological order, by day
        else:
            label = np.cumsum(increasing_dates) - 1
            behavior_type = [f"{t}_{l}" for t, l in zip(behavior_type, label)]

        # Case several identical behaviours per day: add suffix for identical session labels on the same date
        sess_date_counts = defaultdict(int)
        beh_date_list = list(zip(behavior_type, dates))
        for i, (string, date) in enumerate(beh_date_list):
            # Increment count for this string and date combination
            sess_date_counts[(string, date)] += 1

            # If there are more than one occurrence of this string with the same date
            if sess_date_counts[(string, date)] > 1:
                # Add a suffix to the current occurrence
                suffix = "_" + str(sess_date_counts[(string, date)] - 1)
                beh_date_list[i] = (string + suffix, date)

    else:
        # Create behaviour-day/session index labels  aligned to first whisker day
        label = list(range(-n_aud, 0)) + list(range(0, n_wh + n_ctrl))
        label = [f"+{d}" if d > 0 else str(d) for d in label]
        behavior_type = [f"{t}_{l}" for t, l in zip(behavior_type, label)]

    # --------------------------------------
    # Create list of labels for each session
    # --------------------------------------

    training_days = list(zip(sessions_list, behavior_type))

    return training_days


def load_timestamps_data(trial_file, timestamps_dict):
    trial_list = np.load(trial_file, allow_pickle=True)
    trial_timestamps = timestamps_dict['trial_TTL']
    ci_timestamps = timestamps_dict['galvo_position']

    return trial_list, trial_timestamps, ci_timestamps


def get_trial_timestamps_dict(timestamps_dict, behavior_results_file, config_file):
    """
    Get trial timestamps from behavior results file and map them to trial types.
    Args:
        timestamps_dict:
        behavior_results_file:
        config_file:

    Returns:

    """

    # Read results files
    if os.path.splitext(behavior_results_file)[1] == '.json':
        with open(behavior_results_file, 'r') as f:
            perf_json = json.load(f)
        behavior_results = pd.DataFrame(perf_json['results'], columns=perf_json['headers'])
        # Remap GF columns.
        behavior_results = utils_gf.map_result_columns(behavior_results)
        n_trials_max = len(behavior_results)
    else:
        if os.path.splitext(behavior_results_file)[1] == '.txt':
            sep = r'\s+'
        else:
            sep = ','
        behavior_results = pd.read_csv(behavior_results_file, sep=sep, engine='python')
        # Because stitching tables manually with excel changes sep character.
        if behavior_results.columns.shape[0] == 1:
            behavior_results = pd.read_csv(behavior_results_file, sep=';', engine='python')
        n_trials_max = len(behavior_results)

    # TODO: get max number of trials possible
    if len(behavior_results) > len(timestamps_dict['trial_TTL']):
        n_trials_max = len(timestamps_dict['trial_TTL'])
        print(f"Found more trials in .csv file than TTL up/down signal, session must have been stopped "
              f"before saving the very last trial. Ignoring last trials of .csv table.")
    elif len(behavior_results) < len(timestamps_dict['trial_TTL']):
        n_trials_max = len(behavior_results)
        print(f"Found more trials in TTL up/down signal than in .csv file, logging of TTLs must have been interrupted "
                f"before the end of the behaviour. Ignoring last trial TTLs of session.")

    behavior_results = behavior_results.iloc[:n_trials_max, :]  # limit number of trials


    # Get trial outcomes, number of trials and trial types
    trial_outcomes = behavior_results['perf'].values
    trial_types = np.unique(trial_outcomes)

    if (timestamps_dict is None) or (timestamps_dict['trial_TTL'] == []):
        trial_timestamps = np.zeros((len(behavior_results), 2))
        trial_timestamps[:,0] = behavior_results['trial_time'].values
        trial_timestamps[:,1] = behavior_results['trial_time'].values + 1.0
    else:
        trial_timestamps = np.array(timestamps_dict['trial_TTL'])
    trial_timestamps_dict = dict()
    trial_indexes_dict = dict()

    # Get trial timestamps for each trial type
    for trial_type in trial_types:
        trial_idx = np.where(trial_outcomes == trial_type)[0]
        trial_timestamps_dict[trial_type] = np.transpose(np.array(trial_timestamps[trial_idx]))
        trial_indexes_dict[trial_type] = trial_idx

    # Mapping between performance number and trial type
    with open(config_file, 'r') as stream:
        bhv_mapping_file_data = yaml.safe_load(stream)['trial_map']
    old_keys = list(trial_timestamps_dict.keys())
    for old_key in old_keys:
        trial_timestamps_dict[bhv_mapping_file_data[old_key]] = trial_timestamps_dict.pop(old_key)
        trial_indexes_dict[bhv_mapping_file_data[old_key]] = trial_indexes_dict.pop(old_key)

    return trial_timestamps_dict, trial_indexes_dict

def get_piezo_licks_timestamps_dict(timestamps_dict):
    if timestamps_dict is None:
        return None
    else:
        if 'lick_trace' not in list(timestamps_dict.keys()):
            return None
        else:
            piezo_licks_timestamps_dict = timestamps_dict['lick_trace']
    return piezo_licks_timestamps_dict

def get_context_timestamps_dict(timestamps_dict, nwb_trial_table):
    context_timestamps_dict = dict()
    context_sound_dict = dict()

    # Handle case where no context timestamps are found
    if (timestamps_dict is None) or (timestamps_dict['trial_TTL'] == []):
        return None, None
    if 'context' not in list(timestamps_dict.keys()):
        return None, None
    if 'active' in nwb_trial_table['context'].unique() or 'passive' in nwb_trial_table['context'].unique():
        print(f"Ignoring 'active' or 'passive' in 'context' column from csv file as a behavioral epoch")
        return None, None
    if len(np.unique(nwb_trial_table['context'].values[:])) == 1:
        print(f"Found only 1 value in 'context' column from csv file : {nwb_trial_table['context'].values[0]}")
        return None, None


    # Get context timestamps
    context_on_off = timestamps_dict.get('context')
    if len(context_on_off) == 1:
        return None, None
    else:
        print(f"Found {len(context_on_off)} context blocks")

    rewarded_context = []
    context_sound = []
    n_context_blocks = len(context_on_off)
    for context_bloc in range(n_context_blocks):
        on_time = context_on_off[context_bloc][0]
        off_time = context_on_off[context_bloc][1]
        data_table = nwb_trial_table.loc[(nwb_trial_table['start_time'] > on_time) &
                                         (nwb_trial_table['stop_time'] < off_time)]
        # sanity check :
        if context_bloc == (len(context_on_off) - 1) and data_table.empty:
            print("Last context bloc has no trial skip it")
            n_context_blocks = n_context_blocks - 1
            continue
        if len(np.unique(data_table.context.values[:])) > 1:
            print(f"Seems like there is more than one context trial in this {context_bloc} block")
        rewarded_context.append(data_table.context.values[0])
        context_sound.append(data_table.context_background.values[0])

    rewarded_on_off = [context_on_off[i] for i in range(n_context_blocks) if rewarded_context[i] == 1]
    non_rewarded_on_off = [context_on_off[i] for i in range(n_context_blocks) if rewarded_context[i] == 0]

    context_timestamps_dict['rewarded'] = rewarded_on_off
    context_timestamps_dict['non-rewarded'] = non_rewarded_on_off

    rewarded_context = np.array(rewarded_context)
    rewarded_sound = context_sound[np.where(rewarded_context)[0][0]]
    non_rewarded_sound = context_sound[np.where(rewarded_context == 0)[0][0]]

    context_sound_dict['rewarded'] = rewarded_sound
    context_sound_dict['non-rewarded'] = non_rewarded_sound

    return context_timestamps_dict, context_sound_dict


def get_motivated_epoch_ts(timestamps_dict, nwb_trial_table):
    motivated_timestamps_dict = dict()
    trial_times = timestamps_dict['trial_TTL']
    frame_times = timestamps_dict['widefield'] if 'widefield' in timestamps_dict.keys() else timestamps_dict['galvo_position']
    n_trials = nwb_trial_table.shape[0]
    cut_off = n_trials - 50
    motivated_timestamps_dict['motivated'] = [(max(trial_times[0][0], frame_times[0]), trial_times[cut_off][0] - 2)]
    motivated_timestamps_dict['unmotivated'] = [(trial_times[cut_off][0] - 2, min(trial_times[-1][1], frame_times[-1]))]
    return motivated_timestamps_dict
    

def list_trial_type(results_table):
    """
    Get list and name trial types from results table.
    Args:
        results_table:

    Returns:

    """
    auditory_trials = np.where(results_table['is_auditory'])[0].astype(int)
    whisker_trials = np.where(results_table['is_whisker'])[0].astype(int)
    catch_trials = np.where(np.logical_not(results_table['is_stim']).astype(int))[0]
    light_trials = np.where(results_table['is_light'])[0].astype(int)

    n_trials = results_table['perf'].size

    trial_type_list = ["nan" for trial in range(n_trials)]
    for auditory_trial in auditory_trials:
        trial_type_list[auditory_trial] = "auditory"
    for whisker_trial in whisker_trials:
        trial_type_list[whisker_trial] = "whisker"
    for catch_trial in catch_trials:
        trial_type_list[catch_trial] = "catch"
    for light_trial in light_trials:
        trial_type_list[light_trial] = "light"

    return trial_type_list

def list_standard_trial_type(results_table):
    """
    Get list and name trial types from results table.
    Args:
        results_table:

    Returns:

    """
    auditory_trials = np.where(results_table['is_auditory'])[0].astype(int)
    whisker_trials = np.where(results_table['is_whisker'])[0].astype(int)
    no_stim_trials = np.where(np.logical_not(results_table['is_stim']).astype(int))[0]
    light_trials = np.where(results_table['is_light'])[0].astype(int)

    n_trials = results_table['perf'].size

    trial_type_list = [np.nan for trial in range(n_trials)]
    for auditory_trial in auditory_trials:
        trial_type_list[auditory_trial] = "auditory_trial"
    for whisker_trial in whisker_trials:
        trial_type_list[whisker_trial] = "whisker_trial"
    for no_stim_trial in no_stim_trials:
        trial_type_list[no_stim_trial] = "no_stim_trial"
    for light_trial in light_trials:
        trial_type_list[light_trial] = "light_trial"

    return trial_type_list


def build_standard_trial_table(config_file, behavior_results_file, timestamps_dict):
    """
    Build the standard trial table from behavior (and opto) results file and timestamps dictionary, used for all downstream analyses.

    Args:
        config_file: NWB configuration file
        behavior_results_file: results file from behavior
        timestamps_dict: dictionary of timestamps from NWB file

    Returns:

    """

    # Initialize table
    standard_trial_table = pd.DataFrame()

    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    experimenter = config['session_metadata']['experimenter']

    if experimenter == 'GF':  # GF has no config files.
        if config['session_metadata']['optogenetic'] == 'na':
            opto = 0
        else:
            opto = 1
        session_config = {
            'mouse_name': config['subject_metadata']['subject_id'],
            'opto_session': opto,
        }
    else:
        # Read session configuration file
        session_config_file = server_paths.get_session_config_file(config_file=config_file)
        with open(session_config_file) as json_file:
            session_config = json.load(json_file)

    # Get behaviour results and formatted trial type formatted as a list
    if os.path.splitext(behavior_results_file)[1] == '.json':
        with open(behavior_results_file, 'r') as f:
            perf_json = json.load(f)
        trial_table = pd.DataFrame(perf_json['results'], columns=perf_json['headers'])
    else:
        if os.path.splitext(behavior_results_file)[1] == '.txt':
            sep = r'\s+'
        else:
            sep = ','
        trial_table = pd.read_csv(behavior_results_file, sep=sep, engine='python')
        # Because stitching tables manually with excel changes sep character.
        #if trial_table.columns.shape[0] == 1:
        #    trial_table = pd.read_csv(behavior_results_file, sep=';', engine='python')
    if experimenter in ['GF', 'MI']:
        trial_table = utils_gf.map_result_columns(trial_table)

    trial_type_list = list_standard_trial_type(results_table=trial_table)
    n_trials = trial_table['perf'].size
    print(f"Read '.csv' file to build trial NWB trial table ({n_trials} trials)")

    # Get logged timestamps
    if (timestamps_dict is not None) and (timestamps_dict['trial_TTL'] != []):
        trial_timestamps = np.array(timestamps_dict['trial_TTL'])

    # Case when sessions were acquired before continuous logging of behavioral data
    else:
        trial_timestamps = np.zeros((n_trials, 2))
        trial_timestamps[:, 0] = trial_table['trial_time']  # use results table information instead
        trial_timestamps[:, 1] = trial_table['trial_time'] + 1.0  # response window

        if session_config['mouse_name'][0:2] == 'AB' and int(session_config['mouse_name'][2:-1]) < 68: #TODO: remove at one point if mice unused
            trial_table = check_trial_table_content(trial_table=trial_table)


    # Compare number of trials in .csv table and TTL timestamps
    if len(trial_timestamps[:, 0]) > n_trials:
        print(f"The .csv table has less trial than TTL up/down signal, session must have been stopped "
              f"before saving the very last trial. Ignoring last trial TTLs of session.")
        trial_timestamps = trial_timestamps[0:n_trials, :]

    elif len(trial_timestamps[:, 0]) < n_trials:
        print(f"The .csv table has more trials than TTL up/down signal, logging of TTLs must have been interrupted "
                f"before the end of the behaviour. Ignoring last trials of .csv table.")
        n_trials = len(trial_timestamps)
        trial_table = trial_table.iloc[0:n_trials, :]
        trial_type_list = trial_type_list[0:n_trials]

    # Format timestamps for specific events
    whisker_stim_time = [t + trial_table['baseline_window'][i] / 1000 if trial_table.loc[i].is_whisker == 1 else np.nan
                         for i, t in enumerate(trial_timestamps[:, 0])]
    auditory_stim_time = [t + trial_table['baseline_window'][i] / 1000 if trial_table.loc[i].is_auditory == 1 else np.nan
                          for i, t in enumerate(trial_timestamps[:, 0])]
    no_stim_time = [t + trial_table['baseline_window'][i] / 1000 if trial_table.loc[i].is_stim == 0 else np.nan
                    for i, t in enumerate(trial_timestamps[:, 0])]

    # Format response window times, relative to start time
    response_window_start_time = trial_timestamps[:, 0] + (trial_table['artifact_window'] + trial_table['baseline_window']) / 1000
    response_window_stop_time = response_window_start_time + trial_table['response_window'] / 1000

    # Format absence of licks: make reaction time as NaN
    trial_table.replace({'reaction_time': 0}, np.nan, inplace=True)

    # Define rewards availability
    if 'partial_reward_flag' in session_config.keys() and session_config['partial_reward_flag'] == 1:
        reward_available = [1 if(trial_table.loc[i].is_auditory == 1 or
                                 (trial_table.loc[i].is_whisker == 1 and trial_table.loc[i].is_reward == 1)) else 0
                            for i in range(n_trials)]
    else:
        reward_available = [1 if (trial_table.loc[i].is_auditory == 1 or
                                  (trial_table.loc[i].is_whisker == 1 and trial_table.loc[i].wh_reward == 1)) else 0
                            for i in range(n_trials)]

    # Build trial table
    standard_trial_table['id'] = trial_table['trial_number'] - 1  # zero-indexed
    standard_trial_table['start_time'] = trial_timestamps[:, 0]
    standard_trial_table['stop_time'] = trial_timestamps[:, 1]
    standard_trial_table['trial_type'] = trial_type_list
    standard_trial_table['perf'] = trial_table['perf']

    standard_trial_table['whisker_stim'] = trial_table['is_whisker']

    standard_trial_table['whisker_stim_amplitude'] = trial_table['wh_stim_amp']
    standard_trial_table['whisker_stim_duration'] = trial_table['wh_stim_duration']
    standard_trial_table['whisker_stim_time'] = whisker_stim_time

    standard_trial_table['auditory_stim'] = trial_table['is_auditory']
    standard_trial_table['auditory_stim_amplitude'] = trial_table['aud_stim_amp']
    standard_trial_table['auditory_stim_frequency'] = trial_table['aud_stim_freq']
    standard_trial_table['auditory_stim_duration'] = trial_table['aud_stim_duration']
    standard_trial_table['auditory_stim_time'] = auditory_stim_time

    standard_trial_table['no_stim'] = (~trial_table['is_stim'].astype(bool)).astype(int)
    standard_trial_table['no_stim_time'] = no_stim_time
    
    # Combine the stim time of all stim types in one vector. 
    standard_trial_table['stim_onset'] = np.nanmax([no_stim_time,
                                                   auditory_stim_time,
                                                   whisker_stim_time], axis=0)

    standard_trial_table['reward_available'] = reward_available
    standard_trial_table['response_window_start_time'] = response_window_start_time
    standard_trial_table['response_window_stop_time'] = response_window_stop_time

    standard_trial_table['lick_flag'] = trial_table['lick_flag']
    standard_trial_table['lick_time'] = response_window_start_time + trial_table['reaction_time']  # first lick time in response windows only
    standard_trial_table['abort_window_start_time'] = trial_timestamps[:, 0] - trial_table['quiet_window'] / 1000  # baseline is already at start, if not zero
    standard_trial_table['abort_window_stop_time'] = response_window_start_time - trial_table['artifact_window'] / 1000
    standard_trial_table['early_lick'] = trial_table['early_lick']

    # Add contextual information if relevant, nan otherwise
    if 'context_flag' in session_config.keys():
        if session_config['context_flag']:
            standard_trial_table['context'] = trial_table['wh_reward']
            standard_trial_table['context_background'] = trial_table['context_block']
        else:
            standard_trial_table['context'] = trial_table['context_block']  # active or passive
            standard_trial_table['context_background'] = np.nan

            # If 'active' and some nan values, replace all by np.nan
            if set(trial_table['context_block'].unique()) == set([np.nan, 'active']):
                standard_trial_table['context'] = np.nan
                standard_trial_table['context_background'] = np.nan
            # If 'passive' and some nan values, replace all by np.nan
            if set(trial_table['context_block'].unique()) == set([np.nan, 'passive']):
                standard_trial_table['context'] = np.nan
                standard_trial_table['context_background'] = np.nan
            if set(trial_table['context_block'].unique()) == set(['passive', 'active']):
                standard_trial_table['context'] = trial_table['context_block']  # 'active' or 'passive'
                standard_trial_table['context_background'] = np.nan

    else:  # case if context_flag is absence from session_config.json i.e. older sessions prior 2023
        standard_trial_table['context'] = np.nan
        standard_trial_table['context_background'] = np.nan

    # Add optogenetics information if relevant, nan otherwise
    if session_config['opto_session']:
        
        if experimenter == 'GF':
            standard_trial_table['opto_stim'] = trial_table['is_light']
            standard_trial_table['opto_grid_ap'] = np.nan
            standard_trial_table['opto_grid_ml'] = np.nan
            standard_trial_table['opto_grid_no'] = np.nan
            standard_trial_table['opto_stim_start_time'] = 1900
            standard_trial_table['opto_stim_stop_time'] = 3100
            standard_trial_table['opto_stim_amplitude'] = 10
            standard_trial_table['opto_stim_frequency'] = 100
        else:
            opto_config_file = server_paths.get_opto_config_file(config_file=config_file)
            with open(opto_config_file) as json_file:
                opto_config = json.load(json_file)
            opto_results_file = server_paths.get_opto_results_file(config_file=config_file)
            opto_trial_table = pd.read_csv(opto_results_file)

            if 'is_opto' not in opto_trial_table.keys():
                standard_trial_table['opto_stim'] = opto_trial_table['opto_amp'] > 0
            else:
                standard_trial_table['opto_stim'] = opto_trial_table['is_opto']

            standard_trial_table['opto_stim'] = opto_trial_table['is_opto']
            standard_trial_table['opto_grid_ap'] = opto_trial_table['coord_AP']
            standard_trial_table['opto_grid_ml'] = opto_trial_table['coord_ML']
            standard_trial_table['opto_grid_no'] = opto_trial_table['grid_no']
            standard_trial_table['opto_stim_start_time'] = standard_trial_table['start_time'] + opto_trial_table['baseline']
            standard_trial_table['opto_stim_stop_time'] = standard_trial_table['start_time'] + opto_trial_table['baseline'] + opto_trial_table['opto_duration']
            standard_trial_table['opto_stim_amplitude'] = opto_trial_table['opto_amp']
            standard_trial_table['opto_stim_frequency'] = opto_trial_table['opto_freq']
    else:
        standard_trial_table['opto_stim'] = 0
        standard_trial_table['opto_grid_ap'] = np.nan
        standard_trial_table['opto_grid_ml'] = np.nan
        standard_trial_table['opto_grid_no'] = np.nan
        standard_trial_table['opto_stim_start_time'] = np.nan
        standard_trial_table['opto_stim_stop_time'] = np.nan
        standard_trial_table['opto_stim_amplitude'] = np.nan
        standard_trial_table['opto_stim_frequency'] = np.nan

    if experimenter in ['GF', 'MI']:
        # Remove early licks.
        standard_trial_table = standard_trial_table.loc[standard_trial_table.perf != 6]
        standard_trial_table['id'] = np.arange(0,standard_trial_table.shape[0])
        standard_trial_table = standard_trial_table.reset_index(drop=True)

    if experimenter == 'AB':
        # Ensure string formatting of context
        standard_trial_table['context'] = standard_trial_table['context'].astype(str)

    return standard_trial_table


def build_simplified_trial_table(behavior_results_file, timestamps_dict):
    simplified_trial_table = pd.DataFrame()
    trial_table = pd.read_csv(behavior_results_file)
    trial_timestamps = np.array(timestamps_dict['trial_TTL'])
    trial_type_list = list_trial_type(results_table=trial_table)

    n_trials = trial_table['perf'].size
    print(f"Read '.csv' file to build trial NWB trial table ({n_trials} trials)")

    if len(trial_timestamps[:, 0]) > n_trials:
        print(f"csv table has one less trial than TTL up/down signal session must have been stop "
              f"before saving very last trial")
        trial_timestamps = trial_timestamps[0:-1, :]
    lick = trial_table['lick_flag'].values
    trial_outcome = ["Hit" if lick[trial] == 1 else "Miss" for trial in range(n_trials)]
    simplified_trial_table['trial_index'] = trial_table['trial_number']
    simplified_trial_table['start_time'] = trial_timestamps[:, 0]
    simplified_trial_table['stop_time'] = trial_timestamps[:, 1]
    simplified_trial_table['reaction_time'] = trial_table['reaction_time']
    simplified_trial_table['trial_type'] = trial_type_list
    simplified_trial_table['wh_reward'] = trial_table['wh_reward']
    simplified_trial_table['aud_reward'] = trial_table['aud_reward']
    simplified_trial_table['trial_outcome'] = trial_outcome
    simplified_trial_table['early_lick'] = trial_table['early_lick']
    simplified_trial_table['context'] = trial_table['wh_reward']
    simplified_trial_table['context_background'] = trial_table['context_block']

    return simplified_trial_table


def add_trials_to_nwb(nwb_file, trial_table):
    column_names = trial_table.columns
    columns_to_add = column_names[3:]

    for column in columns_to_add:
        nwb_file.add_trial_column(name=column, description="None")

    n_trials = trial_table['trial_type'].size
    for trial in range(n_trials):
        nwb_file.add_trial(start_time=trial_table['start_time'].values[trial],
                           stop_time=trial_table['stop_time'].values[trial],
                           reaction_time=trial_table['reaction_time'].values[trial],
                           trial_type=trial_table['trial_type'].values[trial],
                           wh_reward=trial_table['wh_reward'].values[trial],
                           aud_reward=trial_table['aud_reward'].values[trial],
                           trial_outcome=trial_table['trial_outcome'].values[trial],
                           early_lick=trial_table['early_lick'].values[trial],
                           context=trial_table['context'].values[trial],
                           context_background=trial_table['context_background'].values[trial])

    return

def add_trials_standard_to_nwb(nwb_file, trial_table):
    """
    Add trial table to NWB file.
    Args:
        nwb_file:
        trial_table:

    Returns:

    """

    column_names = trial_table.columns
    columns_to_add = [c for c in column_names if c not in ['id', 'start_time', 'stop_time']]
    columns_to_add.append('trial_id')  # To have trial id's as a column.

    for column in columns_to_add:
        nwb_file.add_trial_column(name=column, description="None")
    
    n_trials = trial_table['trial_type'].size
    for trial in range(n_trials):
        nwb_file.add_trial(id=trial_table['id'].values[trial],
                           trial_id=trial_table['id'].values[trial],
                           start_time=trial_table['start_time'].values[trial],
                           stop_time=trial_table['stop_time'].values[trial],
                           trial_type=trial_table['trial_type'].values[trial],
                           perf=trial_table['perf'].values[trial],

                           whisker_stim=trial_table['whisker_stim'].values[trial],
                           whisker_stim_amplitude=trial_table['whisker_stim_amplitude'].values[trial],
                           whisker_stim_duration=trial_table['whisker_stim_duration'].values[trial],
                           whisker_stim_time=trial_table['whisker_stim_time'].values[trial],

                           auditory_stim=trial_table['auditory_stim'].values[trial],
                           auditory_stim_amplitude=trial_table['auditory_stim_amplitude'].values[trial],
                           auditory_stim_frequency=trial_table['auditory_stim_frequency'].values[trial],
                           auditory_stim_duration=trial_table['auditory_stim_duration'].values[trial],
                           auditory_stim_time=trial_table['auditory_stim_time'].values[trial],

                           no_stim=trial_table['no_stim'].values[trial],
                           no_stim_time=trial_table['no_stim_time'].values[trial],
                           
                           stim_onset=trial_table['stim_onset'].values[trial],

                           reward_available=trial_table['reward_available'].values[trial],
                           response_window_start_time=trial_table['response_window_start_time'].values[trial],
                           response_window_stop_time=trial_table['response_window_stop_time'].values[trial],

                           lick_flag=trial_table['lick_flag'].values[trial],
                           lick_time=trial_table['lick_time'].values[trial],
                           abort_window_start_time=trial_table['abort_window_start_time'].values[trial],
                           abort_window_stop_time=trial_table['abort_window_stop_time'].values[trial],
                           early_lick=trial_table['early_lick'].values[trial],

                           context=trial_table['context'].values[trial],
                           context_background=trial_table['context_background'].values[trial],

                           opto_stim=trial_table['opto_stim'].values[trial],
                           opto_stim_start_time=trial_table['opto_stim_start_time'].values[trial],
                           opto_stim_stop_time=trial_table['opto_stim_stop_time'].values[trial],
                           opto_stim_amplitude=trial_table['opto_stim_amplitude'].values[trial],
                           opto_stim_frequency=trial_table['opto_stim_frequency'].values[trial],
                           opto_grid_ap=trial_table['opto_grid_ap'].values[trial],
                           opto_grid_ml=trial_table['opto_grid_ml'].values[trial],
                           opto_grid_no=trial_table['opto_grid_no'].values[trial]
                           )


def check_trial_table_content(trial_table):
    """
    Check if trial table contains all necessary columns.
    Relevant for older trial tables acquired before continuous logging.
    Somewhat arbitrary values but necessary to have them to add in NWB files.
    Args:
        trial_table:

    Returns:

    """
    if 'artifact_window' not in trial_table.columns:
        trial_table['artifact_window'] = 100
    if 'baseline_window' not in trial_table.columns:
        trial_table['baseline_window'] = 0
    if 'response_window' not in trial_table.columns:
        trial_table['response_window'] = 1000

    return trial_table
