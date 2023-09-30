import os
import numpy as np
import pandas as pd
import yaml
import json

def find_training_days(subject_id, input_folder): #TODO: make this more general OR customable

    """
    Find days of behavioural training, excluding other test/dummy sessions.
    Args:
        subject_id:
        input_folder:

    Returns:

    """
    sessions_list = os.listdir(os.path.join(input_folder, 'Training'))
    sessions_list = [s for s in sessions_list if os.path.isdir(os.path.join(input_folder, 'Training', s))]
    # Ordering in time with lexicographic ordering assumes %Y%m%d data format in session id.
    sessions_list = sorted(sessions_list)

    # Find session type (auditory or whisker day) and label days with integer relative to first whisker training day.
    behavior_type = []
    for isession in sessions_list:
        json_path = os.path.join(input_folder, 'Training', isession, 'session_config.json')
        with open(json_path, 'r') as f:
            json_config = json.load(f)
        behavior_type.append(json_config['behaviour_type'])
    behavior_type = ['free_licking' if behavior == 'free licking' else behavior for behavior in behavior_type]
    n_aud = len([s for s in behavior_type if s in ['free_licking', 'auditory']])
    n_wh = len([s for s in behavior_type if s in ['whisker', 'context']])
    label = list(range(-n_aud, 0)) + list(range(0, n_wh))
    label = [f"+{d}" if d > 0 else str(d) for d in label]
    behavior_type = [f"{t}_{l}" for t, l in zip(behavior_type, label)]

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
    if os.path.splitext(behavior_results_file)[1] == '.txt':
        sep = r'\s+'
    else:
        sep = ','
    behavior_results = pd.read_csv(behavior_results_file, sep=sep, engine='python')

    # Get trial outcomes, number of trials and trial types
    trial_outcomes = behavior_results['perf'].values
    trial_types = np.unique(trial_outcomes)
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


def get_context_timestamps_dict(timestamps_dict, nwb_trial_table):
    context_timestamps_dict = dict()
    context_sound_dict = dict()
    if 'context' not in list(timestamps_dict.keys()):
        return None, None
    if len(np.unique(nwb_trial_table['context_block'].values[:])) == 1:
        print(f"Found only 1 value in 'context_block' column from csv file : {nwb_trial_table['context_block'].values[0]}")
        return None, None

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
        data_table = nwb_trial_table.loc[(nwb_trial_table['trial_start'] > on_time) &
                                         (nwb_trial_table['trial_stop'] < off_time)]
        # sanity check :
        if context_bloc == (len(context_on_off) - 1) and data_table.empty:
            print("Last context bloc has no trial skip it")
            n_context_blocks = n_context_blocks - 1
            continue
        if len(np.unique(data_table.context_block.values[:])) > 1 or len(np.unique(data_table.wh_reward.values[:])) > 1:
            print(f"Seems like there is more than one context trial in this {context_bloc} block")
        rewarded_context.append(data_table.wh_reward.values[0])
        context_sound.append(data_table.context_block.values[0])

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

    trial_type_list = ["NA" for trial in range(n_trials)]
    for auditory_trial in auditory_trials:
        trial_type_list[auditory_trial] = "auditory"
    for whisker_trial in whisker_trials:
        trial_type_list[whisker_trial] = "whisker"
    for catch_trial in catch_trials:
        trial_type_list[catch_trial] = "catch"
    for light_trial in light_trials:
        trial_type_list[light_trial] = "light"

    return trial_type_list


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
    simplified_trial_table['trial_start'] = trial_timestamps[:, 0]
    simplified_trial_table['trial_stop'] = trial_timestamps[:, 1]
    simplified_trial_table['reaction_time'] = trial_table['reaction_time']
    simplified_trial_table['trial_type'] = trial_type_list
    simplified_trial_table['wh_reward'] = trial_table['wh_reward']
    simplified_trial_table['aud_reward'] = trial_table['aud_reward']
    simplified_trial_table['trial_outcome'] = trial_outcome
    simplified_trial_table['early_lick'] = trial_table['early_lick']
    simplified_trial_table['context_block'] = trial_table['context_block']

    return simplified_trial_table

def build_full_trial_table(behavior_results_file, timestamps_dict):
    """
    Build a trial table from the results.csv file session timestamps.
    Args:
        behavior_results_file: path to the results.csv file
        timestamps_dict: dictionary of session timestamps       

    Returns:

    """
    full_trial_table = pd.DataFrame()
    trial_table = pd.read_csv(behavior_results_file)
    trial_timestamps = np.array(timestamps_dict['trial_TTL'])
    trial_type_list = list_trial_type(results_table=trial_table)

    n_trials = trial_table['perf'].size
    print(f"Read '.csv' file to build trial NWB trial table ({n_trials} trials)")

    if len(trial_timestamps[:, 0]) > n_trials:
        print(f"csv table has one less trial than TTL up/down signal session must have been stop "
              f"before saving very last trial in the results.csv file")
        trial_timestamps = trial_timestamps[0:-1, :]


    lick = trial_table['lick_flag'].values
    trial_outcome = ["Hit" if lick[trial] == 1 else "Miss" for trial in range(n_trials)]

    full_trial_table['trial_index'] = trial_table['trial_number']
    full_trial_table['trial_start'] = trial_timestamps[:, 0]
    full_trial_table['trial_stop'] = trial_timestamps[:, 1]
    full_trial_table['reaction_time'] = trial_table['reaction_time']
    full_trial_table['trial_type'] = trial_type_list
    full_trial_table['wh_reward'] = trial_table['wh_reward']
    full_trial_table['aud_reward'] = trial_table['aud_reward']
    full_trial_table['trial_outcome'] = trial_outcome

    full_trial_table['association_flag'] = trial_table['association_flag']
    full_trial_table['quiet_window'] = trial_table['quiet_window']
    full_trial_table['iti'] = trial_table['iti']
    full_trial_table['response_window'] = trial_table['response_window']
    full_trial_table['artifact_window'] = trial_table['artifact_window']
    full_trial_table['is_reward'] = trial_table['is_reward']

    return full_trial_table


def add_trials_to_nwb(nwb_file, trial_table):
    column_names = trial_table.columns
    columns_to_add = column_names[3:]

    for column in columns_to_add:
        nwb_file.add_trial_column(name=column, description="None")

    n_trials = trial_table['trial_type'].size
    for trial in range(n_trials):
        nwb_file.add_trial(start_time=trial_table['trial_start'].values[trial],
                           stop_time=trial_table['trial_stop'].values[trial],
                           reaction_time=trial_table['reaction_time'].values[trial],
                           trial_type=trial_table['trial_type'].values[trial],
                           wh_reward=trial_table['wh_reward'].values[trial],
                           aud_reward=trial_table['aud_reward'].values[trial],
                           trial_outcome=trial_table['trial_outcome'].values[trial],
                           early_lick=trial_table['early_lick'].values[trial],
                           context_block=trial_table['context_block'].values[trial])

    return


def add_trials_full_to_nwb(nwb_file, trial_table):
    """
    Add trial table to NWB file.
    Args:
        nwb_file: NWB file object
        trial_table: trial table pandas dataframe

    Returns:

    """

    column_names = trial_table.columns
    columns_to_add = [c for c in column_names if c not in ['trial_index', 'trial_start', 'trial_stop']]

    for column in columns_to_add:
        nwb_file.add_trial_column(name=column, description="None")

    n_trials = trial_table['trial_type'].size
    for trial in range(n_trials):
        nwb_file.add_trial(start_time=trial_table['trial_start'].values[trial],
                           stop_time=trial_table['trial_stop'].values[trial],
                           reaction_time=trial_table['reaction_time'].values[trial],
                           trial_type=trial_table['trial_type'].values[trial],
                           wh_reward=trial_table['wh_reward'].values[trial],
                           aud_reward=trial_table['aud_reward'].values[trial],
                           trial_outcome=trial_table['trial_outcome'].values[trial],
                           association_flag=trial_table['association_flag'].values[trial],
                           quiet_window=trial_table['quiet_window'].values[trial],
                           iti=trial_table['iti'].values[trial],
                           response_window=trial_table['response_window'].values[trial],
                           artifact_window=trial_table['artifact_window'].values[trial],
                           is_reward=trial_table['is_reward'].values[trial]
                           )
    return



