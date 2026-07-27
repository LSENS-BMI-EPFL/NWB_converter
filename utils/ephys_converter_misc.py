#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: NWB_converter
@file: ephys_converter_misc.py
@time: 8/24/2023 9:25 AM
"""
import datetime
import itertools
import os
import pathlib
import json
import numpy as np
import pandas as pd
import yaml
import re
import matplotlib.pyplot as plt
from pandas import Int64Dtype
from scipy.spatial import cKDTree


from utils import server_paths
from utils.continuous_processing import detect_piezo_lick_times, plot_exposure_times
#from utils.read_sglx import readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ChannelCountsNI
from utils.readSLGX import readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ChannelCountsNI
# MAP of (AP,ML) coordinates relative to bregma
# Update this for new target areas
AREA_COORDINATES_MAP = {
    'wS1': 'IOS',
    'wS2': 'IOS',
    'A1': 'IOS',
    'wM1': (1, 1), # Esmaeili et al.
    'wM2': (2, 1), # Esmaeili et al.
    'ALM': (2.5,1.5), # Esmaeili et al.
    'OFC': (3, 1),      # ?
    'mPFC': (2, 0.5), # Esmaeili et al., Oryschuuk et al.
    'Vis': (-3.8, 2.5), # Esmaeili et al.
    'PPC': (-2, 1.75), # Fritjof Helmchen papers ('PPC-A')
    'dCA1': (-2.7, 2), # Esmaeili et al.
    'tjM1': (2, 2), # Mayrhofer et al.
    'DLS': (0, 3.5), # Esmaeili et al., Sippy et al., 
    'SC':  (-3.8, 0.5), # ?
    'RSP': (-1.5, 0.5), # Bech, Dard et al., frontal
    'tjS1': (0.6, 3.8) # Bech, Dard et al.
}

# Note: from https://billkarsh.github.io/SpikeGLX/Sgl_help/Metadata_Help.html
# Requires updating for new NP probes
NP_PROBE_TYPE_MAP = {
    # NP 1.0-like
    0:    'NP1.0',
    1020: 'NP1.0',
    1030: 'NP1.0',
    1100: 'NP1.0',
    1120: 'NP1.0',
    1121: 'NP1.0',
    1122: 'NP1.0',
    1123: 'NP1.0',
    1200: 'NP1.0',
    1300: 'NP1.0',
    # NP 2.0 single shank
    21:   'NP2.0',
    2003: 'NP2.0',
    # NP 2.0 four-shank
    24:   'NP2.0',
    2013: 'NP2.0',
    # Quad-base
    2020: 'NP2.0',
    # UHD programmable
    1110: 'NP1.0',
}

DEBUG_PLOT = False

def get_probe_insertion_info(config_file):
    """
    Read probe insertion information from a metadata external file.
    Args:
        config_file:

    Returns:

    """
    # This is experimenter-specific tracking of that information
    path_to_probe_info = server_paths.get_path_to_probe_insertion_info(config_file)
    probe_info_df = pd.read_excel(path_to_probe_info)

    return probe_info_df


def get_target_location(config_file, device_name):
    """
    Read location target: hemisphere, stereotaxic coordinate, angles from a metadata external file.
    Args:
        config_file: Path to config file
        device_name: Name of the device (e.g. imec0)

    Returns:
    """

    # Read config file
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    location_df = get_probe_insertion_info(config_file=config_file)
    session_date = config['session_metadata']['session_id'].split('_')[1]
    # session date is formatted as YYYYMMDD in here, format as DD.MM.YYYY
    YYYY,MM,DD = session_date[0:4], session_date[4:6], session_date[6:8]
    session_date = f"{DD}.{MM}.{YYYY}"
    # Convert date fo location_df as datetime with only DD.MM.YYYY
    location_df['date'] =  pd.to_datetime(location_df['date']).dt.strftime('%d.%m.%Y').astype(str)

    # Keep subset for mouse and probe_id
    mouse_name = config.get('subject_metadata').get('subject_id')
    location_df = location_df[(location_df['mouse_name'] == mouse_name)
                             & (location_df['date'] == session_date)
                              & (location_df['probe_id'] == int(device_name[-1]))
                              ]

    # Get coordinates of target area
    target_area = location_df['target_area'].values[0]
    if target_area in AREA_COORDINATES_MAP.keys():

        if type(AREA_COORDINATES_MAP[target_area]) is tuple:

            ap = AREA_COORDINATES_MAP[target_area][0]
            ml = AREA_COORDINATES_MAP[target_area][1]

        elif type(AREA_COORDINATES_MAP[target_area]) is str:

            ap = AREA_COORDINATES_MAP[target_area]
            ml = AREA_COORDINATES_MAP[target_area]
        else:
            print('Unknown type for AP, ML coordinates. Setting to NaN')
            ap, ml = (np.nan, np.nan)

    else:
        print(f'No standard coordinates found for this target ({target_area}) area. Setting to NaN')
        ap, ml = (np.nan, np.nan)

    # Create ephys target location dictionary
    location_dict = {
        'hemisphere': 'left',
        'area': location_df['target_area'].values[0],
        'ap': ap,
        'ml': ml,
        'azimuth': location_df['azimuth'].values[0],
        'elevation': location_df['elevation'].values[0],
        'depth': location_df['depth'].values[0],
    }
    return location_dict

def read_ephys_binary_data(bin_file, meta_file):
    """
    Read ephys binary data and return a dictionary with the data.
    This only reads the analog data of these binary files.
    Args:
        bin_file: path to binary file
        meta_file: path to meta file
    Returns:
    """
    print('Read ephys binary data')

    # Parameters about what data to read
    # This can be user-specific
    # TODO: read a config file ephys_channel_dict from the yaml file

    t_start = 0
    t_end = -1
    channel_dict = {0: 'sync',
                    1: 'trial_TTL',
                    2: 'whisker_stim',
                    3: 'auditory_stim',
                    4: 'valve',
                    5: 'cam1',
                    6: 'cam2',
                    7: 'lick_trace'}
    channel_list = list(channel_dict.keys())

    # Read metafile
    meta_dict = readMeta(pathlib.Path(meta_file))

    # Parameters common to NI and IMEC data
    s_rate = SampRate(meta_dict)
    first_sample = int(s_rate * t_start)
    last_sample = int(s_rate * t_end)

    # Read binary file
    raw_data = makeMemMapRaw(pathlib.Path(bin_file), meta_dict)

    # Note: this deals with analog data only
    select_data = raw_data[channel_list, first_sample:last_sample]

    # Read IMEC data
    if meta_dict['typeThis'] == 'imec':
        # Apply gain correction and convert to uV
        conv_data = 1e6 * GainCorrectIM(select_data, channel_list, meta_dict)

        conv_data_dict = {}
        conv_data_dict['imec'] = conv_data

    # Read NI data
    else:
        #MN, MA, XA, DW = ChannelCountsNI(meta_dict)
        #print("NI channel counts: %d, %d, %d, %d" % (MN, MA, XA, DW))
        # Apply gain correction and convert to Volt
        conv_data = GainCorrectNI(select_data, channel_list, meta_dict)

        conv_data_dict = {}
        for chan_idx, chan_key in channel_dict.items():
            channel_data = conv_data[chan_idx, :]

            if chan_key == 'lick_trace':
                channel_data = np.abs(channel_data)
            conv_data_dict[chan_key] = channel_data

    return conv_data_dict

def load_ephys_sync_timestamps(config_file, log_timestamps_dict, experimenter=None):
    """
    Load sync timestamps derived from CatGT/TPrime from config file.
    Add and compare timestamps with log_continuous.bin timestamps.
    Args:
        config_file: path to config file
        log_timestamps_dict: dictionary of timestamps from log_continuous.bin
    Returns:

    """

    event_map = {
        'trial_start_times': 'trial_TTL',
        'cam0_frame_times': 'cam1',
        'cam1_frame_times': 'cam2',
        'whisker_stim_times': 'whisker_stim_times',
        'auditory_stim_times': 'auditory_stim_times',
        'valve_times': 'reward_times',
    }

    # List event times existing in folder
    sync_event_times_folder = server_paths.get_sync_event_times_folder(config_file, experimenter=experimenter)
    event_files = [f for f in os.listdir(sync_event_times_folder) if f.endswith('.txt')]
    event_keys = [f.split('.')[0] for f in event_files]
    print('Existing sync event times:', event_keys)

    timestamps_dict = {}
    events_to_do = ['trial_start_times', 'cam0_frame_times', 'cam1_frame_times', 'valve_times']
    events_available = [event for event in events_to_do if event in event_keys]
    for event in events_available:
        print('Ephys session with {} event'.format(event))

        # Load sync timestamps
        timestamps = np.loadtxt(os.path.join(sync_event_times_folder, event + '.txt'))

        # Make sure same number as from log_continuous.bin
        if event == 'trial_start_times':
            if len(timestamps) != len(log_timestamps_dict[event_map[event]]):
                print(
                    'Warning: {} has {} timestamps from nidq.bin (CatGT), while {} has {} timestamps from log_continuous.bin'.format(
                        event, len(timestamps), event_map[event], len(log_timestamps_dict[event_map[event]]))
                )

        # Add to dictionary
        timestamps_dict[event_map[event]] = timestamps

    # Add piezo lick timestamps separately
    sync_delay = get_sglx_behaviour_log_delay(log_timestamps_dict, timestamps_dict)
    timestamps_dict['lick_trace'] = log_timestamps_dict['lick_trace'] + sync_delay

    return timestamps_dict


def format_ephys_timestamps(config_file, ephys_timestamps_dict, n_frames_dict):
    """
    Format ephys timestamps as (on, off) tuples for NWB.
    Args:
        config_file: path to config file
        ephys_timestamps_dict: dictionary of timestamps from SpikeGLX/CatGT/TPrime NIDQ acquisition

    Returns:

    """

    # Init. new timestamps dict
    timestamps_dict = {}
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config.get('behaviour_metadata').get('camera_flag'):
        movie_files = server_paths.get_session_movie_files(config_file)
        if movie_files is not None:
            print(f'Movie files {len(movie_files)} during ephys:', [os.path.basename(f) for f in movie_files])
            movie_file_names = [os.path.basename(f) for f in movie_files]
            movie_file_suffix = [f.split('-')[0] for f in movie_file_names]
            movie_file_suffix = [f.split('_')[1] for f in movie_file_suffix]
            movie_file_suffix = [f.split(' ')[0] for f in movie_file_suffix]
        else:
            movie_files = None
            movie_file_suffix = None
    else:
        movie_files = None
        movie_file_suffix = None

    # Format each timestamps type separately
    for event in ephys_timestamps_dict.keys():

        timestamps = ephys_timestamps_dict[event]

        if event == 'trial_TTL':

            # Remove last timestamp that signals session end
            ts_on = timestamps[:-1]

            # Get trial stop times
            behavior_results_file = server_paths.get_behavior_results_file(config_file)
            trial_table = pd.read_csv(behavior_results_file)
            trial_response_windows = trial_table.response_window.values / 1000
            trial_artifact_windows = trial_table.artifact_window.values / 1000
            trial_durations_sec = trial_response_windows + trial_artifact_windows
            trial_durations_sec = trial_durations_sec.astype(float)

            # Keep only as many trials as in recorded behaviour output
            n_trials_behaviour = len(trial_durations_sec)
            n_trials_ts = len(ts_on)
            if n_trials_behaviour > n_trials_ts:
                trial_durations_sec = trial_durations_sec[:n_trials_ts]
            elif n_trials_behaviour < n_trials_ts:
                ts_on = ts_on[:n_trials_behaviour]
            else:
                ts_on = ts_on
            ts_off = ts_on + trial_durations_sec
            timestamps = list(zip(ts_on, ts_off))

            timestamps_dict[event] = timestamps

        elif event in ['cam1', 'cam2']:

            view_key_mapper = {
                'cam1': 'top',
                'cam2': 'side'
            }
            # If not movies or specific movie absent, set timestamps to empty list
            if movie_files is None:
                timestamps_dict[event] = []
            elif view_key_mapper[event] not in movie_file_suffix:
                timestamps_dict[event] = []
            else:
                ts_on = timestamps

                # Remove first/last pulses due to camera being turned ON or OFF
                # These pulse are several tens of ms long >>> 2ms exposure time
                diff_ts_on = np.diff(ts_on)
                startup_pulse_thresh = 0.05 # 50 ms
                if diff_ts_on[0] > startup_pulse_thresh:
                    ts_on = ts_on[1:]
                if diff_ts_on[-1] > startup_pulse_thresh:
                    ts_on = ts_on[:-1]

                # Check if last exposure cut (detected with behaviour binary file)
                if '{}_info'.format(event) in n_frames_dict.keys():
                    if n_frames_dict['{}_info'.format(event)]['last_exposure_cut']:
                        ts_on = ts_on[:-1]
                        print('Removed last exposure TTL of {}'.format(event))

                # Get timestamps as (on, off) tuples
                exposure_time = float(config['behaviour_metadata']['camera_exposure_time']) / 1000
                ts_off = ts_on + exposure_time
                timestamps = list(zip(ts_on, ts_off))
                timestamps_dict[event] = timestamps


        elif event == 'reward_times':
            timestamps = list(zip(timestamps, itertools.repeat(np.nan)))
            timestamps_dict[event] = timestamps

        elif event == 'lick_trace':
            print('Info: lick_trace timestamps are already formatted as (on,off) tuples.')

        else:
            print('Warning: {} is not a recognized timestamp type'.format(event))

    print('Done formatting ephys timestamps as tuples.')
    return timestamps_dict


def get_sglx_behaviour_log_delay(log_timestamps_dict, ephys_timestamps_dict):
    """
    Get delay between SpikeGLX and behaviour logging timestamps.
    SpikeGLX sessions start recording before behaviour sessions.
    Args:
        log_timestamps_dict: dictionary of timestamps from log_continuous.bin
        ephys_timestamps_dict: dictionary of timestamps from CatGT/TPrime NIDQ acquisition

    Returns:

    """

    # Get trial timestamps
    log_trial_ts = log_timestamps_dict['trial_TTL']
    ephys_trial_ts = ephys_timestamps_dict['trial_TTL']

    # Get first trials timestamps onset
    log_sess_start = log_trial_ts[0][0]  # (on,off)-formatted
    ephys_trial_ts = ephys_trial_ts[0]

    if isinstance(ephys_trial_ts, tuple):
        ephys_sess_start = ephys_trial_ts[0] # (on,off)-formatted
    else:
        ephys_sess_start = ephys_trial_ts # before (on,off)-format

    time_delay = ephys_sess_start - log_sess_start

    return time_delay


def extract_ephys_timestamps(config_file, continuous_data_dict, threshold_dict, log_timestamps_dict, n_frames_dict, experimenter=None):
    """
    Load and format ephys timestamps for continuous_log_analysis.
    Args:
        config_file: path to config file
        continuous_data_dict: dictionary of continuous data from SpikeGLX
        threshold_dict: dictionary of thresholds for continuous data processing
        log_timestamps_dict: dictionary of timestamps from log_continuous.bin
        n_frames_dict: dictionary of number of frames for each camera
        experimenter: (Optional) experimenter initials, provide if experimenter and mouse initials are different

    Returns:

    """
    print("Extract ephys timestamps")

    # Load and format existing timestamps extracted by CatGT and TPrime
    timestamps_dict = load_ephys_sync_timestamps(config_file, log_timestamps_dict, experimenter=experimenter)
    timestamps_dict = format_ephys_timestamps(config_file, timestamps_dict, n_frames_dict)

    # Extract timestamps from ephys-related binary files
    ephys_nidq_meta, _ = server_paths.get_raw_ephys_nidq_files(config_file)
    meta_dict = readMeta(pathlib.Path(ephys_nidq_meta))
    lick_threshold = threshold_dict.get('lick_trace')
    lick_timestamps = detect_piezo_lick_times(continuous_data_dict,
                                              ni_session_sr=meta_dict['niSampRate'],
                                              lick_threshold=lick_threshold,
                                              sigma=500)
    # Format as tuples of on/off times for NWB
    lick_timestamps_on_off = list(zip(lick_timestamps, itertools.repeat(np.nan)))
    timestamps_dict['lick_trace'] = lick_timestamps_on_off

    # The only mandatory timestamps
    assert 'trial_TTL' in timestamps_dict.keys()
    assert isinstance(timestamps_dict['trial_TTL'][0], tuple)

    n_frames_dict = {k: len(v) for k, v in timestamps_dict.items()}

    return timestamps_dict, n_frames_dict


def create_electrode_table(nwb_file):
    """
    Create electrode table in nwb file.
    Args:
        nwb_file: NWB file object

    Returns:

    """
    # Create ElectrodeTable object
    dict_columns_to_add = {'index_on_probe': 'index of saved channel per probe per shank',
                           'shank': 'shank number',
                           'shank_col': 'column number of electrode on shank',
                           'shank_row': 'row number of electrode on shank',
                           # Sample space
                           #'ccf_id': 'ccf region ID',
                           #'ccf_acronym': 'ccf region acronym',
                           #'ccf_name': 'ccf region name',
                           #'ccf_parent_id': 'ccf parent region ID',
                           #'ccf_parent_acronym': 'ccf parent region acronym',
                           #'ccf_parent_name': 'ccf parent region name',
                           # Atlas space
                           'ccf_atlas_ml': 'ccf atlas coordinate in ml axis',
                           'ccf_atlas_ap': 'ccf atlas coordinate in ap axis',
                           'ccf_atlas_dv': 'ccf atlas coordinate in dv axis',
                           'ccf_atlas_id': 'ccf atlas region ID',
                           'ccf_atlas_acronym': 'ccf atlas region acronym',
                           'ccf_atlas_name': 'ccf atlas region name',
                           'ccf_atlas_parent_id': 'ccf atlas parent region ID',
                           'ccf_atlas_parent_acronym': 'ccf atlas parent region acronym',
                           'ccf_atlas_parent_name': 'ccf atlas parent region name',
                           }

    for col_key, col_desc in dict_columns_to_add.items():
        nwb_file.add_electrode_column(name=col_key, description=col_desc)

    return


def create_simplified_unit_table(nwb_file):
    """
    Create a simplified units table in nwb file.
    Args:
        nwb_file: NWB file object

    Returns:

    """

    # Create Units table (default columns are id and spike_times)
    dict_columns_to_add = {
        'cluster_id': 'cluster index, from KS(probe-wise)',
        'peak_channel': 'electrode with max waveform amplitude, from KS',
        'electrode_group': 'ElectrodeGroup object (i.e. probe) recording the unit',
        'depth': 'depth of peak electrode, in probe space, from KS',
        'ks_label': 'unit quality label, form Kilosort and curation (Phy): “good”, “mua”',
        'firing_rate': 'total firing rate in session, in Hz',
        'waveform_mean': 'mean spike waveform (a vector), in uV',
        'sampling_rate': 'sampling rate used for that probe, in Hz',
        'duration': 'spike duration, in ms, from trough to peak',
        'pt_ratio': 'peak-to-trough ratio',
    }
    for col_key, col_desc in dict_columns_to_add.items():
        nwb_file.add_unit_column(name=col_key, description=col_desc)

    return


def create_unit_table(nwb_file):
    """
    Create units table in nwb file.
    Args:
        nwb_file: NWB file object

    Returns:

    """

    # Create Units table (default columns are id and spike_times)
    dict_columns_to_add = {
        'cluster_id': 'cluster index, from KS(probe-wise)',
        'peak_channel': 'electrode with max waveform amplitude, from KS',
        'electrode_group': 'ElectrodeGroup object (i.e. probe) recording the unit',
        'depth': 'depth of peak electrode, in probe space, from KS',
        'ks_label': 'unit quality label, from Kilosort: “good”, “mua”',
        'group': 'unit quality label, after Phy curation: “good”, “mua”, "noise"',
        'bc_label': 'unit quality label, from Bombcell: "good","mua","non-soma"',
        'firing_rate': 'total firing rate in session, in Hz',
        'maxChannels': 'channel of max waveform amplitude',
        #'bc_cluster_id': 'bombcell-based cluster ID',
        'useTheseTimesStart': 'start time for quality metric calculation',
        'useTheseTimesStop': 'stop time for quality metric calculation',
        'percentageSpikesMissing_gaussian': 'esimated percentage of spikes missing',
        'percentageSpikesMissing_symmetric': 'estimated percentage of spikes missing symmetrically',
        'fractionRPVs_estimatedTauR': 'estimated percent of refractory period violations (Hill et al., 2011)',
        'presenceRatio': 'number of time chunks of specific size containing at least one spike over total number of time chunks',
        'maxDriftEstimate':'difference between maximum and minimum median peak channels across bins',
        'cumDriftEstimate':'same as maxDriftEstimate but cumulative',
        'nSpikes': 'number of spikes',
        'nPeaks': 'number of template waveform peaks on peak channel',
        'nTroughs': 'number of template waveform troughs on peak channel',
        #'isSomatic': 'waveforms classified as somatic (Deligkaris et al., 2016)',
        'waveformDuration_peakTrough': 'peak-to-trough template waveform duration, in us',
        'spatialDecaySlope': 'slope of spatial decay of template waveform amplitude across channels up to 100 um away, in (a.u.)/um',
        'waveformBaselineFlatness': 'ratio of max. value in baseline window vs. max. value in waveform window',
        'rawAmplitude': 'raw mean waveform maximum amplitude, in uV',
        'signalToNoiseRatio': 'maximum waveform value (peak channel) divided by the variance across its raw extracted waveform baselines ',
        'Lratio':'How likely are spikes outside this cluster to actually belong inside it',
        'isolationDistance': 'interpreted as a measure of distance from the unit to the nearest cluster',
        'waveform_mean': 'mean spike waveform from actual data, in uV',
        'sampling_rate': 'sampling rate used for that probe, in Hz',
        'duration': 'spike duration, in ms, from trough to peak',
        'pt_ratio': 'peak-to-trough ratio',
        'ccf_ml': 'ccf peak channel coordinate in ml axis',
        'ccf_ap': 'ccf peak channel coordinate in ap axis',
        'ccf_dv': 'ccf peak channel coordinate in dv axis',
        'ccf_id': 'ccf region ID from histology only',
        'ccf_acronym': 'ccf region acronym from histology only',
        'ccf_name': 'ccf region name from histology only',
        'ccf_parent_id': 'ccf parent region ID',
        'ccf_parent_acronym': 'ccf parent region acronym',
        'ccf_parent_name': 'ccf parent region name',
        'ccf_atlas_ml': 'ccf atlas coordinate in ml axis after ephys-atlas alignment',
        'ccf_atlas_ap': 'ccf atlas coordinate in ap axis after ephys-atlas alignment',
        'ccf_atlas_dv': 'ccf atlas coordinate in dv axis after ephys-atlas alignment',
        'ccf_atlas_id': 'ccf atlas region ID after ephys-atlas alignment',
        'ccf_atlas_acronym': 'ccf atlas region acronym after ephys-atlas alignment',
        'ccf_atlas_name': 'ccf atlas region name after ephys-atlas alignment',
        'ccf_atlas_parent_id': 'ccf atlas parent region ID after ephys-atlas alignment',
        'ccf_atlas_parent_acronym': 'ccf atlas parent region acronym after',
        'ccf_atlas_parent_name': 'ccf atlas parent region name after ephys-atlas alignment',

    }
    for col_key, col_desc in dict_columns_to_add.items():
        nwb_file.add_unit_column(name=col_key, description=col_desc)

    return

def build_unit_table(imec_folder, sync_spike_times_path):
    """
    Build unit table from spike sorting/curation output.
    Args:
        imec_folder:
        sync_spike_times_path:

    Returns:

    """

    # Init. table
    unit_table = pd.DataFrame()

    # ----------------------------
    # Load Kilosort cluster table
    # ----------------------------
    imec_folder = pathlib.Path(imec_folder)
    kilosort_outputs = list(imec_folder.glob('kilosort*'))
    kilosort_outputs = [k for k in kilosort_outputs if 'kilosort_like' not in k.name]
    if len(kilosort_outputs) > 1: # if multiple kilosort versions, get the latest
        versions = []
        for ks_folder in kilosort_outputs:
            # Extract version number after 'kilosort'
            match = re.search(r'kilosort(\d+(?:\.\d+)*)', ks_folder.name.lower())
            version_str = match.group(1)
            # Convert to tuple of integers for proper comparison (e.g., "2.5" -> (2, 5))
            version_tuple = tuple(map(int, version_str.split('.')))
            versions.append((version_tuple, ks_folder))

        # Find the ks_folder with the highest version
        kilosort_output = max(versions, key=lambda x: x[0])[1]
        print(f"Multiple kilosort versions found. Using latest: {kilosort_output.name}")

    elif len(kilosort_outputs) == 1:
        kilosort_output = kilosort_outputs[0]
    else:
        print('No spike sorting at: {}'.format(imec_folder))
        return None

    ks_folder_name = kilosort_output.name
    # Spikeinterface adds another ks_folder 'sorter_output' in the kilosort ks_folder
    if (kilosort_output / 'sorter_output').exists():
        kilosort_output = kilosort_output / 'sorter_output'

    cluster_info_path = kilosort_output / 'cluster_info.tsv'
    try:
        cluster_info_df = pd.read_csv(cluster_info_path, sep='\t')
    except FileNotFoundError:
        print('No spike sorting at: {}'.format(cluster_info_path))
        return None

    cluster_info_df.rename(columns={'KSLabel': 'ks_label',
                                    'Amplitude': 'amplitude',
                                    'ContamPct': 'contam_pct',
                                    'bc_unitType': 'bc_label'}, inplace=True)

    # Find if cluster had a curated label
    try:
        cluster_info_df['curated'] = cluster_info_df.apply(lambda x: 0 if pd.isnull(x.group) else 1, axis=1)
        # Phy-based new clusters/ new splits have no ks_label: convert NaN to None
        cluster_info_df.fillna(value='', inplace=True)  # returns None
    except AttributeError:
        cluster_info_df['curated'] = 0
        cluster_info_df['group'] = np.nan

    # Format columns
    cluster_info_df['bc_label'] = cluster_info_df['bc_label'].str.lower()

    # Get valid cluster indices only based on automatic curation
    valid_cluster_ids = cluster_info_df.index
    cluster_info_df_sub = cluster_info_df

    # Add cluster information
    unit_table['cluster_id'] = cluster_info_df_sub['cluster_id']
    unit_table['peak_channel'] = cluster_info_df_sub['ch']
    #unit_table['depth'] = cluster_info_df_sub['depth']
    unit_table['ks_label'] = cluster_info_df_sub['ks_label']  # "KSLabel" is the KS raw label
    unit_table['group'] = cluster_info_df_sub['group']  # "group" is the Phy-curated label
    unit_table['bc_label'] = cluster_info_df_sub['bc_label']  # automatic curation from bombcell
    unit_table['firing_rate'] = cluster_info_df_sub['fr']

    # Load spikes times
    sync_spike_time_file = os.path.join(imec_folder, f"{imec_folder.name}_{ks_folder_name}_spike_times_sec_sync.npy")
    spike_times_sync = np.load(sync_spike_time_file)
    spike_times_sync_df = pd.DataFrame(data=spike_times_sync, columns=['spike_times'])
    spike_times_sync_df.index.name = 'spike_id'

    # Load spike cluster assignments
    spike_clusters = np.load(kilosort_output / 'spike_clusters.npy')
    spike_clusters_df = pd.DataFrame(data=spike_clusters, columns=['cluster_id'])
    spike_clusters_df.index.name = 'spike_id'

    # Note: Iterate over clusters #TODO: keep for now
    #for c_id in cluster_info_df.cluster_id.values:
    #    spike_ids = spike_clusters_df[spike_clusters_df.cluster_id == c_id].index
    #    try:
    #        spike_times_per_cluster.append(np.array(spike_times_sync_df.iloc[spike_ids].spike_times))
    #    except:
    #        print('Error with cluster {} - check kilosort output'.format(c_id))
    #        spike_times_per_cluster.append(np.array([]))
    #cluster_info_df['spike_times'] = spike_times_per_cluster

    # Group spike times by cluster once
    spike_times_by_cluster = spike_times_sync_df.groupby(spike_clusters_df['cluster_id'])['spike_times'].apply(np.array)
    spike_times_per_cluster = [
        spike_times_by_cluster.get(c_id, np.array([]))
        for c_id in cluster_info_df.cluster_id.values
    ]
    cluster_info_df['spike_times'] = spike_times_per_cluster

    unit_table['spike_times'] = cluster_info_df.loc[valid_cluster_ids].spike_times

    # -----------------------------------------
    # Load bombcell quality metrics
    # -----------------------------------------
    if ks_folder_name == 'kilosort4':
        bc_file_path = kilosort_output / 'bombcell' / 'templates._bc_qMetrics.parquet'
    else:
        bc_file_path = kilosort_output / 'qMetrics' / 'templates._bc_qMetrics.parquet'
    bc_info_df = pd.read_parquet(bc_file_path)

    #try: # TODO: keep for now
    #    bc_info_df_sub = bc_info_df.loc[valid_cluster_ids, :]
   # except KeyError:
   #     print('Error with valid cluster indices - check kilosort/bombcell output.')
   #     valid_cluster_ids_temp = [idx for idx in valid_cluster_ids if idx in bc_info_df.index]
   #     bc_info_df_sub = bc_info_df.loc[valid_cluster_ids_temp, :]


    # Add bombcell quality metrics — merge on cluster_id
    bc_cols = [
        'phy_clusterID',
        'maxChannels',
        'useTheseTimesStart',
        'useTheseTimesStop',
        'percentageSpikesMissing_gaussian',
        'percentageSpikesMissing_symmetric',
        'fractionRPVs_estimatedTauR',
        'presenceRatio',
        'maxDriftEstimate',
        'cumDriftEstimate',
        'nSpikes',
        'nPeaks',
        'nTroughs',
        'waveformDuration_peakTrough',
        'spatialDecaySlope',
        'waveformBaselineFlatness',
        'rawAmplitude',
        'signalToNoiseRatio',
        'Lratio',
        'isolationDistance',
    ]
    bc_info_df_sub = bc_info_df[bc_cols].rename(columns={'phy_clusterID': 'cluster_id'})
    unit_table = unit_table.merge(bc_info_df_sub, on='cluster_id', how='left')

    # -----------------------------------------------------
    # Load mean waveforms and waveform metrics from C_Waves
    # -----------------------------------------------------

    mean_wfs = np.load(kilosort_output / 'cwaves' / 'mean_waveforms.npy')
    peak_channels = cluster_info_df_sub.loc[valid_cluster_ids, 'ch'].values
    mean_wfs = mean_wfs[valid_cluster_ids, peak_channels, :]  # note: keep only valid clusters and peak channels
    unit_table['waveform_mean'] = pd.DataFrame(mean_wfs).to_numpy().tolist()

    #median_wfs = np.load(kilosort_output / 'cwaves' / 'median_peak_waveforms.npy')
    #median_wfs = median_wfs[valid_cluster_ids, :]
    #unit_table['waveform_peak_median'] = pd.DataFrame(median_wfs).to_numpy().tolist()

    # Load mean waveform metrics — merge on cluster_id, probe unique
    mean_wf_metrics = pd.read_csv(kilosort_output / 'cwaves' / 'waveform_metrics.csv')
    mean_wf_metrics['cluster_id'] = cluster_info_df_sub.loc[valid_cluster_ids, 'cluster_id'].values
    unit_table = unit_table.merge(mean_wf_metrics[['cluster_id', 'duration', 'pt_ratio']], on='cluster_id', how='left')

    if DEBUG_PLOT:
        # -------------------------------------------------------
        # Waveform cross-reference plot
        # 4 sources per cluster:
        #   - CWaves mean (82 pts)          -> scaled to ms
        #   - CWaves median (82 pts)        -> scaled to ms
        #   - BC raw (61 pts)          -> scaled to ms
        # All normalized to [-1, 1] for shape comparison.
        # -------------------------------------------------------
        SR = 30000  # Hz
        n_plot = min(36, len(unit_table))

        #Only sample from good/mua waveforms
        unit_table_sub = unit_table[unit_table.bc_label.isin(['mua','good'])]
        sample_idx = np.sort(np.random.choice(len(unit_table_sub), n_plot, replace=False))

        ncols = 6
        nrows = int(np.ceil(n_plot / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5), sharex=True, sharey=False)
        axes = axes.flatten()

        def norm(w):
            """Normalize waveform to [-1, 1] for shape comparison."""
            rng = np.max(np.abs(w))
            return w / rng if rng > 0 else w

        def baseline_correct(w, n_baseline=10):
            """Remove DC offset using first n_baseline samples as baseline."""
            return w - np.mean(w[:n_baseline])

        def scale_uv(w, raw_amplitude):
            """Scale normalized waveform to µV using Bombcell rawAmplitude."""
            #rng = np.max(np.abs(w))
            #if rng > 0:
            #    return w / rng * float(raw_amplitude)
            return w

        def to_ms(n_pts):
            """Convert sample indices to ms timeline centered midpoint, which is the peak time detection."""
            return (np.arange(n_pts) - n_pts // 2) / SR * 1000

        wf_sources = [
            ('waveform_mean', 'CWaves mean', 'steelblue', '-', 1.5), #cwaves
            #('waveform_peak_median', 'CWaves median', 'tomato', '-', 1.5),
            ('waveform_bc_raw', 'Bombcell mean', 'forestgreen', '-', 1.5),
        ]

        for plot_i, unit_i in enumerate(sample_idx):
            ax = axes[plot_i]

            cid = unit_table_sub['cluster_id'].iloc[unit_i]
            label = unit_table_sub['bc_label'].iloc[unit_i]
            raw_amp = unit_table_sub['rawAmplitude'].iloc[unit_i]  # µV, from Bombcell

            for col, src_label, color, ls, lw in wf_sources:
                wf = np.array(unit_table_sub[col].iloc[unit_i])
                if wf.ndim != 1 or len(wf) == 0:
                    continue
                t = to_ms(len(wf))
                ax.plot(t, wf, color=color, lw=lw, linestyle=ls, label=src_label, alpha=0.6)

            ax.axhline(0, color='grey', lw=0.3, linestyle=':')
            ax.axvline(0, color='grey', lw=0.3, linestyle=':')
            ax.set_xlabel('Time (ms)', fontsize=4.5)
            ax.set_ylabel(r'Amplitude ($\mu$V)', fontsize=4.5)
            ax.set_title(f'cluster {cid} | {label}', fontsize=4.5, pad=2)
            ax.tick_params(labelsize=3.5, length=1.5, width=0.4)
            for spine in ax.spines.values():
                spine.set_linewidth(0.3)

        for ax in axes[n_plot:]:
            ax.set_visible(False)

        handles = [
            plt.Line2D([0], [0], color=color, lw=lw, linestyle=ls, label=lbl)
            for _, lbl, color, ls, lw in wf_sources
        ]
        fig.legend(handles=handles, fontsize=4.5, loc='lower center',
                   ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(
            f'Waveform examples',
            fontsize=7
        )
        plt.tight_layout()
        fig.subplots_adjust()
        fig_path = kilosort_output / 'waveforms_crossref_sample.png'
        plt.savefig(fig_path, dpi=400)
        plt.close()

    return unit_table


def add_ccf_parent_info(df, path_to_atlas, ccf_id_col):
    """
    For each entry with ccf_id, add its parent structure info (id, acronym, name).
    :param df: pd.DataFrame with a ccf_id column
    :param path_to_atlas: config dictionary
    :param ccf_id_col: name of the column with ccf_id
    :return: area_table with added parent structure columns
    """

    # Load structures data
    with open(os.path.join(path_to_atlas, 'structures.json')) as f:
        structures = json.load(f)

    is_atlas_space = True if ccf_id_col=='ccf_atlas_id' else False
    df[ccf_id_col] = pd.to_numeric(df[ccf_id_col], errors='coerce')
    df[ccf_id_col] = df[ccf_id_col].fillna(997).astype(int)
    ccf_ids = df[ccf_id_col].values

    # Create a quick lookup for structures by ID
    structures_by_id = {s['id']: s for s in structures}
    structures_by_id.update({0: {'acronym':'void', 'id':0, 'name':'void', 'rgb_triplet':[255,255,255], 'structure_id_path':[0]}}) #from IBL GUI

    # Determine each CCF region's parent ID (or root/void = 997)
    # Like above but also root and void
    parent_ids = {
        ccf_id: (structures_by_id[ccf_id]['structure_id_path'][-2]
                 if structures_by_id[ccf_id]['name'] not in ['root', 'void'] else ccf_id)
        for ccf_id in ccf_ids
    }

    # Build parent lookups
    parent_info = {
        pid: structures_by_id[pid] for pid in set(parent_ids.values())
    }

    # Add missing column for ccf_atlas_name
    if is_atlas_space:
        df['ccf_atlas_name'] = [structures_by_id[ccf_id]['name'] for ccf_id in ccf_ids]

    # Add to table
    if is_atlas_space:
        df['ccf_atlas_parent_id'] = [parent_info[parent_ids[ccf_id]]['id'] for ccf_id in ccf_ids]
        df['ccf_atlas_parent_acronym'] = [parent_info[parent_ids[ccf_id]]['acronym'] for ccf_id in ccf_ids]
        df['ccf_atlas_parent_name'] = [parent_info[parent_ids[ccf_id]]['name'] for ccf_id in ccf_ids]
    else:
        df['ccf_parent_id'] = [parent_info[parent_ids[ccf_id]]['id'] for ccf_id in ccf_ids]
        df['ccf_parent_acronym'] = [parent_info[parent_ids[ccf_id]]['acronym'] for ccf_id in ccf_ids]
        df['ccf_parent_name'] = [parent_info[parent_ids[ccf_id]]['name'] for ccf_id in ccf_ids]

    return df

def fill_missing_ccf_coords(df,
                            axial_col='axial',
                            lateral_col='lateral',
                            coord_cols=('x', 'y', 'z'),
                            shank_col=None,
                            method='nearest',
                            k=1,
                            max_distance=None):
    """
    Fill missing CCF anatomical coordinates (x, y, z) in a dataframe
    using nearest neighbor or weighted interpolation based on physical
    probe coordinates (axial, lateral).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least axial, lateral, and optionally shank columns.
    axial_col : str, default='axial'
        Name of the column representing axial (depth) coordinates in microns.
    lateral_col : str, default='lateral'
        Name of the column representing lateral coordinates in microns.
    coord_cols : tuple, default=('x', 'y', 'z')
        Names of the columns containing anatomical CCF coordinates to be filled.
    shank_col : str or None, default=None
        If provided, ensures that nearest neighbors are only searched within the same shank.
    method : str, {'nearest', 'weighted'}, default='nearest'
        - 'nearest': Assigns missing values from the closest known neighbor.
        - 'weighted': Interpolates using inverse-distance weighting from k neighbors.
    k : int, default=1
        Number of neighbors to consider when `method='weighted'`.
    max_distance : float or None, default=None
        Maximum allowed distance (in microns) for neighbor assignment.
        If None, no distance cutoff is applied.

    Returns
    -------
    pd.DataFrame
        A copy of the dataframe with missing `x, y, z` filled in where possible.
    """

    df = df.copy()  # Avoid modifying original

    # Check columns exist
    required_cols = [axial_col, lateral_col] + list(coord_cols)
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from dataframe.")

    # Separate known and missing entries
    known = df.dropna(subset=coord_cols).copy()
    missing = df[df[list(coord_cols)].isnull().any(axis=1)].copy()

    if known.empty:
        raise ValueError("No known CCF coordinates available for interpolation.")
    if missing.empty:
        return df  # Nothing to fill

    # Prepare storage for updated coordinates
    filled_coords = missing[list(coord_cols)].copy()

    # Helper: build KDTree per shank if needed
    def build_tree(sub_df):
        coords = sub_df[[axial_col, lateral_col]].values
        return cKDTree(coords), coords

    # Function to fill for one subset (either per shank or all data)
    def process_subset(miss_subset, known_subset):
        tree, _ = build_tree(known_subset)

        query_coords = miss_subset[[axial_col, lateral_col]].values

        if method == 'nearest':
            # k=1 is enforced here
            distances, indices = tree.query(query_coords, k=1)
            new_vals = known_subset.iloc[indices][list(coord_cols)].values

            if max_distance is not None:
                mask = distances <= max_distance
                new_vals[~mask] = np.nan  # Set too-far matches to NaN

        elif method == 'weighted':
            distances, indices = tree.query(query_coords, k=k)

            # Handle case where k=1 to avoid shape issues
            if k == 1:
                distances = distances[:, np.newaxis]
                indices = indices[:, np.newaxis]

            weights = 1.0 / (distances + 1e-9)  # Avoid divide-by-zero
            weights /= weights.sum(axis=1, keepdims=True)  # Normalize

            known_vals = known_subset.iloc[indices.flatten()][list(coord_cols)].values
            known_vals = known_vals.reshape(indices.shape[0], indices.shape[1], len(coord_cols))

            new_vals = np.sum(known_vals * weights[..., np.newaxis], axis=1)

            if max_distance is not None:
                # Check distance of closest neighbor
                too_far = distances[:, 0] > max_distance
                new_vals[too_far] = np.nan

        else:
            raise ValueError("Invalid method. Use 'nearest' or 'weighted'.")

        return new_vals

    # Process either per shank or globally
    if shank_col and shank_col in df.columns:
        for shank_id in missing[shank_col].unique():
            miss_subset = missing[missing[shank_col] == shank_id]
            known_subset = known[known[shank_col] == shank_id]
            if known_subset.empty:
                continue
            filled_coords.loc[miss_subset.index] = process_subset(miss_subset, known_subset)
    else:
        filled_coords[:] = process_subset(missing, known)

    # Merge back into the dataframe
    df.loc[filled_coords.index, list(coord_cols)] = filled_coords
    return df

def linear_interpolate_coords(df,
                              axial_col='axial',
                              coord_cols=['x', 'y', 'z'],
                              shank_col=None):
    """
    Linearly interpolate missing CCF coordinates (x, y, z) along the axial axis of the probe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing axial positions and CCF coordinates (x, y, z).
    axial_col : str
        Column name for the axial positions (depth along probe).
    coord_cols : tuple
        Columns to interpolate.
    shank_col : str or None
        If provided, interpolation is done separately for each shank.

    Returns
    -------
    pd.DataFrame
        Copy of dataframe with missing coordinates filled by linear interpolation.
    """
    df_out = df.copy()

    if shank_col and shank_col in df.columns:
        # Interpolate per shank
        for shank_id, group in df_out.groupby(shank_col):
            df_out.loc[group.index, coord_cols] = group.sort_values(axial_col).interpolate(
                method='linear', axis=0, limit_direction='both')[coord_cols]
    else:
        # Interpolate globally along axial axis
        df_out = df_out.sort_values(axial_col)
        df_out[coord_cols] = df_out[coord_cols].interpolate(
            method='linear', axis=0, limit_direction='both')[coord_cols]

    return df_out


def build_area_table(config_file, imec_folder, experimenter=None):
    """
    Build area table from brainreg output.
    Args:
        config_file: path to config file
        imec_folder: path to imec folder processed neural data
        experimenter: name of experimenter

    Returns:

    """

    # Read config file
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # -----------------------------------------
    # Load ccf probe track areas (sample space)
    # -----------------------------------------

    imec_id = imec_folder[-1]
    mouse_name = config['subject_metadata']['subject_id']
    path_to_proc_anat = server_paths.get_anat_probe_track_folder(config_file, experimenter=experimenter)
    path_to_sample_space_track_folder = pathlib.Path(path_to_proc_anat) / 'sample_space' / 'tracks'
    if (path_to_sample_space_track_folder / config['session_metadata']['session_id']).exists():
        path_to_sample_space_track_folder = path_to_sample_space_track_folder / config['session_metadata']['session_id']
    path_to_sample_space_track = path_to_sample_space_track_folder / 'imec{}.csv'.format(imec_id)
    area_table = pd.read_csv(path_to_sample_space_track)

    # -------------------------------------------------------
    # Format table content and match electrodes to table rows
    # -------------------------------------------------------

    # Format table for future shank row matching
    area_table.rename(columns={'Position':'shank_row',
                               'Distance from first position [um]':'distance',  # brainreg-segmentation output update
                               'Index':'shank_row', # brainreg-segmentation output update
                               'Region ID': 'ccf_id',
                               'Region acronym': 'ccf_acronym',
                               'Region name': 'ccf_name'}, inplace=True)

    # Set outside brain points to atlas root
    area_table['ccf_id'] = pd.to_numeric(area_table['ccf_id'], errors='coerce')
    area_table.loc[area_table['ccf_id'] == 'Not found in brain', 'ccf_id'] = 997
    area_table.loc[area_table['ccf_acronym'] == 'Not found in brain', 'ccf_acronym'] = 'root'
    area_table.loc[area_table['ccf_name'] == 'Not found in brain', 'ccf_name'] = 'root'

    # Reverse order of rows (from probe tip upwards)
    area_table = area_table.iloc[::-1]  # reverse order (from probe tip upwards)
    area_table = area_table.iloc[9:, :]  # remove first 9 rows (probe tip)

    # Make values start at 0 to match probe geometry
    max_position = np.max(area_table['shank_row'].values)
    area_table['shank_row'] = max_position - area_table['shank_row'].values  # make values start at 0

    # ------------------------------------------------------------
    # Simplify CCF hierarchical nomenclature with parent structure
    # Relevant for cortical layers <-> cortical area
    # ------------------------------------------------------------

    # Get path to atlas metadata for hierarchy information
    path_to_atlas = config['ephys_metadata']['path_to_atlas']

    atlas_name = pathlib.PureWindowsPath(path_to_atlas).name
    path_to_atlas = os.path.join(server_paths.get_analysis_root(), 'Axel_Bisi', 'Anatomy', atlas_name)
    if not os.path.exists(path_to_atlas):
        print(f'{path_to_atlas} does not exist- check or add it at location.')

    # Apply function
    area_table = add_ccf_parent_info(area_table, path_to_atlas, ccf_id_col='ccf_id')

    # -----------------------------------------
    # Load ccf coordinates (ccf atlas space)
    # -----------------------------------------

    path_to_atlas_space_track = os.path.join(path_to_proc_anat, 'atlas_space', 'tracks')
    coords = np.load(os.path.join(path_to_atlas_space_track, 'imec{}.npy'.format(imec_id)))
    coords = coords[::-1] #from tip to superficial
    coords = coords[9:, :] # remove tip-length (no recording sites)

    area_table['ccf_ap'] = coords[:, 0]
    area_table['ccf_ml'] = coords[:, 2] # nota bene
    area_table['ccf_dv'] = coords[:, 1]

    return area_table
