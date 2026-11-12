import os
import yaml
import numpy as np
import pandas as pd

from scipy.signal import find_peaks
from utils.server_paths import get_dlc_file_path
from utils.dlc_utils import *

from pynwb.base import TimeSeries
from pynwb.behavior import BehavioralEvents, BehavioralTimeSeries, BehavioralEpochs


def convert_dlc_data(nwb_file, config_file, video_timestamps, remove_extra_ts=False):

    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    dlc_file_path = get_dlc_file_path(config_file)

    print("Creating behaviour processing module")
    if 'behavior' in nwb_file.processing:
        bhv_module = nwb_file.processing['behavior']
    else:
        bhv_module = nwb_file.create_processing_module('behavior', 'contains behavioral processed data')

    # Get the behavioral events module (that we will use for discrete data, eg lick times)
    try:
        behavior_events = bhv_module.get(name='BehavioralEvents')
    except KeyError:
        behavior_events = BehavioralEvents(name='BehavioralEvents')
        bhv_module.add_data_interface(behavior_events)

    # Get the behavioral timeseries module (that we will use for continuous data)
    try:
        behavior_t_series = bhv_module.get(name='BehavioralTimeSeries')
    except KeyError:
        behavior_t_series = BehavioralTimeSeries(name='BehavioralTimeSeries')
        bhv_module.add_data_interface(behavior_t_series)

    # Retrieve the multiindex dataframes with the DLC results for top and side views
    side_dlc, top_dlc = get_dlc_dataframe(dlc_file_path)

    # Top camera rotated 180 deg (AB / MH sessions only): on setup 2 the top camera was mounted upside down (nose up,
    # spout from the top) in some sessions (MH006 0123, MH007 0202-0204, MH008 0203 / 0205, MH009 0206-0208, MH010
    # 0207-0209, MH025 0421 / 0422, MH027 0422; checked on video frames by the user). Whether the top keypoints are
    # upside down is decided from the DLC data, not from a session list (user, 2026-11-04): DLC may have been run on the
    # raw (upside-down) video or on a video flipped before inference. Upright, the nose tip is below the nose base in
    # the image (median nose_tip_y - nose_base_y +61..+106 px in setup 1 / setup 2 sessions); upside down it is above
    # (-49..-60 px in the existing DLC of the sessions above). Frames with both nose points at likelihood >= 0.6 are
    # used; a median below -20 px rotates the keypoints back before the kinematics (frames 640 x 480: x' = 639 - x,
    # y' = 479 - y; distances and whisker_angle unchanged, top_nose_angle and raw positions in the setup 1
    # orientation). Fewer than 1% confident frames or |median| <= 20 px: orientation undetermined, not rotated (warning).
    # ibl-ai-agent skills/ssl-valid-data/references/nwb_session_exceptions.md.
    session_id = config['session_metadata']['session_id']
    if session_id.startswith(('AB', 'MH')) and len(top_dlc) > 0 and \
            {'nose_tip_y', 'nose_base_y', 'nose_tip_likelihood', 'nose_base_likelihood'} <= set(top_dlc.columns):
        conf = (top_dlc['nose_tip_likelihood'] >= 0.6) & (top_dlc['nose_base_likelihood'] >= 0.6)
        dy = float(np.nanmedian((top_dlc['nose_tip_y'] - top_dlc['nose_base_y'])[conf])) if conf.mean() >= 0.01 \
            else np.nan
        if np.isfinite(dy) and dy < -20:
            frame_width, frame_height = 640, 480
            for col in top_dlc.columns:
                if col.endswith('_x'):
                    top_dlc[col] = (frame_width - 1) - top_dlc[col]
                elif col.endswith('_y'):
                    top_dlc[col] = (frame_height - 1) - top_dlc[col]
            print(f'{session_id}: top keypoints upside down (median nose tip - base y = {dy:.0f} px over '
                  f'{conf.mean():.0%} confident frames), rotated back 180 deg')
        elif not (np.isfinite(dy) and dy > 20):
            print(f'WARNING {session_id}: top keypoint orientation undetermined (median nose tip - base y = {dy}, '
                  f'{conf.mean():.0%} confident frames); not rotated')

    # If one of the two is missing, stop processing
    if len(side_dlc) == 0 or len(top_dlc) == 0:
        ValueError("DLC data can't be found or needs to be analyzed")

    # Compute kinematics for each of the views.
    if (config['session_metadata']['experimenter'] in ['AB', 'MH']
            or (config['session_metadata']['experimenter'] == 'PB' and config['ephys_metadata']['setup']=='Neuropixels setup 1 AI3209')):
        side_dlc = compute_kinematics_alt(side_dlc, 'sideview')
        top_dlc = compute_kinematics_alt(top_dlc, 'topview')
        pcutoff_tongue = 0.5

    else:
        side_dlc = compute_kinematics(side_dlc, 'sideview')
        top_dlc = compute_kinematics(top_dlc, 'topview')
        pcutoff_tongue = 0.6 #cutoff for tongue/lick events detection

    # Setup 2 camera scale (AB / MH sessions only): the yaml behaviour 'setup' is set per experimenter, so these setup 2
    # sessions (probe sheet joint_probe_insertion_info.xlsx for ephys days; training days from the top / side pixel size of
    # the head, ssl-dlc-utils-validation 004, and the rotated top camera) were converted with the setup 1 px/mm. Only the
    # px/mm reference is changed here (mice_info/reference_setup2.xlsx), not the yaml.
    setup2_sessions = [
        'MH004_20241205_115227', 'MH004_20241206_093158',
        'MH006_20250123_131233',
        'MH007_20250202_165003', 'MH007_20250203_140412', 'MH007_20250204_092159',
        'MH008_20250203_110046', 'MH008_20250205_153729',
        'MH009_20250206_104829', 'MH009_20250207_131953', 'MH009_20250208_192403',
        'MH010_20250207_112153', 'MH010_20250208_204052', 'MH010_20250209_150421',
        'MH025_20250421_115741', 'MH025_20250422_141134',
        'MH027_20250422_111013',
    ]
    if session_id.startswith(('AB', 'MH')) and session_id in setup2_sessions:
        config['behaviour_metadata']['setup'] = 'Neuropixels setup 2 AI3209'
        print(f'{session_id}: setup 2 camera scale (reference_setup2)')
    px_ref = get_reference_from_grid(config)

    # Camera key of each view, as in behavior_to_nwb.py (video files): for AB (and MH) sessions cam1 = top camera and
    # cam2 = side camera (NIDQ cam0_frame_times / cam1_frame_times; checked on both rigs 2026-11-02 with the video
    # start times), for other experimenters cam1 = side and cam2 = top.
    if config['session_metadata']['experimenter'] in ['AB', 'MH']:
        side_cam, top_cam = 'cam2', 'cam1'
    else:
        side_cam, top_cam = 'cam1', 'cam2'

    # Frames vs camera timestamps: a difference of less than 1 s of frames is accepted (frames paired from the first
    # timestamp); a larger one is reported, the camera pulse runs must then be checked (see CAMERA_LAST_PULSE_RUN_SESSIONS
    # in utils/ephys_converter_misc.py)
    for view, dlc, cam in (('side', side_dlc, side_cam), ('top', top_dlc, top_cam)):
        if len(dlc) > 0 and len(video_timestamps.get(cam, [])) > 1:
            cam_ts = [timestamp[0] for timestamp in video_timestamps[cam]]
            n_diff = len(dlc) - len(cam_ts)
            cam_rate = 1 / np.median(np.diff(cam_ts[0:200]))
            if abs(n_diff) >= cam_rate:
                print(f'WARNING: {view} DLC has {len(dlc)} frames vs {len(cam_ts)} {cam} timestamps '
                      f'({n_diff / cam_rate:+.1f} s): check camera pulse runs and video')
        elif len(dlc) > 0:
            print(f'WARNING: {view} DLC data but no {cam} timestamps: the {view} DLC series will have no timestamps')

    # End-aligned DLC (AB / MH sessions only): the camera sent pulses before the first saved video frame, so the last DLC
    # frame belongs to the last camera timestamp; the first (n timestamps - n frames) timestamps have no frame. Evidence
    # (lick-bout-locked tongue / whisker traces and frame offset per session third): ibl-ai-agent
    # projects/ssl-dlc-alignment-debug 006; list in skills/ssl-valid-data/references/nwb_session_exceptions.md.
    end_aligned_dlc_sessions = {
        'MH013_20250218_110530': ['side', 'top'],  # 882 side / 868 top timestamps before the first frame (4.4 s)
        'MH036_20250517_140456': ['side', 'top'],  # 510 side / 515 top timestamps before the first frame (2.6 s)
        'AB127_20240823_094915': ['top'],  # top video 2542 frames longer than the camera pulses: first 2542 frames dropped
        'AB153_20250209_153912': ['side', 'top'],  # 47 timestamps before the first frame (top: from licks, 37 only)
        'AB116_20240726_094206': ['side', 'top'],  # frames dropped in the first 2,175 s only: end-aligned after (5,837 side)
        'MH028_20250503_112316': ['side', 'top'],  # 2,678 side / 2,219 top timestamps before the first frame (camera test run 0-17.5 s); last run from 51.97 s = 511,279 frames
    }
    session_id = config['session_metadata']['session_id']
    end_aligned_views = []
    if session_id.startswith(('AB', 'MH')) and session_id in end_aligned_dlc_sessions:
        end_aligned_views = end_aligned_dlc_sessions[session_id]

    # More frames than timestamps (video started before the camera pulses were recorded): drop the first frames
    for view, cam in (('side', side_cam), ('top', top_cam)):
        n_ts = len(video_timestamps.get(cam, []))
        dlc = side_dlc if view == 'side' else top_dlc
        if view in end_aligned_views and len(dlc) > n_ts > 0:
            print(f'{session_id}: {view} DLC end-aligned, first {len(dlc) - n_ts} frames dropped')
            dlc = dlc.iloc[len(dlc) - n_ts:].reset_index(drop=True)
            if view == 'side':
                side_dlc = dlc
            else:
                top_dlc = dlc

    reported = set()

    # Frames dropped during the video (AB / MH sessions only): the camera pulse is recorded but the frame is not saved,
    # so the offset (pulse index - DLC frame index) grows in steps along the video. Piecewise mapping per camera:
    # (first DLC frame of the segment, offset), estimated from the lick-locked tongue (side) and the side step times for
    # the top camera (ibl-ai-agent projects/ssl-dlc-alignment-debug 008, figures/<session>_piecewise.png).
    piecewise_dlc_sessions = {
        'MH013_20250219_170437': {
            'side': [(0, 0), (87536, 83), (97045, 105), (116989, 112), (230893, 140), (240443, 145), (260519, 174),
                     (276635, 206)],
            'top': [(0, 0), (85255, 91), (96986, 117), (128716, 126), (231298, 145), (243588, 157), (282378, 214)],
        },
        'MH017_20250302_120918': {
            'side': [(0, 38), (50747, 56), (74383, 71), (95807, 77), (227021, 86), (252402, 91), (260674, 102),
                     (290865, 130)],
            'top': [(0, 38), (56865, 70), (294625, 127)],  # top: 3 steps resolved only (weak top signal)
        },
        'MH004_20241206_093158': {
            'side': [(0, 0), (509751, 55)],
            'top': [(0, 0), (509751, 95)],  # top: own pooled lag change (-17 -> +78) at the side step; checked on heatmaps
        },
    }
    piecewise_views = {}

    # DLC unusable before a frame (AB / MH sessions only): frames dropped continuously at the start of the video, not
    # recoverable; DLC data set to NaN before this frame (frame numbering kept) and DLC licks there dropped. AB116 26/07:
    # end-aligned and exact from frame 434,963 (2,175 s of video; jaw-locked check, ssl-dlc-alignment-debug 008).
    dlc_unusable_before = {
        'AB116_20240726_094206': {'side': 434963, 'top': 434963},
    }
    unusable_before = {}
    if session_id.startswith(('AB', 'MH')) and session_id in dlc_unusable_before:
        unusable_before = dlc_unusable_before[session_id]
    if session_id.startswith(('AB', 'MH')) and session_id in piecewise_dlc_sessions:
        piecewise_views = piecewise_dlc_sessions[session_id]

    def view_timestamps(view, cam, n_frames):
        ts = [timestamp[0] for timestamp in video_timestamps[cam]]
        if view in piecewise_views:
            segments = piecewise_views[view]
            starts = np.array([seg[0] for seg in segments])
            offsets = np.array([seg[1] for seg in segments])
            frames = np.arange(n_frames)
            pulse_idx = frames + offsets[np.searchsorted(starts, frames, side='right') - 1]
            assert pulse_idx[0] >= 0, f'{session_id} {view}: piecewise mapping out of range'
            # the last frames can map past the last recorded pulse: they have no timestamp and are dropped
            pulse_idx = pulse_idx[pulse_idx < len(ts)]
            if view not in reported:
                print(f'{session_id}: {view} DLC piecewise frame -> pulse mapping ({len(segments)} segments, '
                      f'{n_frames - len(pulse_idx)} last frames without pulse dropped)')
                reported.add(view)
            return list(np.asarray(ts)[pulse_idx])
        if view in end_aligned_views and len(ts) > n_frames:
            if view not in reported:
                print(f'{session_id}: {view} DLC end-aligned, first {len(ts) - n_frames} {cam} timestamps dropped')
                reported.add(view)
            ts = ts[len(ts) - n_frames:]
        return ts

    print('Adding DLC time series and events to NWB file...')
    if len(side_dlc)>0:
        for name, data in side_dlc.items():
            ts = view_timestamps('side', side_cam, len(data))
            data = data.iloc[:len(ts)] if len(ts) < len(data) and 'side' in piecewise_views else data
            if 'side' in unusable_before:
                data = data.astype(float).copy()
                data.iloc[:unusable_before['side']] = np.nan
            rate = np.round(1 / np.median(np.diff(ts[0:200])), 2)

            if 'velocity' in name:  # scale frame difference wrt. to sampling rate
                data = data * rate

            if (len(ts) > len(data)) and (remove_extra_ts):
                print(f'Find. {len(ts) - len(data)} more timestamps than dlc data points; remove extra timestamps')
                ts = ts[:len(data)]

            # Add times series for bodybarts
            timeseries = TimeSeries(name=f'{name}',
                                    data=data.to_numpy(),
                                    unit='seconds',
                                    resolution=-1.0,
                                    conversion=[1/px_ref[key].values[0] for key in px_ref.keys() if "side" in key][0],
                                    offset=0.0,
                                    timestamps=ts,
                                    starting_time=None,
                                    rate=None,
                                    comments='no comments',
                                    description=f'no description',
                                    control=None,
                                    control_description=None,
                                    continuity='continuous')

            behavior_t_series.add_timeseries(timeseries)

    if len(top_dlc)>0:
        for name, data in top_dlc.items():
            if 'nose' in name or 'particle' in name:
                name = 'top_' + name

            ts = view_timestamps('top', top_cam, len(data))
            data = data.iloc[:len(ts)] if len(ts) < len(data) and 'top' in piecewise_views else data
            if 'top' in unusable_before:
                data = data.astype(float).copy()
                data.iloc[:unusable_before['top']] = np.nan
            rate = np.round(1 / np.median(np.diff(ts[0:200])), 2)
            if 'velocity' in name: # scale frame difference wrt. to sampling rate
                data = data * rate
            if (len(ts) > len(data)) and (remove_extra_ts):
                print(f'Find. {len(ts) - len(data)} more timestamps than dlc data points; remove extra timestamps')
                ts = ts[:len(data)]
            # Add times series for bodybarts
            timeseries = TimeSeries(name=f'{name}',
                                    data=data.to_numpy(),
                                    unit='seconds',
                                    resolution=-1.0,
                                    conversion=[1/px_ref[key].values[0] for key in px_ref.keys() if "top" in key][0],
                                    offset=0.0,
                                    timestamps=ts,
                                    starting_time=None,
                                    rate=None,
                                    comments='no comments',
                                    description=f'no description',
                                    control=None,
                                    control_description=None,
                                    continuity='continuous')

            behavior_t_series.add_timeseries(timeseries)

    if len(side_dlc)>0:
        ts = [timestamp[0] for timestamp in video_timestamps[side_cam]]
        rate = np.round(1 / np.median(np.diff(ts[0:200])), 2)
        # Add lick times counted as the peaks of tongue distance
        tongue_licks, _ = find_peaks(np.where(side_dlc['tongue_likelihood'] > pcutoff_tongue, side_dlc['tongue_distance'], np.nan), distance= 20)
        if 'side' in unusable_before:
            tongue_licks = tongue_licks[tongue_licks >= unusable_before['side']]

        data_to_store = np.arange(len(tongue_licks))  # data would be lick index
        timestamps_to_store = tongue_licks / rate  # same length as n_licks absolute times of licks
        lick_timeseries = TimeSeries(name=f'tongue_dlc_licks',
                                    data=data_to_store,
                                    unit='seconds',
                                    resolution=-1.0,
                                    conversion=1.0,
                                    offset=0.0,
                                    timestamps=timestamps_to_store,
                                    starting_time=None,
                                    rate=None,
                                    comments='no comments',
                                    description=f'index (data) and timestamps of DLC detected licks',
                                    control=None,
                                    control_description=None,
                                    continuity='instantaneous')

        behavior_events.add_timeseries(lick_timeseries)

        # Add movement times as peaks of jaw angle
        jaw_licks, _ = find_peaks(np.where(side_dlc['jaw_likelihood'] > 0.6, side_dlc['jaw_angle'], np.nan), prominence=side_dlc['jaw_angle'].std() * 1.8)
        if 'side' in unusable_before:
            jaw_licks = jaw_licks[jaw_licks >= unusable_before['side']]

        data_to_store = np.arange(len(jaw_licks))  # data would be lick index
        timestamps_to_store = jaw_licks / rate  # same length as n_licks absolute times of licks
        lick_timeseries = TimeSeries(name=f'jaw_dlc_licks',
                                    data=data_to_store,
                                    unit='seconds',
                                    resolution=-1.0,
                                    conversion=1.0,
                                    offset=0.0,
                                    timestamps=timestamps_to_store,
                                    starting_time=None,
                                    rate=None,
                                    comments='no comments',
                                    description=f'index (data) and timestamps of DLC detected licks',
                                    control=None,
                                    control_description=None,
                                    continuity='instantaneous')

        behavior_events.add_timeseries(lick_timeseries)

    print('Done DLC conversion to NWB.')
    return