"""
Session exception: sessions recorded as two SpikeGLX runs that were joined before sorting (CatGT supercat with
-supercat_trim_edges, one Kilosort 4 sorting on the joined stream). Project ibl-ai-agent ssl-ab155-session-merge
(scripts 010-014 built and validated the test NWB this module reproduces; user 2026-11-06).

AB155_20250208_175103 (day 0):
  part 1 = run g0: passive pre + active (behaviour results_corrected_bis.csv, 239 trials, behaviour log)
  part 2 = run g1: passive post (behaviour results_2.csv, 71 trials, no behaviour log); recorded as a separate behaviour
           session (AB155_20250208_185559), which is not converted on its own.
  The processed Ephys folder holds the joined sorting (run 4819418, DREDge + Kilosort 4 drift correction) and the
  TPrime event times of the joined NIDQ (synced to imec0). The raw NIDQ in Recording is the joined (supercat) NIDQ.

What differs from a normal session (each applied where the converter reads the input):
  1. results: part 1 + part 2 results concatenated, trials renumbered (get_behavior_results_file).
  2. trial TTLs: the part-1 end TTL (last TTL before the junction) is removed; the converter then drops the part-2 end
     TTL as usual, so the 310 results rows pair with the TTLs by index (load_ephys_sync_timestamps).
  3. cameras: the camera on / off pulses (> 50 ms from the neighbouring pulse) are removed in each part, not only at the
     ends of the whole train: part 1 on pulse, part 2 off pulse (load_ephys_sync_timestamps). The DLC files in
     <session>/Video are the part 1 + part 2 DLC concatenated (595,292 + 77,288 rows per camera).
  4. licks: part 1 from the behaviour log, aligned with the part-1 TTL pairs only (the log covers part 1); part 2 from
     the converter's piezo detector on the joined NIDQ, mapped to imec0 through the NIDQ trial TTLs
     (extract_ephys_timestamps). Part 2 (passive post) has no licks.
  5. time base: physical time. Every time at or after the junction is shifted by gap_shift_s (= real start of g1 -
     joined start of g1, from the SpikeGLX fileCreateTime_original of g0 and g1, +- 1 s; 012_insert_gap_nwb.py), epochs
     are recomputed, and the gap is stored as BehavioralEpochs 'recording_gap' and acquisition 'recording_gap'
     (applied to the saved NWB file, convert_data_to_nwb).
"""
import datetime
import os
import tempfile
import uuid

import h5py
import numpy as np
import pandas as pd
import yaml

JOINED_SESSIONS = {
    'AB155_20250208_175103': {
        'part2_results': 'results_2.csv',
        # fileCreateTime_original(g1) - fileCreateTime_original(g0) - joined NIDQ time of g1 t = 0 (012_insert_gap_nwb.py)
        'gap_shift_s': 903.667601,
        'sorting': 'run 4819418 (DREDge + Kilosort 4 drift correction)',
    },
}
CAMERA_ONOFF_PULSE_S = 0.05
TRIAL_TIME_COLS = ('start_time', 'stop_time', 'abort_window_start_time', 'abort_window_stop_time', 'auditory_stim_time',
                   'whisker_stim_time', 'no_stim_time', 'lick_time', 'response_window_start_time',
                   'response_window_stop_time', 'stim_onset', 'opto_stim_start_time', 'opto_stim_stop_time')


def _session_id(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        return yaml.safe_load(stream)['session_metadata']['session_id']


def is_joined_session(session_id):
    return session_id.startswith(('AB', 'MH')) and session_id in JOINED_SESSIONS


def get_junction_times(config_file, experimenter=None):
    """Junction (start of part 2) in the joined clock, from CatGT <run>_sc_offsets.txt: (imec0 time, NIDQ time)."""
    from utils import server_paths
    folder = server_paths.get_ephys_folder(config_file, experimenter=experimenter)
    f = [x for x in os.listdir(folder) if x.endswith('_sc_offsets.txt')][0]
    off = {l.split()[0].rstrip(':'): l.split()[1:] for l in open(os.path.join(folder, f)) if l.strip()}
    return float(off['sec_imap0'][1]), float(off['sec_nidq'][1])


def get_joined_results_file(config_file, part1_results_file):
    """Part 1 + part 2 behaviour results, trials renumbered, written to a temporary csv."""
    session_id = _session_id(config_file)
    part2 = os.path.join(os.path.dirname(part1_results_file), JOINED_SESSIONS[session_id]['part2_results'])
    r1 = pd.read_csv(part1_results_file, sep=None, engine='python')
    r2 = pd.read_csv(part2, sep=None, engine='python')
    assert list(r1.columns) == list(r2.columns), 'results columns differ between parts'
    r = pd.concat([r1, r2], ignore_index=True)
    r['trial_number'] = np.arange(1, len(r) + 1)
    out = os.path.join(tempfile.gettempdir(), f'{session_id}_results_joined.csv')
    r.to_csv(out, index=False)
    print(f'Session exception {session_id} (joined runs): results {len(r1)} + {len(r2)} = {len(r)} trials')
    return out


def fix_joined_sync_event(session_id, event, timestamps, tj):
    """Trial TTLs: drop the part-1 end TTL. Camera pulses: drop the on / off pulse at the ends of each part."""
    x = np.asarray(timestamps, dtype=float)
    if event == 'trial_start_times':
        i_end = int(np.flatnonzero(x < tj)[-1])
        print(f'Session exception {session_id} (joined runs): part-1 end trial TTL #{i_end} at {x[i_end]:.3f} s removed')
        return np.delete(x, i_end)
    if event.endswith('_frame_times'):
        keep = []
        for name, part in (('part 1', x[x < tj]), ('part 2', x[x >= tj])):
            d = np.diff(part)
            lo, hi = int(d[0] > CAMERA_ONOFF_PULSE_S), len(part) - int(d[-1] > CAMERA_ONOFF_PULSE_S)
            print(f'Session exception {session_id} (joined runs) {event} {name}: {len(part)} pulses, '
                  f'{lo} on / {len(part) - hi} off pulse removed')
            keep.append(part[lo:hi])
        return np.concatenate(keep)
    return x


def get_part2_nidq_licks(config_file, nidq_lick_times, tj_nidq, experimenter=None):
    """Part-2 piezo licks (NIDQ detector, NIDQ clock) mapped to imec0 through the NIDQ / TPrime trial TTLs."""
    from scipy.interpolate import interp1d
    from utils import server_paths
    folder = server_paths.get_ephys_folder(config_file, experimenter=experimenter)
    nidq_ttl = np.loadtxt(os.path.join(folder, [f for f in os.listdir(folder) if f.endswith('nidq.xa_1_0.txt')][0]))
    imec_ttl = np.loadtxt(os.path.join(server_paths.get_sync_event_times_folder(config_file, experimenter=experimenter),
                                       'trial_start_times.txt'))
    assert len(nidq_ttl) == len(imec_ttl), 'NIDQ and TPrime trial TTL counts differ'
    nl = np.asarray(nidq_lick_times, dtype=float).ravel()
    to_imec = interp1d(nidq_ttl, imec_ttl, bounds_error=False, fill_value='extrapolate')
    p2 = np.sort(to_imec(nl[nl >= tj_nidq]))
    print(f'Session exception {_session_id(config_file)} (joined runs): {len(p2)} part-2 licks (NIDQ piezo detector)')
    return p2


def _shift_ds(ds, tj, shift):
    x = ds[()]
    if x.dtype.kind in 'SO':
        v = np.array([float(s.decode() if isinstance(s, bytes) else s) for s in x])
        m = v >= tj
        v[m] += shift
        ds[...] = np.array([repr(float(s)).encode() for s in v], dtype=object)
        return int(m.sum())
    m = np.isfinite(x) & (x >= tj)
    if m.any():
        x[m] += shift
        ds[...] = x
    return int(m.sum())


def add_recording_gap_annotation(nwb_file, tj):
    """Acquisition AnnotationSeries 'recording_gap' (part 1 end, part 2 start); both at the junction in the joined
    clock, the part-2 start is set by insert_recording_gap after saving."""
    from pynwb.misc import AnnotationSeries
    nwb_file.add_acquisition(AnnotationSeries(
        name='recording_gap', data=np.array(['part 1 end (run g0, trimmed edge)',
                                             'part 2 start (run g1, passive post, trimmed edge)']),
        timestamps=np.array([tj, tj]), description='joined session: recording gap between the two runs'))


def insert_recording_gap(nwb_path, session_id, tj):
    """Shift every time at or after the junction tj by the real gap; recompute epochs; add the 'recording_gap' epoch;
    set the 'recording_gap' annotation (added before saving by add_recording_gap_annotation). Applied in place to the
    saved NWB file (h5py: pynwb append fails on the camera series names)."""
    from utils.behavior_converter_misc import get_passive_active_epoch_bounds
    shift = JOINED_SESSIONS[session_id]['gap_shift_s']
    print(f'Session exception {session_id} (joined runs): inserting the {shift:.3f} s recording gap at {tj:.3f} s')
    with h5py.File(nwb_path, 'r+') as f:
        n = 0
        for c in TRIAL_TIME_COLS:
            if f'intervals/trials/{c}' in f:
                n += _shift_ds(f[f'intervals/trials/{c}'], tj, shift)
        ts_paths = []
        f.visititems(lambda name, o: ts_paths.append(name) if isinstance(o, h5py.Dataset)
                     and name.endswith('/timestamps') and 'BehavioralEpochs' not in name
                     and 'acquisition/recording_gap' not in name else None)
        for name in ts_paths:
            n += _shift_ds(f[name], tj, shift)
        for name in ('processing/behavior/BehavioralEvents/piezo_lick_times/data', 'units/spike_times',
                     'units/useTheseTimesStart', 'units/useTheseTimesStop'):
            if name in f:
                n += _shift_ds(f[name], tj, shift)
        print(f'  shifted {n} values')
        tr = f['intervals/trials']
        t = pd.DataFrame({'start_time': tr['start_time'][()], 'stop_time': tr['stop_time'][()],
                          'context': [x.decode() if isinstance(x, bytes) else x for x in tr['context'][()]]})
        ep = 'processing/behavior/BehavioralEpochs'
        for name, (a, b) in get_passive_active_epoch_bounds(t).items():
            f[f'{ep}/{name}/timestamps'][...] = np.array([a, b])
            print(f'  epoch {name}: {a:.3f}-{b:.3f}')
        f.copy(f'{ep}/passive_pre', f'{ep}/recording_gap')
        g = f[f'{ep}/recording_gap']
        g['timestamps'][...] = np.array([tj, tj + shift])
        g.attrs['object_id'] = str(uuid.uuid4())
        g.attrs['description'] = (
            'No recording: between the end of run g0 (part 1) and the start of run g1 (part 2, passive post). Exclude '
            'from time-based metrics (e.g. presence / coverage ratio, firing rates over the session).')
        print(f'  epoch recording_gap: {tj:.3f}-{tj + shift:.3f}')
        g = f['acquisition/recording_gap']
        g['timestamps'][...] = np.array([tj, tj + shift])
        g.attrs['description'] = (
            f'Joined session (CatGT supercat of runs g0 and g1, -supercat_trim_edges). Times are physical: seconds from '
            f'the start of g0. Sorting ran on the joined stream, where part 2 started at {tj:.6f} s (imec0); every time '
            f'from part 2 was shifted by {shift:.3f} s (+- 1 s, from the SpikeGLX file creation times of g0 and g1). '
            f'No recording between the two timestamps. Sorting: {JOINED_SESSIONS[session_id]["sorting"]}.')
    print(f'  done ({datetime.datetime.now():%H:%M:%S})')
