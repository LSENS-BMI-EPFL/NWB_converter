#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: NWB_converter
@file: ephys_to_nwb.py
"""

import os
import json
import pathlib
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import read_sglx
from utils import readSLGX
from utils.ephys_converter_misc import ( NP_PROBE_TYPE_MAP,
    build_unit_table,
    build_area_table,
    create_electrode_table,
    create_simplified_unit_table,
    create_unit_table,
    get_probe_insertion_info,
    get_target_location,
    add_ccf_parent_info,
)
from utils.server_paths import get_imec_probe_folder_list, get_sync_event_times_folder, get_raw_ephys_folder
from utils.sglx_meta_to_coords import MetaToCoords, readMeta
from pynwb.ecephys import ElectricalSeries, LFP

DEBUG_PLOT = True

def load_ibl_channel_locations(path_channel_loc, bregma_xyz=None):
    """
    Parse IBL ephys-atlas channel_locations.json into a tidy DataFrame.
    Channel indices are read directly from JSON keys (channel_0, channel_1, ...).
    Coordinates are transformed from bregma-relative to absolute CCF voxel space.

    IBL convention: x=ML, y=AP, z=DV (bregma-relative, in um).
    Transform: ML = x + bregma[0], AP = -y + bregma[1], DV = -z + bregma[2].

    Args:
        path_channel_loc: path to channel_locations.json
        bregma_xyz: optional override for bregma coords [ML, AP, DV]; read from JSON if None

    Returns:
        pd.DataFrame with columns: ch_id, peak_channel, axial, lateral,
        ccf_atlas_ml, ccf_atlas_ap, ccf_atlas_dv, ccf_atlas_id, ccf_atlas_acronym
    """
    with open(path_channel_loc, 'r') as f:
        data = json.load(f)

    if bregma_xyz is None:
        bregma_xyz = np.array(data['origin']['bregma']).astype(float)

    rows = []
    for key, val in data.items():
        if key == 'origin':
            continue
        ch_id = int(key.split('_')[-1])
        rows.append({
            'ch_id':             ch_id,
            'peak_channel':      ch_id,
            'axial':             float(val['axial']),
            'lateral':           float(val['lateral']),
            'ccf_atlas_ml':      float(val['x']) + bregma_xyz[0],   # IBL x = ML
            'ccf_atlas_ap':     -float(val['y']) + bregma_xyz[1],   # IBL y = AP
            'ccf_atlas_dv':     -float(val['z']) + bregma_xyz[2],   # IBL z = DV
            'ccf_atlas_id':      int(val['brain_region_id']),
            'ccf_atlas_acronym': str(val['brain_region']),
        })

    return pd.DataFrame(rows)


def fill_missing_channels(ephys_align_df, xcoords, ycoords, n_channels=384):
    """
    Add rows for probe channels absent from the IBL alignment JSON
    (i.e. channels above the brain surface), then fill missing anatomical
    labels and coordinates.

    Region labels (ccf_atlas_acronym, ccf_atlas_id) are filled by forward/back-fill
    along the sorted axial axis (equivalent to nearest-neighbor along the shank).
    Coordinates (ccf_atlas_ml/ap/dv) are linearly interpolated then ffill/bfill.

    Args:
        ephys_align_df: DataFrame from load_ibl_channel_locations
        xcoords: array of probe lateral coordinates (um), shape (384,)
        ycoords: array of probe axial coordinates (um), shape (384,)
        n_channels: total number of probe channels (default 384 for NP1)

    Returns:
        DataFrame with one row per channel (0 to n_channels-1), all columns filled.
    """
    matched_ch_ids = set(ephys_align_df['ch_id'].values)
    missing_ch_ids = [ch for ch in range(n_channels) if ch not in matched_ch_ids]

    if missing_ch_ids:
        missing_df = pd.DataFrame({
            'ch_id':        missing_ch_ids,
            'peak_channel': missing_ch_ids,
            'axial':        ycoords[missing_ch_ids],
            'lateral':      xcoords[missing_ch_ids],
        })
        ephys_align_df = pd.concat([ephys_align_df, missing_df], ignore_index=True)

    # Sort by axial depth for sensible fill direction
    ephys_align_df = ephys_align_df.sort_values('axial').reset_index(drop=True)

    # Fill region labels by nearest neighbor (ffill then bfill along axial axis)
    ephys_align_df[['ccf_atlas_acronym', 'ccf_atlas_id']] = (
        ephys_align_df[['ccf_atlas_acronym', 'ccf_atlas_id']].ffill().bfill()
    )

    # Fill coordinates by linear interpolation, then ffill/bfill for edge channels
    coord_cols = ['ccf_atlas_ml', 'ccf_atlas_ap', 'ccf_atlas_dv']
    ephys_align_df[coord_cols] = (
        ephys_align_df[coord_cols]
        .interpolate(method='linear')
        .ffill()
        .bfill()
    )

    return ephys_align_df


def plot_channel_map(imec_id, xcoords, ycoords, channel_map,
                     area_table, ephys_align_df, imec_folder):

    shank_rows = np.divide(ycoords, 20)

    # Build area label lists for both panels first
    sample_acronyms = []
    for ch in range(384):
        row_id = int(shank_rows[ch])
        if row_id in area_table.index and not pd.isna(area_table.loc[row_id, 'ccf_acronym']):
            sample_acronyms.append(str(area_table.loc[row_id, 'ccf_acronym']))
        else:
            sample_acronyms.append('unknown')

    atlas_acronyms = []
    for ch in range(384):
        row = ephys_align_df[ephys_align_df['ch_id'] == ch]
        atlas_acronyms.append(
            str(row.iloc[0]['ccf_atlas_acronym']) if len(row) > 0 else 'unknown'
        )

    # Unified color map across both panels
    all_areas = sorted(set(sample_acronyms) | set(atlas_acronyms))
    cmap = plt.cm.get_cmap('tab20', len(all_areas))
    color_map = {a: cmap(i) for i, a in enumerate(all_areas)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 12), sharey=True)
    fig.suptitle(f'Probe IMEC{imec_id} — Channel Map', fontsize=13)

    for ax, acronyms, title in zip(
        axes,
        [sample_acronyms, atlas_acronyms],
        ['Sample space (brainreg)', 'Atlas space (IBL alignment)'],
    ):
        ax.set_title(title, fontsize=10)

        for ch in range(384):
            marker = 's' if ch in channel_map else 'x'
            ax.scatter(xcoords[ch], ycoords[ch],
                       color=color_map[acronyms[ch]],
                       marker=marker, s=18, linewidths=0.5)

        prev_area = None
        for ch in range(384):
            area = acronyms[ch]
            if area != prev_area:
                ax.axhline(ycoords[ch], color='k', linewidth=0.4, linestyle='--', alpha=0.5)
                ax.text(70, ycoords[ch] + 2, area, fontsize=5, color='k', va='bottom')
                prev_area = area

        ax.set_xlabel('X (um)')
        ax.set_xlim(-10, 80)

    axes[0].set_ylabel('Y (um)')

    # Single shared legend for both panels
    patches = [mpatches.Patch(color=color_map[a], label=a) for a in all_areas]
    fig.legend(handles=patches, fontsize=6, loc='lower center', ncol=6,
               title='Brain area', title_fontsize=7,
               bbox_to_anchor=(0.5, 0.02))

    fig.text(0.5, 0.005,
             'square = active channel (in KS channel map)  |  x = excluded channel',
             ha='center', fontsize=7)
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    fig_path = pathlib.Path(imec_folder, f'probe_imec{imec_id}_channel_map_annotated.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return


def convert_ephys_recording(nwb_file, config_file, add_recordings=False, experimenter=None):
    """
    Converts ephys recording to NWB file.

    Anatomical localization uses two sources:
      - Sample-space (brainreg histology track): ccf_* columns, secondary/cross-check.
      - Atlas-space (IBL ephys-atlas GUI alignment): ccf_atlas_* columns, primary.

    Args:
        nwb_file: NWBFile object
        config_file: path to session YAML config file
        add_recordings: whether to add LFP data to NWB file
        experimenter: experimenter initials string
    """
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    # Create dynamic tables
    create_electrode_table(nwb_file=nwb_file)
    if config.get('ephys_metadata').get('unit_table') == 'simple':
        create_simplified_unit_table(nwb_file=nwb_file)
    else:
        create_unit_table(nwb_file=nwb_file)

    electrode_counter = 0
    neuron_counter = 0

    imec_probe_list = get_imec_probe_folder_list(config_file=config_file, experimenter=experimenter)

    for _, imec_folder in enumerate(imec_probe_list):
        imec_id = int(pathlib.Path(imec_folder).stem[-1])
        print(f'\nProbe IMEC{imec_id}: {imec_folder}')

        # ------------------
        # Validate probe
        # ------------------
        probe_info_df = get_probe_insertion_info(config_file=config_file)
        mouse_name = config.get('subject_metadata').get('subject_id')
        probe_info = probe_info_df[
            (probe_info_df['mouse_name'] == mouse_name) &
            (probe_info_df['probe_id'] == imec_id)
        ]
        is_valid_probe = probe_info['valid'].values[0]
        if not is_valid_probe:
            print(f'Skipping {mouse_name} IMEC{imec_id}: invalid recording.')
            continue

        # ------------------
        # Probe metadata
        # ------------------
        ap_meta_file = [f for f in os.listdir(imec_folder) if 'ap.meta' in f][0]
        ap_meta_data = readMeta(pathlib.Path(imec_folder, ap_meta_file))
        assert isinstance(ap_meta_data, dict)
        probe_serial_number = ap_meta_data['imDatPrb_sn']

        device_name = f'imec{imec_id}'
        device = nwb_file.create_device(
            name=device_name,
            description=probe_serial_number,
            manufacturer='IMEC',
        )
        location_dict = get_target_location(config_file=config_file, device_name=device_name)
        electrode_group = nwb_file.create_electrode_group(
            name=device_name + '_shank0',  # TODO: update for multiple shanks
            description='IMEC Neuropixels 1.0 probe',
            device=device,
            location=location_dict.get('area', str(location_dict)),
        )

        # ------------------
        # Probe geometry
        # ------------------
        if experimenter in ['Axel_Bisi', 'Myriam_Hamon']:
            channel_map = np.load(
                pathlib.Path(imec_folder, 'kilosort4', 'sorter_output', 'channel_map.npy')
            ).flatten()
        else:
            channel_map = np.load(
                pathlib.Path(imec_folder, 'kilosort2', 'channel_map.npy')
            ).flatten()

        coords = MetaToCoords(
            metaFullPath=pathlib.Path(imec_folder, ap_meta_file), outType=0, showPlot=False
        )
        xcoords    = coords[0]
        ycoords    = coords[1]
        shank_id   = coords[2]
        shank_cols = np.tile([1, 3, 0, 2], reps=int(xcoords.shape[0] / 4))
        shank_rows = np.divide(ycoords, 20)   # TODO: update for NP2
        n_chan_total = int(coords[4])

        if DEBUG_PLOT:
            # Plot probe geometry and color-code by "dead" channel map
            colors = ['grey' if ch in channel_map else 'r' for ch in range(384)]

            plt.figure(figsize=(6, 10))
            plt.scatter(xcoords, ycoords, c=colors, marker='s', s=10)
            for i, ch in enumerate(range(384)):
                plt.text(xcoords[i]+1, ycoords[i], str(ch), color='k', fontsize=6, ha='center', va='center')
            plt.title(f'Probe geometry IMEC{imec_id} - red channels not in channel map')
            plt.xlabel('X (um)')
            plt.ylabel('Y (um)')

            # Save figure
            fig_path = pathlib.Path(imec_folder, f'probe_imec{imec_id}_channel_map.png')
            plt.savefig(fig_path, dpi=300)
            plt.close()

        # --------------------------------------------------------
        # Sample-space anatomical localization (brainreg)
        # Secondary/cross-check source. Resolution: one value per
        # shank row (20 um), so two channels on the same row share
        # the same area estimate — expected for NP1 geometry.
        # --------------------------------------------------------
        area_table = build_area_table(
            config_file=config_file, imec_folder=imec_folder, experimenter=experimenter
        )
        area_table = area_table.sort_values('shank_row', ascending=True)
        area_table.set_index('shank_row', drop=True, inplace=True)
        area_table = area_table.reindex(
            labels=np.arange(0, np.max(shank_rows) + 1), fill_value=np.nan, axis=0
        )

        # --------------------------------------------------------
        # Atlas-space anatomical localization (IBL ephys-atlas GUI)
        # Primary localization source. One entry per inserted channel
        # in the JSON; missing channels (above brain surface) filled
        # by nearest-neighbor (labels) and linear interpolation (coords).
        # --------------------------------------------------------
        if mouse_name.startswith('MH'):
            imec_folder_ibl = imec_folder.replace('Axel_Bisi', 'Myriam_Hamon')
        else:
            imec_folder_ibl = imec_folder
        path_channel_loc = pathlib.Path(imec_folder_ibl, 'ibl_format', 'channel_locations.json')
        assert os.path.exists(path_channel_loc), f'Missing: {path_channel_loc}'

        ephys_align_df = load_ibl_channel_locations(path_channel_loc)
        ephys_align_df = fill_missing_channels(ephys_align_df, xcoords, ycoords, n_channels=384)
        # Add CCF parent info for atlas space
        ephys_align_df = add_ccf_parent_info(df=ephys_align_df, config=config, ccf_id_col='ccf_atlas_id')

        # --------------------------------------------------------
        # QC figure: sample-space vs atlas-space channel map
        # --------------------------------------------------------
        plot_channel_map(
            imec_id=imec_id,
            xcoords=xcoords,
            ycoords=ycoords,
            channel_map=channel_map,
            area_table=area_table,
            ephys_align_df=ephys_align_df,
            imec_folder=imec_folder,
        )

        # --------------------------------------------------------
        # Build unit table from KS/Bombcell/C_Waves output
        # --------------------------------------------------------
        sync_path = get_sync_event_times_folder(config_file, experimenter=experimenter)
        spike_times_sync_file = [f for f in os.listdir(sync_path) if device_name in f]
        try:
            sync_spike_times_path = pathlib.Path(sync_path, spike_times_sync_file[0])
        except IndexError:
            print(f'Skipping {mouse_name} IMEC{imec_id}: no synced spike time file.')
            continue

        unit_table = build_unit_table(imec_folder=imec_folder, sync_spike_times_path=sync_spike_times_path)
        if unit_table is None:
            print(f'Skipping {mouse_name} IMEC{imec_id}: no spike sorting output.')
            continue

        # --------------------------------------------------------
        # Merge sample-space localization via shank_row
        # NP: two channels per row → both get the same area estimate.
        # --------------------------------------------------------
        unit_table['shank_row'] = unit_table['peak_channel'].map(lambda x: int(shank_rows[x]))
        sample_cols = ['ccf_ap', 'ccf_ml', 'ccf_dv', 'ccf_id', 'ccf_acronym', 'ccf_name']
        area_table_reset = area_table[sample_cols].reset_index()  # shank_row back as column
        unit_table = unit_table.merge(area_table_reset, on='shank_row', how='left')

        # Add sample-space CCF parent info
        unit_table = add_ccf_parent_info(df=unit_table, config=config, ccf_id_col='ccf_id')

        # --------------------------------------------------------
        # Merge atlas-space localization via peak_channel (1:1)
        # ephys_align_df has exactly one row per channel after fill.
        # --------------------------------------------------------
        unit_table['peak_channel'] = unit_table['peak_channel'].astype(int)
        ephys_align_merge = ephys_align_df[
            ['peak_channel', 'axial', 'lateral',
             'ccf_atlas_ml', 'ccf_atlas_ap', 'ccf_atlas_dv',
             'ccf_atlas_id', 'ccf_atlas_acronym', 'ccf_atlas_name',
             'ccf_atlas_parent_id', 'ccf_atlas_parent_acronym', 'ccf_atlas_parent_name']
        ].copy()
        ephys_align_merge['peak_channel'] = ephys_align_merge['peak_channel'].astype(int)
        unit_table = unit_table.merge(right=ephys_align_merge, how='left', on='peak_channel')
        unit_table['depth'] = unit_table['axial']

        # Convert non-array columns to string for NWB compatibility
        cols_to_str = [c for c in unit_table.columns if c not in ['spike_times', 'waveform_mean']]
        unit_table[cols_to_str] = unit_table[cols_to_str].astype(str)

        # Filter: remove void-region units and noise/non-soma
        unit_table = unit_table[
            (unit_table['ccf_atlas_acronym'] != 'void') &
            (~unit_table['bc_label'].isin(['noise', 'non-soma']))
        ]

        # --------------------------------------------------------
        # Add units to NWB Units table
        # --------------------------------------------------------
        for neuron_id in range(len(unit_table)):
            nwb_file.add_unit(
                id=neuron_counter,
                cluster_id=unit_table['cluster_id'].values[neuron_id],
                peak_channel=unit_table['peak_channel'].values[neuron_id],
                electrode_group=electrode_group,
                depth=unit_table['depth'].values[neuron_id],
                ks_label=unit_table['ks_label'].values[neuron_id],
                group=unit_table['group'].values[neuron_id],
                bc_label=unit_table['bc_label'].values[neuron_id],
                firing_rate=unit_table['firing_rate'].values[neuron_id],
                spike_times=unit_table['spike_times'].values[neuron_id],
                waveform_mean=unit_table['waveform_mean'].values[neuron_id],
                sampling_rate=ap_meta_data['imSampRate'],
                duration=unit_table['duration'].values[neuron_id],
                pt_ratio=unit_table['pt_ratio'].values[neuron_id],
                # Sample-space localization (brainreg histology track)
                ccf_ap=unit_table['ccf_ap'].values[neuron_id],
                ccf_ml=unit_table['ccf_ml'].values[neuron_id],
                ccf_dv=unit_table['ccf_dv'].values[neuron_id],
                ccf_id=unit_table['ccf_id'].values[neuron_id],
                ccf_acronym=unit_table['ccf_acronym'].values[neuron_id],
                ccf_name=unit_table['ccf_name'].values[neuron_id],
                ccf_parent_id=unit_table['ccf_parent_id'].values[neuron_id],
                ccf_parent_acronym=unit_table['ccf_parent_acronym'].values[neuron_id],
                ccf_parent_name=unit_table['ccf_parent_name'].values[neuron_id],
                # Atlas-space localization (IBL ephys-atlas alignment)
                ccf_atlas_ap=unit_table['ccf_atlas_ap'].values[neuron_id],
                ccf_atlas_ml=unit_table['ccf_atlas_ml'].values[neuron_id],
                ccf_atlas_dv=unit_table['ccf_atlas_dv'].values[neuron_id],
                ccf_atlas_id=unit_table['ccf_atlas_id'].values[neuron_id],
                ccf_atlas_acronym=unit_table['ccf_atlas_acronym'].values[neuron_id],
                ccf_atlas_name=unit_table['ccf_atlas_name'].values[neuron_id],
                ccf_atlas_parent_id=unit_table['ccf_atlas_parent_id'].values[neuron_id],
                ccf_atlas_parent_acronym=unit_table['ccf_atlas_parent_acronym'].values[neuron_id],
                ccf_atlas_parent_name=unit_table['ccf_atlas_parent_name'].values[neuron_id],
                # Bombcell QC metrics
                maxChannels=unit_table['maxChannels'].values[neuron_id],
                #bc_cluster_id=unit_table['bc_cluster_id'].values[neuron_id],
                useTheseTimesStart=unit_table['useTheseTimesStart'].values[neuron_id],
                useTheseTimesStop=unit_table['useTheseTimesStop'].values[neuron_id],
                percentageSpikesMissing_gaussian=unit_table['percentageSpikesMissing_gaussian'].values[neuron_id],
                percentageSpikesMissing_symmetric=unit_table['percentageSpikesMissing_symmetric'].values[neuron_id],
                presenceRatio=unit_table['presenceRatio'].values[neuron_id],
                nSpikes=unit_table['nSpikes'].values[neuron_id],
                nPeaks=unit_table['nPeaks'].values[neuron_id],
                nTroughs=unit_table['nTroughs'].values[neuron_id],
                waveformDuration_peakTrough=unit_table['waveformDuration_peakTrough'].values[neuron_id],
                spatialDecaySlope=unit_table['spatialDecaySlope'].values[neuron_id],
                waveformBaselineFlatness=unit_table['waveformBaselineFlatness'].values[neuron_id],
                rawAmplitude=unit_table['rawAmplitude'].values[neuron_id],
                signalToNoiseRatio=unit_table['signalToNoiseRatio'].values[neuron_id],
                fractionRPVs_estimatedTauR=unit_table['fractionRPVs_estimatedTauR'].values[neuron_id],
            )
            neuron_counter += 1
        print(f'Done adding {len(unit_table)} units for IMEC{imec_id}')

        # --------------------------------------------------------
        # Add electrodes to ElectrodeTable (atlas-space localization)
        # One row per physical channel
        # --------------------------------------------------------
        len_table = nwb_file.electrodes.to_dataframe().shape[0]
        for electrode_id in range(n_chan_total - 1):  # exclude sync channel 768
            area_info = ephys_align_df[ephys_align_df['ch_id'] == electrode_id]
            if len(area_info) == 0:
                print(f'Warning: no atlas info for electrode {electrode_id}, skipping.')
                continue
            area_info = area_info.iloc[0].astype(str)
            nwb_file.add_electrode(
                id=electrode_counter,
                index_on_probe=electrode_id,
                group=electrode_group,
                group_name=device_name,
                rel_x=xcoords[electrode_id],
                rel_y=ycoords[electrode_id],
                rel_z=0.0,
                shank=shank_id[electrode_id],
                shank_col=shank_cols[electrode_id],
                shank_row=shank_rows[electrode_id],
                ccf_dv=area_info['ccf_atlas_dv'],
                ccf_ml=area_info['ccf_atlas_ml'],
                ccf_ap=area_info['ccf_atlas_ap'],
                ccf_id=area_info['ccf_atlas_id'],
                ccf_acronym=area_info['ccf_atlas_acronym'],
                ccf_name=area_info['ccf_atlas_name'],
                ccf_parent_id=area_info['ccf_atlas_parent_id'],
                ccf_parent_acronym=area_info['ccf_atlas_parent_acronym'],
                ccf_parent_name=area_info['ccf_atlas_parent_name'],
                location=area_info['ccf_atlas_acronym'],
            )
            electrode_counter += 1

        # DynamicTableRegion referencing this probe's electrodes rows
        all_table_region = nwb_file.create_electrode_table_region(
            region=list(range(len_table, electrode_counter)),
            description=f'all electrodes from {device_name}',
        )

        # --------------------------------------------------------
        # Optionally add AP and LFP raw data for this probe
        # AP band -> acquisition (raw, unprocessed)
        # LFP band -> ecephys processing module (filtered/downsampled)
        # --------------------------------------------------------
        if add_recordings:

            # --- AP band: raw acquisition ---
            raw_ephys_folder = pathlib.Path(get_raw_ephys_folder(config_file))
            #ap_data_file = [f for f in os.listdir(raw_ephys_folder) if f.endswith('ap.bin')]
            imec_raw_candidates = list(raw_ephys_folder.glob(f'*_imec{imec_id}'))
            imec_raw_folder = imec_raw_candidates[0]
            ap_data_files = list(imec_raw_folder.glob('*.ap.bin'))
            if ap_data_files:
                ap_data_file = ap_data_files[0]
                raw_data_ap = read_sglx.makeMemMapRaw(pathlib.Path(raw_ephys_folder, ap_data_file), ap_meta_data)

                probe_type = readSLGX.readMeta(ap_data_file)['imDatPrb_type']
                probe_type = NP_PROBE_TYPE_MAP[int(probe_type)]
                if probe_type == 'NP1.0':
                    filter_desc = f'High-pass filter at 300 Hz ({probe_type})'
                elif probe_type == 'NP2.0':
                    filter_desc = f'Full-band ({probe_type})'

                ap_electrical_series = ElectricalSeries(
                    name=f'ElectricalSeries_ap_{device_name}',
                    data=raw_data_ap,
                    electrodes=all_table_region,
                    starting_time=0.0,
                    rate=float(ap_meta_data['imSampRate']),
                    filtering=filter_desc,
                    description=f'SpikeGLX AP band raw data from {device_name}',
                )
                nwb_file.add_acquisition(ap_electrical_series)
                print(f'Added AP raw data for {device_name}: {ap_data_file}')
            else:
                print(f'Warning: no ap.bin found for {device_name}, skipping AP raw data.')

            # --- LFP band: processed (filtered + downsampled) ---
            #lfp_meta_file = [f for f in os.listdir(raw_ephys_folder) if f.endswith('lf.meta')]
            #lfp_data_file = [f for f in os.listdir(raw_ephys_folder) if f.endswith('lf.bin')]
            lfp_meta_files = list(imec_raw_folder.glob('*.lf.meta'))
            lfp_data_files = list(imec_raw_folder.glob('*.lf.bin'))
            if lfp_meta_files and lfp_data_files:
                lfp_meta_dict = read_sglx.readMeta(pathlib.Path(raw_ephys_folder, lfp_data_files[0]))
                raw_data_lfp = read_sglx.makeMemMapRaw(pathlib.Path(raw_ephys_folder, lfp_meta_files[0]), lfp_meta_dict)

                lfp_electrical_series = ElectricalSeries(
                    name=f'ElectricalSeries_lfp_{device_name}',
                    data=raw_data_lfp,
                    electrodes=all_table_region,
                    starting_time=0.0,
                    rate=float(lfp_meta_dict['imSampRate']),
                    filtering='Low-pass filter at 500 Hz',
                    description=f'SpikeGLX LFP band data from {device_name}',
                )
                if 'ecephys' in nwb_file.processing:
                    ecephys_module = nwb_file.processing['ecephys']
                else:
                    ecephys_module = nwb_file.create_processing_module(
                        name='ecephys',
                        description='processed extracellular electrophysiology data',
                    )
                ecephys_module.add(
                    LFP(electrical_series=lfp_electrical_series, name=f'lfp_{device_name}')
                )
                print(f'Added LFP data for {device_name}: {lfp_data_file}')
            else:
                print(f'Warning: no lf.bin/lf.meta found for {device_name}, skipping LFP data.')

    print('\nDone ephys conversion to NWB.')