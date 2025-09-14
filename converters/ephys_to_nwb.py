#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: NWB_converter
@file: ephys_to_nwb.py
@time: 8/25/2023 9:52 AM
"""

# Imports
import os
import json
import pathlib
import pandas as pd
import numpy as np
import yaml

from utils import read_sglx
from utils.ephys_converter_misc import (build_unit_table,
                                        build_area_table,
                                        create_electrode_table,
                                        create_simplified_unit_table,
                                        create_unit_table,
                                        get_probe_insertion_info,
                                        get_target_location)
from utils.server_paths import (get_imec_probe_folder_list,
                                get_sync_event_times_folder)
from utils.sglx_meta_to_coords import MetaToCoords, readMeta
from pynwb.ecephys import ElectricalSeries, LFP

def convert_ephys_recording(nwb_file, config_file, add_recordings=False):
    """
    Converts ephys recording to NWB file.
    Args:
        nwb_file (object): NWB file object
        config_file: path to subject config file
        add_recordings: bool, whether to add raw/LFP data to NWB file

    Returns:

    """
    # TODO: this will require modifications for other types of Neuropixels probes (see TODOs)

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    # -----------------------------
    # First, create dynamic tables
    # -----------------------------
    create_electrode_table(nwb_file=nwb_file)
    if config.get('ephys_metadata').get('unit_table') == 'simple':
        create_simplified_unit_table(nwb_file=nwb_file)
    else:
        create_unit_table(nwb_file=nwb_file)

    # Counter for total number of items
    electrode_counter = 0
    neuron_counter = 0

    # Get number of probes used
    imec_probe_list = get_imec_probe_folder_list(config_file=config_file)
    # ------------------------------------------------------
    # Then, iterate over each probe/device used in recording
    # ------------------------------------------------------
    for _, imec_folder in enumerate(imec_probe_list):
        imec_id = int(pathlib.Path(imec_folder).stem[-1])
        print('Probe IMEC{}'.format(imec_id), imec_folder)

        # Check if recording is valid (otherwise skip)
        probe_info_df = get_probe_insertion_info(config_file=config_file)
        mouse_name = config.get('subject_metadata').get('subject_id')
        probe_row = probe_info_df[(probe_info_df['mouse_name'] == mouse_name)
                                  &
                                  (probe_info_df['probe_id'] == imec_id)
                                  ]
        is_valid_probe = probe_row['valid'].values[0]
        if not is_valid_probe:
            print('Skipping {} probe IMEC{} because invalid recording.'.format(mouse_name, imec_id))
            continue

        # ------------------------
        # Get probe insertion data
        # ------------------------

        # Get serial number
        ap_meta_file = [f for f in os.listdir(imec_folder) if 'ap.meta' in f][0]
        ap_meta_data = readMeta(pathlib.Path(imec_folder, ap_meta_file))
        probe_serial_number = ap_meta_data['imDatPrb_sn']

        # Create Device object
        device_name = 'imec{}'.format(imec_id)  # SpikeGLX indices at acquisition time
        device = nwb_file.create_device(
            name=device_name,
            description=probe_serial_number,  # serial number
            manufacturer='IMEC'
        )

        # Get stereotaxic targeted location from external metadata file
        location_dict = get_target_location(config_file=config_file, device_name=device_name)

        # Create ElectrodeGroup object
        electrode_group = nwb_file.create_electrode_group(
            name=device_name + '_shank0', #TODO: update for multiple shanks
            description='IMEC probe',
            device=device,
            location=str(location_dict),
        )

        # Get saved channels information- number of shanks, channels, etc. #TODO: update for NP2
        coords = MetaToCoords(metaFullPath=pathlib.Path(imec_folder, ap_meta_file), outType=0, showPlot=False)
        xcoords = coords[0]
        ycoords = coords[1]
        shank_id = coords[2]
        shank_cols = np.tile([1, 3, 0, 2], reps=int(xcoords.shape[0] / 4))
        shank_rows = np.divide(ycoords, 20) #TODO: to update for NP2
        connected = coords[3]  # whether they are bad channels
        n_chan_total = int(coords[4])  # includes SY sync channel 768

        # ----------------------------------
        # Get anatomical reconstruction data
        # ----------------------------------

        # Build table with anatomical location estimates of each electrode
        area_table = build_area_table(config_file=config_file, imec_folder=imec_folder, probe_info=probe_row)

        # Reindex to match shank electrode order
        area_table = area_table.sort_values(by=['shank_row'], ascending=True, axis=0)
        area_table.set_index(keys='shank_row', drop=True, inplace=True)
        area_table = area_table.reindex(labels=np.arange(0, np.max(shank_rows)+1), fill_value=np.nan, axis=0)

        # Compare physical number of electrode rows with interpolated rows in area table
        physical_rows = probe_row['n_rows'].values[0]
        interpolated_rows = len(area_table)
        if abs(physical_rows - area_table.shape[0]) > 1000:
            print(f'Warning: physical number of rows inserted ({physical_rows}) is very different from interpolated rows in area table ({interpolated_rows}), \
                  this may be due to an error during the probe track interpolation.')

        # --------------------------------
        # Add electrodes to ElectrodeTable
        # --------------------------------

        len_table = nwb_file.electrodes.to_dataframe().shape[0]

        for electrode_id in range(n_chan_total - 1):  # ignore reference channel 768

            # For each electrode, get anatomical info
            row_id = int(shank_rows[electrode_id]) # this accounts for two electrodes per row
            area_info = area_table.iloc[row_id, :] # from tip upwards
            area_info = area_info.astype(str)


            nwb_file.add_electrode(
                id=electrode_counter,
                index_on_probe=electrode_id,
                group=electrode_group, # required argument
                group_name=device_name,
                rel_x=xcoords[electrode_id],
                rel_y=ycoords[electrode_id],
                rel_z=0.0,
                shank=shank_id[electrode_id],
                shank_col=shank_cols[electrode_id],
                shank_row=shank_rows[electrode_id],
                ccf_dv=area_info['ccf_dv'],
                ccf_ml=area_info['ccf_ml'],
                ccf_ap=area_info['ccf_ap'],
                ccf_id=area_info['ccf_id'],
                ccf_acronym=area_info['ccf_acronym'],
                ccf_name=area_info['ccf_name'],
                ccf_parent_id=area_info['ccf_parent_id'],
                ccf_parent_acronym=area_info['ccf_parent_acronym'],
                ccf_parent_name=area_info['ccf_parent_name'],
                location=str(area_info['ccf_acronym']) # required string argument
            )

            # Increment total number of electrode
            electrode_counter += 1

        # Create a list of reference electrodes for ElectricalSeries objects (raw voltage, LFP)
        all_table_region = nwb_file.create_electrode_table_region(
            region=list(range(len_table, electrode_counter)),  # exclude sync channel 768
            description="all electrodes",
        )

        # ---------------------------
        # Get electrophysiology data
        # ---------------------------

        # Get path to preprocessed sync spike times
        sync_path = get_sync_event_times_folder(config_file)
        spike_times_sync_file = [f for f in os.listdir(sync_path) if device_name in f]
        try:
            sync_spike_times_path = pathlib.Path(sync_path, spike_times_sync_file[0])
        except IndexError:
            print('Skipping {} probe IMEC{} because no synced spike time file found.'.format(mouse_name, imec_id))
            continue


        # Build unit table
        unit_table = build_unit_table(imec_folder=imec_folder, sync_spike_times_path=sync_spike_times_path)

        if unit_table is None:
            print('Skipping {} probe IMEC{} because no spike sorting.'.format(mouse_name, imec_id))
            continue

        # Join anatomical info to each unit entry
        unit_table['shank_row'] = unit_table['peak_channel'].map(lambda x: int(shank_rows[x])) # get shank row from peak channel
        unit_table.set_index(keys='shank_row', drop=True, inplace=True)
        unit_table = unit_table.merge(area_table, on='shank_row', how='left') #shank_row as indices on both dataframes
        cols_to_str = [c for c in unit_table.columns if c not in ['spike_times', 'waveform_mean']]
        unit_table[cols_to_str] = unit_table[cols_to_str].astype(str) # convert to string to avoid error when adding to NWB file

        # --------------------------------------------------------
        # Load anatomical data after IBL ephys-atlas GUI alignment
        # --------------------------------------------------------
        path_channel_loc = pathlib.Path(imec_folder, 'ibl_format', 'channel_locations.json')
        if is_valid_probe:
            if os.path.exists(path_channel_loc):
                with open(path_channel_loc, "r") as f:
                    data = json.load(f)
            else:
                print(f'Warning: No ibl_format/channel_locations.json found for {mouse_name} IMEC{imec_id}, '
                      f'skipping ephys-atlas alignment.')

        ephys_align_df = pd.DataFrame.from_dict(data, orient='index')  # flatten dict and create df

        # Transform coordinates from bregma-centry to absolute CCF space
        bregma_xyz = ephys_align_df.loc['origin', 'bregma'] # get bregma coords in CCF space
        bregma_xyz = np.array(bregma_xyz).astype(float)
        ephys_align_df['x'] = ephys_align_df['x'].map(lambda x: float(x) + bregma_xyz[0])  # ML
        ephys_align_df['y'] = ephys_align_df['y'].map(lambda y: -float(y) + bregma_xyz[1])  # AP
        ephys_align_df['z'] = ephys_align_df['z'].map(lambda z: -float(z) + bregma_xyz[2])  # DV
        ephys_align_df = ephys_align_df[ephys_align_df.index != 'origin'] # remove bregma origin entry

        # Match index on channel id
        ephys_align_df.reset_index(inplace=True)  # reset index to move channels into column
        ephys_align_df.rename(columns={'index': 'peak_channel'}, inplace=True)  # rename to existing column from neural df
        ephys_align_df['peak_channel'] = ephys_align_df['peak_channel'].map(lambda x: int(x.split('_')[-1])) # keep int
        col_mapper = {
            'x': 'ccf_atlas_ml', #x=ML
            'y': 'ccf_atlas_ap', #y=AP
            'z': 'ccf_atlas_dv', #z=DV
            'brain_region_id': 'ccf_atlas_id',
            'brain_region': 'ccf_atlas_acronym',
        }
        ephys_align_df = ephys_align_df.rename(columns=col_mapper)  # rename columns to match existing anatomical columns

        # Join ephys-aligned anatomical info to each unit channel using 'ch' col
        unit_table['peak_channel'] = unit_table['peak_channel'].astype(int)
        unit_table = unit_table.merge(ephys_align_df, left_on='peak_channel', right_on='peak_channel', how='left')


        # -----------------------
        # Add units to Unit table
        # -----------------------

        n_neurons = len(unit_table)
        for neuron_id in range(n_neurons):
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
                ccf_dv=unit_table['ccf_dv'].values[neuron_id],
                ccf_ml=unit_table['ccf_ml'].values[neuron_id],
                ccf_ap=unit_table['ccf_ap'].values[neuron_id],
                ccf_id=unit_table['ccf_id'].values[neuron_id],
                ccf_acronym=unit_table['ccf_acronym'].values[neuron_id],
                ccf_name=unit_table['ccf_name'].values[neuron_id],
                ccf_parent_id=unit_table['ccf_parent_id'].values[neuron_id],
                ccf_parent_acronym=unit_table['ccf_parent_acronym'].values[neuron_id],
                ccf_parent_name=unit_table['ccf_parent_name'].values[neuron_id],
                ccf_atlas_dv=unit_table['ccf_atlas_dv'].values[neuron_id],
                ccf_atlas_ml=unit_table['ccf_atlas_ml'].values[neuron_id],
                ccf_atlas_ap=unit_table['ccf_atlas_ap'].values[neuron_id],
                ccf_atlas_id=unit_table['ccf_atlas_id'].values[neuron_id],
                ccf_atlas_acronym=unit_table['ccf_atlas_acronym'].values[neuron_id],
                maxChannels=unit_table['maxChannels'].values[neuron_id],
                bc_cluster_id=unit_table['bc_cluster_id'].values[neuron_id],
                useTheseTimesStart=unit_table['useTheseTimesStart'].values[neuron_id],
                useTheseTimesStop=unit_table['useTheseTimesStop'].values[neuron_id],
                percentageSpikesMissing_gaussian=unit_table['percentageSpikesMissing_gaussian'].values[neuron_id],
                percentageSpikesMissing_symmetric=unit_table['percentageSpikesMissing_symmetric'].values[neuron_id],
                presenceRatio=unit_table['presenceRatio'].values[neuron_id],
                nSpikes=unit_table['nSpikes'].values[neuron_id],
                nPeaks=unit_table['nPeaks'].values[neuron_id],
                nTroughs=unit_table['nTroughs'].values[neuron_id],
                #isSomatic=unit_table['isSomatic'].values[neuron_id],
                waveformDuration_peakTrough=unit_table['waveformDuration_peakTrough'].values[neuron_id],
                spatialDecaySlope=unit_table['spatialDecaySlope'].values[neuron_id],
                waveformBaselineFlatness=unit_table['waveformBaselineFlatness'].values[neuron_id],
                rawAmplitude=unit_table['rawAmplitude'].values[neuron_id],
                signalToNoiseRatio=unit_table['signalToNoiseRatio'].values[neuron_id],
                fractionRPVs_estimatedTauR=unit_table['fractionRPVs_estimatedTauR'].values[neuron_id],

            )

            # Increment total number of neuron
            neuron_counter += 1

        print('Done adding spike data for IMEC{}'.format(imec_id))

        add_recordings = False
        if add_recordings:

            # ------------------------
            # Add LFP data to NWB file
            # ------------------------

            # Read LFP data and metadata
            lfp_meta_file = [f for f in os.listdir(imec_folder) if 'lf.meta' in f][0]
            lfp_meta_dict = read_sglx.readMeta(pathlib.Path(imec_folder, lfp_meta_file))
            lfp_data_file = [f for f in os.listdir(imec_folder) if 'lf' in f][0]
            raw_data_lfp = read_sglx.makeMemMapRaw(pathlib.Path(imec_folder, lfp_data_file), lfp_meta_dict)

            # Create ElectricalSeries object
            lfp_electrical_series = ElectricalSeries(
                name="ElectricalSeries",
                data=raw_data_lfp,
                electrodes=all_table_region, # indices of electrodes to which this data is relevant
                starting_time=0.0,
                rate=float(lfp_meta_dict['imSampRate']),
                filtering='0.5-500Hz',
                description='SpikeGLX-acquired LFP data from IMEC{}'.format(imec_id)
            )

            # Store in a LFP object
            lfp = LFP(electrical_series=lfp_electrical_series,
                      name='lfp_{}'.format(device_name)
            )

            # Create a module for processed ecephys data
            print("Creating ecephys processing module")
            if 'ecephys' in nwb_file.processing:
                ecephys_module = nwb_file.processing['ecephys']
            else:
                ecephys_module = nwb_file.create_processing_module(
                    name='ecephys',
                    description='contains processed extracellular electrophysiology data'
                )

            ecephys_module.add(lfp)

    print('Done ephys conversion to NWB.')

    return
