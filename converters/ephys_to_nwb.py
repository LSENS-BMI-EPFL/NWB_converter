#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: NWB_converter
@file: ephys_to_nwb.py
@time: 8/25/2023 9:52 AM
"""

# Imports
import os
import pathlib

import numpy as np
import pandas as pd
import yaml

from utils.ephys_converter_misc import (build_simplified_unit_table,
                                        create_electrode_table,
                                        create_simplified_unit_table,
                                        create_unit_table, get_target_location)
from utils.server_paths import (get_imec_probe_folder_list,
                                get_sync_event_times_folder)
from utils.sglx_meta_to_coords import MetaToCoords, readMeta


def convert_ephys_recording(nwb_file, config_file):
    """
    Converts ephys recording to NWB file.
    Args:
        nwb_file (object): NWB file object
        config_file: path to subject config file

    Returns:

    """
    # TODO: this will require modifications for other types of Neuropixels probes

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    # First, create dynamic tables that will be filled with data
    create_electrode_table(nwb_file=nwb_file)
    if config.get('ephys_metadata').get('unit_table') == 'simple':
        create_simplified_unit_table(nwb_file=nwb_file)
    else:
        create_unit_table(nwb_file=nwb_file)

    # Get number of probes used
    imec_probe_list = get_imec_probe_folder_list(config_file=config_file)

    # Counter for total number of electrode in recording
    electrode_counter = 0
    neuron_counter = 0

    # Then, iterate over each probe/device used in recording
    for imec_id, imec_folder in enumerate(imec_probe_list):
        print('Probe IMEC{}'.format(imec_id), imec_folder)

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

        # Get stereotaxic targeted location from metadata file
        location_dict = get_target_location(config_file=config_file,
                                            device_name=device_name)

        # Create ElectrodeGroup object
        electrode_group = nwb_file.create_electrode_group(
            name=device_name + '_shank0',
            description='IMEC probe',
            device=device,
            location=str(location_dict),
        )

        # Get saved channels information- number of shanks, channels, etc.
        coords = MetaToCoords(metaFullPath=pathlib.Path(imec_folder, ap_meta_file), outType=0, showPlot=False)
        xcoords = coords[0]
        ycoords = coords[1]
        shank_id = coords[2]
        shank_cols = np.tile([1, 3, 0, 2], reps=int(xcoords.shape[0] / 4))
        shank_rows = np.divide(ycoords, 20)
        connected = coords[3]  # whether bad channels
        n_chan_total = int(coords[4])  # includes SY sync channel 768

        # Add electrodes to ElectrodeTable
        for electrode_id in range(n_chan_total - 1):  # ignore reference channel 768

            nwb_file.add_electrode(
                id=electrode_counter,
                index_on_probe=electrode_id,
                group=electrode_group,
                group_name=device_name,
                # TODO: resolve this for location (ElectrodeGroup vs Electrode), SGLX vs anatomical estimates
                location=str(location_dict),
                ccf_location='nan',
                rel_x=xcoords[electrode_id],
                rel_y=ycoords[electrode_id],
                rel_z=0.0,
                shank=shank_id[electrode_id],
                shank_col=shank_cols[electrode_id],
                shank_row=shank_rows[electrode_id],
                ccf_dv='nan',
                ccf_ml='nan',
                ccf_ap='nan'
            )

            # Increment total number of electrode
            electrode_counter += 1


        # Get path to preprocessed sync spike times
        sync_path = get_sync_event_times_folder(config_file)
        spike_times_sync_file = [f for f in os.listdir(sync_path) if str(imec_id) in f][0]
        sync_spike_times_path = pathlib.Path(sync_path, spike_times_sync_file)

        # Different table types
        if config.get('ephys_metadata').get('unit_table') == 'simple':


            # Build unit table
            unit_table = build_simplified_unit_table(imec_folder=imec_folder,
                                                     sync_spike_times_path=sync_spike_times_path
                                                     )

            # Add units to unit table
            n_neurons = len(unit_table)
            for neuron_id in range(n_neurons):

                nwb_file.add_unit(
                    cluster_id=unit_table['cluster_id'].values[neuron_id],
                    peak_channel=unit_table['peak_channel'].values[neuron_id],
                    electrode_group=electrode_group,
                    depth=unit_table['depth'].values[neuron_id],
                    ks_label=unit_table['ks_label'].values[neuron_id],
                    firing_rate=unit_table['firing_rate'].values[neuron_id],
                    spike_times=unit_table['spike_times'].values[neuron_id],
                    waveform_mean=unit_table['waveform_mean'].values[neuron_id],
                    sampling_rate=ap_meta_data['imSampRate'],
                    id=neuron_counter,

                )

                # Increment total number of neuron
                neuron_counter += 1

        #elif config.get('ephys_metadata').get('unit_table') == 'standard':
        #    print('Standard unit table not yet implemented')
#
        #    # Build standard unit table
        #    # TODO: implement this
        #    unit_table = build_standard_unit_table(imec_folder=imec_folder,
        #                                            sync_spike_times_path=sync_spike_times_path
        #                                            )
#
        #    # Add units to unit table
        #    n_neurons = len(unit_table)
        #    for neuron_id in range(n_neurons):
#
        #        nwb_file.add_unit()
#
#
        #        # Increment total number of neuron
        #        neuron_counter += 1

        print('Done adding data for IMEC{}'.format(imec_id))

    print('Done ephys conversion to NWB. ')

    return
