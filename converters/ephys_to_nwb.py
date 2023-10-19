#! /usr/bin/env/python3
"""
@author: Axel Bisi
@project: NWB_converter
@file: ephys_to_nwb.py
@time: 8/25/2023 9:52 AM
"""

# Imports
import os
import yaml
import pathlib
import numpy as np
from utils.sglx_meta_to_coords import readMeta, MetaToCoords
from utils.ephys_converter_misc import get_target_location, create_electrode_table, create_units_table
from utils.server_paths import get_imec_probe_folder_list


def convert_ephys_recording(nwb_file, config_file):
    """
    Converts ephys recording to NWB file.
    Args:
        nwb_file (object): NWB file object
        config_file: path to subject config file

    Returns:

    """
    #TODO: this will require modifications for other types of Neuropixels probes

    # Create dynamic tables
    create_electrode_table(nwb_file=nwb_file)
    create_units_table(nwb_file=nwb_file)

    # Read config file to know what data to convert.
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)

    # Get number of probes used
    imec_probe_list = get_imec_probe_folder_list(config_file=config_file)

    # Counter for total number of electrode in recording
    electrode_counter = 0

    # Iterate over each probe/device used in recording
    for imec_id, imec_folder in enumerate(imec_probe_list):
        print('Probe IMEC{}'.format(imec_id), imec_folder)

        # Get serial number
        ap_meta_file = [f for f in os.listdir(imec_folder) if 'ap.meta' in f][0]
        ap_meta_data = readMeta(pathlib.Path(imec_folder, ap_meta_file))
        probe_serial_number = ap_meta_data['imDatPrb_sn']

        # Create Device object
        device_name = 'imec{}'.format(imec_id) # SpikeGLX indices at acquisition time
        device = nwb_file.create_device(
            name=device_name,
            description=probe_serial_number, # serial number
            manufacturer='IMEC'
        )

        # Get stereotaxic targeted location from meta-data file
        location_dict = get_target_location(config_file=config_file,
                                            device_name=device_name)

        # Create ElectrodeGroup object
        electrode_group = nwb_file.create_electrode_group(
            name=device_name +'_shank',
            description='imec probe',
            device=device,
            location=str(location_dict),
        )

        # Get saved channels information- number of shanks, channels, etc.
        coords = MetaToCoords(metaFullPath=pathlib.Path(imec_folder, ap_meta_file), outType=0, showPlot=False)
        xcoords = coords[0]
        ycoords = coords[1]
        shank_id = coords[2]
        shank_cols = np.tile([1,3,0,2], reps=int(xcoords.shape[0]/4))
        shank_rows = np.divide(ycoords, 20)
        connected = coords[3] #whether bad channels
        n_chan_total = int(coords[4]) #includes SY sync channel 768


        # Add electrodes to ElectrodeTable
        for electrode_id in range(n_chan_total-1): # ignore reference channel 768

            nwb_file.add_electrode(
               id=electrode_counter,
               index_on_probe=electrode_id,
               group=electrode_group,
               group_name=device_name,
                # TODO: resolve this for location (ElectrodeGroup vs Electrode), SGLX vs anatomical estimates
               location= str(location_dict),
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


    return