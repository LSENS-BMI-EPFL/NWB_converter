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
from utils.read_sglx import readMeta
from utils.sglx_meta_to_coords import shankMapToGeom, geomMapToGeom, MetaToCoords
from utils.ephys_utils import get_target_location
from utils.server_paths import get_ephys_folder, get_imec_probe_folder_list, get_nwb_folder


def convert_ephys_recording(nwb_file, config_file):
    """
    Converts ephys recording to NWB file.
    Args:
        nwb_file (object):
        config_file:

    Returns:

    """

    # Read config file to know what data to convert.
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)

    # Get number of probes used
    imec_probe_list = get_imec_probe_folder_list(config_file=config_file)

    # Counter for total number of electrode in recording
    electrode_counter = 0

    # Iterate over each probe/device used in recording
    for imec_id, imec_folder in enumerate(imec_probe_list):
        print(imec_id, imec_folder)

        # Get serial number
        ap_meta_file = [f for f in os.listdir(imec_folder) if 'ap.meta' in f][0]
        ap_meta_data = readMeta(pathlib.Path(imec_folder, ap_meta_file))
        probe_serial_number = ap_meta_data['imDatPrb_sn']

        # Create Device object
        device_name = 'imec{}'.format(imec_id) # SpikeGLX indices at acquisition time
        device = nwb_file.create_device(
            name = device_name,
            description = probe_serial_number, # serial number
            manufacturer = 'IMEC'
        )


        # Get stereotaxic target location
        location_dict = get_target_location(config_file=config_file,
                                            device_name=device_name)

        assert type(location_dict) == dict

        # Get saved channels information- number of shanks, channels, etc.
        coords = MetaToCoords(metaFullPath=pathlib.Path(imec_folder, ap_meta_file), outType=0, showPlot=True)
        xcoords = coords[0]
        ycoords = coords[1]
        shank_id = coords[2]
        connected = coords[3]
        n_chan_total = coords[4]

       # Create ElectrodeTable object

        for electrode_id in range(n_chan_total):
            nwb_file.add_electrode(
               id=electrode_counter,
               index_on_probe=electrode_id,
               group=device,
               group_name=device_name,
               location= location_dict,
               ccf_location='nan',
               rel_x=xcoords[electrode_id],
               rel_y=ycoords[electrode_id],
               rel_z='nan',
               shank=shank_id[electrode_id],
               shank_col=2 + electrode_id/4,
               shank_row=0 + electrode_id%2,
               ccf_dv='nan',
               ccf_ml='nan',
               ccf_ap='nan'
            )

            # Increment total number of electrode
            electrode_counter += 1
    return