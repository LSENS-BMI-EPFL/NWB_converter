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
from utils.ephys_utils import get_target_location
from utils.server_paths import get_ephys_folder, get_imec_probe_folder_list, get_nwb_folder


def convert_ephys_recording(nwb_file, config_file):
    """
    Converts ephys recording to NWB file.
    Args:
        nwb_file:
        config_file:

    Returns:

    """

    # Read config file to know what data to convert.
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)

    # Get number of probes used
    imec_probe_list = get_imec_probe_folder_list(config_file=config_file)

    electrode_counter = 0
    for imec_id, imec_folder in enumerate(imec_probe_list):
        print(imec_id, imec_folder)

        # Get serial number
        ap_meta_file = [f for f in os.listdir(imec_folder) if 'ap.meta' in f][0]
        ap_meta_data = readMeta(pathlib.Path(imec_folder,ap_meta_file))
        probe_serial_number = ap_meta_data['imDatPrb_sn']

        # Create Device object
        device_name = 'imec{}'.format(imec_id) # SpikeGLX indices at acquisition time
        device = nwb_file.create_device(
            name = device_name,
            description = probe_serial_number, # serial number
            manufacturer = 'IMEC'
        )

        # Get saved channels information- number of probes, channels, etc.


        # Get stereotaxic target location
        location_dict = get_target_location(config_file=config_file,
                                            device_name=device_name)

        assert type(location_dict) == dict
        print(location_dict.keys())


       ## Create ElectrodeTable object
       #nwb_file.add_electrode(
       #    id=electrode_counter,
       #   # index_on_probe=
       #    group=device,
       #    group_name=device_name,
       #    location= location_dict,
       #    ccf_location=
       #    rel_x=
       #    rel_y=
       #    rel_z=
       #    shank=
       #    shank_col=
       #    shank_row=
       #    ccf_dv=
       #    ccf_ml=
       #    ccf_ap=
       #    imp=

       #)

        electrode_counter += 1



    return