import os
from datetime import datetime
import gc
gc.collect()

from pynwb import NWBHDF5IO


def save_nwb_file(nwb_file, output_folder, with_time_string=False, suffix=None):
    """
    Save nwb file to output folder.
    Args:
        nwb_file: NWB file object
        output_folder: output folder path
        suffix: optional suffix to add to the file name
        with_time_string: optional, add creation time string to NWB filename

    Returns:

    """
    if with_time_string:
        time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        if suffix:
            nwb_name = nwb_file.identifier + "_" + time_str + "_" + suffix + ".nwb"
        else:
            nwb_name = nwb_file.identifier + "_" + time_str + ".nwb"
    else:
        nwb_name = nwb_file.identifier + ".nwb"

    with NWBHDF5IO(os.path.join(output_folder, nwb_name), 'w') as io:
        io.write(nwb_file)

    print("NWB file created at : " + str(os.path.join(output_folder, nwb_name)))

    gc.collect()
