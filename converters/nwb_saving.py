import os
from datetime import datetime
from pynwb import NWBHDF5IO


def save_nwb_file(nwb_file, output_folder):
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    nwb_name = nwb_file.identifier + "_" + time_str + ".nwb"

    with NWBHDF5IO(os.path.join(output_folder, nwb_name), 'w') as io:
        io.write(nwb_file)

    print("NWB file created at : " + str(os.path.join(output_folder, nwb_name)))
