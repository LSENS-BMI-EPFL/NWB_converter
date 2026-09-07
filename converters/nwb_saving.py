import os
from datetime import datetime
import pandas as pd
import gc
gc.collect()

from pynwb import NWBHDF5IO


def save_nwb_file(nwb_file, output_folder, with_time_string=False, suffix=None, debug_cols=False):
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

    if debug_cols:
        # This is used for debugging table columns that throws an error when saving
        df = nwb_file.trials.to_dataframe()
        for col in df.columns:
            if df[col].dtype == object:
                non_str = df[col].apply(lambda x: not isinstance(x, str) and pd.notna(x))
                if non_str.any():
                    print(col, df[col][non_str].unique())
                nan_mask = df[col].isna()
                if nan_mask.any():
                    print(f"{col} has {nan_mask.sum()} NaN/None values")

        if nwb_file.units is not None:
            df = nwb_file.units.to_dataframe()
            for col in df.columns:
                if df[col].dtype == object:
                    non_str = df[col].apply(lambda x: not isinstance(x, str) and pd.notna(x))
                    if non_str.any():
                        print(col, df[col][non_str].unique())
                    nan_mask = df[col].isna()
                    if nan_mask.any():
                        print(f"{col} has {nan_mask.sum()} NaN/None values")

    with NWBHDF5IO(os.path.join(output_folder, nwb_name), 'w') as io:
        io.write(nwb_file)

    print("NWB file created at : " + str(os.path.join(output_folder, nwb_name)))

    gc.collect()
