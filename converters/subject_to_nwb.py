import numpy as np
import yaml
from pynwb import NWBHDF5IO
from pynwb import NWBFile
from pynwb.file import Subject
from datetime import datetime
from dateutil.tz import tzlocal
from continuous_log_processing.misc_functions.utils import print_info_dict


def create_nwb_file(subject_data_yaml_file, session_data_yaml_file):
    """
    Create an NWB file object using all metadata containing in YAML file

    Args:
        subject_data_yaml_file (str): Absolute path to YAML file containing the subject metadata
        session_data_yaml_file (str): Absolute path to YAML file containing the session metadata

    """

    subject_data_yaml = None
    with open(subject_data_yaml_file, 'r') as stream:
        subject_data_yaml = yaml.load(stream, Loader=yaml.FullLoader)
    if subject_data_yaml is None:
        print(f"Issue while reading the file {subject_data_yaml}")
        return None

    session_data_yaml = None
    with open(session_data_yaml_file, 'r') as stream:
        session_data_yaml = yaml.load(stream, Loader=yaml.FullLoader)
    if session_data_yaml is None:
        print(f"Issue while reading the file {session_data_yaml}")
        return None

    # Subject info
    keys_kwargs_subject = ["age", "age__reference", "description", "genotype", "sex", "species", "subject_id",
                           "weight", "date_of_birth", "strain"]
    kwargs_subject = dict()
    for key in keys_kwargs_subject:
        kwargs_subject[key] = subject_data_yaml.get(key)
        if kwargs_subject[key] is not None:
            kwargs_subject[key] = str(kwargs_subject[key])
    if "date_of_birth" in kwargs_subject:
        date_of_birth = datetime.strptime(kwargs_subject["date_of_birth"], '%m/%d/%Y')
        date_of_birth = date_of_birth.replace(tzinfo=tzlocal())
        kwargs_subject["date_of_birth"] = date_of_birth
    print(f'Subject')
    print_info_dict(kwargs_subject)
    subject = Subject(**kwargs_subject)

    # Session info
    keys_kwargs_nwb_file = ["session_description", "identifier", "session_id", "session_start_time",
                            "experimenter", "experiment_description", "institution", "keywords",
                            "notes", "pharmacology", "protocol", "related_publications",
                            "source_script", "source_script_file_name",  "surgery", "virus",
                            "stimulus_notes", "slices", "lab"]

    kwargs_nwb_file = dict()
    for key in keys_kwargs_nwb_file:
        kwargs_nwb_file[key] = session_data_yaml.get(key)
        if kwargs_nwb_file[key] is not None:
            if not isinstance(kwargs_nwb_file[key], list):
                kwargs_nwb_file[key] = str(kwargs_nwb_file[key])
    if "session_description" not in kwargs_nwb_file:
        print(f"session_description is needed in the file {session_data_yaml_file}")
        return
    if "identifier" not in kwargs_nwb_file:
        print(f"identifier is needed in the file {session_data_yaml_file}")
        return
    if "session_start_time" not in kwargs_nwb_file:
        print(f"session_start_time is needed in the file {session_data_yaml_file}")
        return
    else:
        session_start_time = datetime.strptime(kwargs_nwb_file["session_start_time"], '%Y/%m/%d %H:%M:%S')
        session_start_time = session_start_time.replace(tzinfo=tzlocal())
        kwargs_nwb_file["session_start_time"] = session_start_time
    if "session_id" not in kwargs_nwb_file:
        kwargs_nwb_file["session_id"] = kwargs_nwb_file["identifier"]

    #####################################
    # ###    creating the NWB file    ###
    #####################################
    print(f'Session')
    print_info_dict(kwargs_nwb_file)

    kwargs_nwb_file["subject"] = subject
    kwargs_nwb_file["file_create_date"] = datetime.now(tzlocal())

    nwb_file = NWBFile(**kwargs_nwb_file)

    return nwb_file
