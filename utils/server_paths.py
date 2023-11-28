import os
import yaml

EXPERIMENTER_MAP = {
    'AR': 'Anthony_Renard',
    'RD': 'Robin_Dard',
    'AB': 'Axel_Bisi',
    'MP': 'Mauro_Pulin',
    'PB': 'Pol_Bech',
    'MM': 'Meriam_Malekzadeh',
    'LS': 'Lana_Smith',
    'GF': 'Anthony_Renard',
    'MI': 'Anthony_Renard',
}


def get_subject_data_folder(subject_id):
    data_folder = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'data', subject_id)

    return data_folder


def get_subject_analysis_folder(subject_id):
    if subject_id == 'PB124':
        experimenter = 'Robin_Dard'
    else:
        # Map initials to experimenter to get analysis folder path.
        experimenter = EXPERIMENTER_MAP[subject_id[:2]]
    analysis_folder = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis',
                                   experimenter, 'data', subject_id)
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)

    return analysis_folder


def get_subject_mouse_number(subject_id):
    """Get mouse number for integer comparison"""
    if len(subject_id) != 5:
        raise ValueError('Subject mouse name must be 5 characters long. Check subject name.')
    initials = subject_id[:2]
    mouse_number = subject_id[2:] # a string

    if int(mouse_number[0]) > 0:
        mouse_number = int(float(mouse_number))
    else:
        mouse_number = int(float(mouse_number[1:]))

    return mouse_number, initials


def get_nwb_folder(subject_id):
    if subject_id in ['PB124', 'AR103', 'AR071']:
        experimenter = 'Robin_Dard'
    else:
        experimenter = EXPERIMENTER_MAP[subject_id[:2]]
    nwb_folder = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis',
                              experimenter, 'NWB')
    if not os.path.exists(nwb_folder):
        os.makedirs(nwb_folder)

    return nwb_folder


def get_ref_weight_folder(experimenter):
    """
    Get the path to the folder where the reference weights are stored.
    Args:
        experimenter:

    Returns:

    """

    ref_weight_folder = os.path.join('\\\\sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis',
                                     EXPERIMENTER_MAP[experimenter], 'mice_info')
    if not os.path.exists(ref_weight_folder):
        os.makedirs(ref_weight_folder)

    return ref_weight_folder


def get_behavior_results_file(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_data_folder(mouse_name)
    behavior_results_file = os.path.join(data_folder, 'Training', session_name, 'results.csv')
    if not os.path.exists(behavior_results_file):
        behavior_results_file = os.path.join(data_folder, 'Training', session_name, 'results.txt')

    return behavior_results_file


def get_session_config_file(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_data_folder(mouse_name)
    session_config_file = os.path.join(data_folder, 'Training', session_name, 'session_config.json')

    return session_config_file


def get_calibration_file(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    session_date = session_name.split('_')[1]
    file_name = '{}_{}_stim_coil_calibration.mat'.format(mouse_name, session_date)
    data_folder = get_subject_data_folder(mouse_name)
    calibration_file = os.path.join(data_folder, 'Training', session_name, file_name)

    return calibration_file


def get_log_continuous_file(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_data_folder(mouse_name)
    log_continuous_file = os.path.join(data_folder, 'Training', session_name, 'log_continuous.bin')

    return log_continuous_file


def get_movie_files(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_data_folder(mouse_name)
    movies_path = os.path.join(data_folder, 'Recording', 'Video', session_name)
    if not os.path.exists(movies_path):
        movies = None
        return movies
    movies = [os.path.join(movies_path, m) for m in os.listdir(movies_path)
              if os.path.splitext(m)[1] in ['.avi', '.mp4']]
    if not movies:
        movies = None

    return movies


def get_session_movie_files(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_data_folder(mouse_name)
    movies_path = os.path.join(data_folder, 'Recording', session_name, 'Video')
    if not os.path.exists(movies_path):
        movies = None
        return movies
        #os.makedirs(movies_path)
    movies = [os.path.join(movies_path, m) for m in os.listdir(movies_path) if
              os.path.splitext(m)[1] in ['.avi', '.mp4']]
    if not movies:
        movies = None

    return movies


def get_imaging_file(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_data_folder(mouse_name)
    tiff_path = os.path.join(data_folder, 'Recording', 'Imaging', session_name)
    if not os.path.exists(tiff_path):
        tiff_file = None
        return tiff_file
    tiff_file = [os.path.join(tiff_path, m) for m in os.listdir(tiff_path)
                 if os.path.splitext(m)[1] in ['.tif', '.tiff']]

    if not tiff_file:
        tiff_file = None

    return tiff_file


def get_suite2p_folder(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_analysis_folder(mouse_name)
    suite2p_path = os.path.join(data_folder, session_name, 'suite2p')
    if not os.path.exists(suite2p_path):
        print(f"No suite2p folder found for {session_name} session from {mouse_name}")
        return None
    else:
        return suite2p_path


def get_raw_ephys_folder(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_data_folder(mouse_name)
    raw_ephys_path = os.path.join(data_folder, 'Recording', session_name, 'Ephys')
    run_name = [f for f in os.listdir(raw_ephys_path)][0]
    raw_ephys_run_folder = os.path.join(raw_ephys_path, run_name)

    return raw_ephys_run_folder

def get_raw_ephys_nidq_files(config_file):
    raw_ephys_folder = get_raw_ephys_folder(config_file)
    raw_nidq_meta = [f for f in os.listdir(raw_ephys_folder) if 'nidq.meta' in f][0]
    raw_nidq_meta = os.path.join(raw_ephys_folder, raw_nidq_meta)
    raw_nidq_bin = [f for f in os.listdir(raw_ephys_folder) if 'nidq.bin' in f][0]
    raw_nidq_bin = os.path.join(raw_ephys_folder, raw_nidq_bin)
    return raw_nidq_meta, raw_nidq_bin


def get_ephys_folder(config_file):
    """Returns the path to the ephys folder for a given session."""
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_analysis_folder(mouse_name)
    ephys_path = os.path.join(data_folder, session_name, 'Ephys')
    ephys_folder = [f for f in os.listdir(ephys_path) if 'catgt' in f][0]
    ephys_path = os.path.join(ephys_path, ephys_folder)
    if not os.path.exists(ephys_path):
        print(f"No Ephys folder found for {session_name} session from {mouse_name}")
        return None
    else:
        return ephys_path


def get_imec_probe_folder_list(config_file):
    """ Get list of all imec probe folders for a given session."""
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_ephys_folder(config_file)
    imec_folder_list = [f for f in os.listdir(data_folder) if 'imec' in f]
    imec_folder_list = [os.path.join(data_folder, f) for f in imec_folder_list]
    if not imec_folder_list:
        print(f"No imec folder found for {session_name} session from {mouse_name}")
        return None
    else:
        return imec_folder_list


def get_sync_event_times_folder(config_file):
    """ Get the path to the sync_event_times folder for a given session."""
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_ephys_folder(config_file)
    sync_event_times_path = os.path.join(data_folder, 'sync_event_times')
    if not os.path.exists(sync_event_times_path):
        print(f"No sync_event_times folder found for {session_name} session from {mouse_name}")
        return None
    else:
        return sync_event_times_path


def get_cwaves_folder(imec_probe_folder):
    """ Get the path to the cwaves folder for a given imec probe folder."""
    cwaves_folder = os.path.join(imec_probe_folder, 'cwaves')
    if not os.path.exists(cwaves_folder):
        print(f"No cwaves folder found for {imec_probe_folder}")
        return None
    else:
        return cwaves_folder


def get_anat_images_files(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_data_folder(mouse_name)
    anat_images_path = os.path.join(data_folder, 'Recording', 'Imaging', 'Anat', session_name)
    if not os.path.exists(anat_images_path):
        anat_images = None
        return anat_images
    anat_images = [os.path.join(anat_images_path, m) for m in os.listdir(anat_images_path)
                   if os.path.splitext(m)[1] in ['.tif', '.tiff']]

    if not anat_images:
        anat_images = None

    return anat_images


def get_opto_results_file(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_data_folder(mouse_name)
    behavior_results_file = os.path.join(data_folder, 'Training', session_name, 'results_opto.csv')
    if not os.path.exists(behavior_results_file):
        print(f"No opto results file found for {session_name} session from {mouse_name}")
        return None

    return behavior_results_file

def get_opto_config_file(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_analysis_folder(mouse_name)
    opto_config_file = os.path.join(data_folder, 'Training', session_name, 'opto_config.json')

    if not os.path.exists(opto_config_file):
        print(f"No opto config file found for {session_name} session from {mouse_name}")
        return None

    return opto_config_file
