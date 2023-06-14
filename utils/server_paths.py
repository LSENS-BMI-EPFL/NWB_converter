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
        }


def get_subject_data_folder(subject_id):
    data_folder = os.path.join('\\\\sv2files.epfl.ch', 'Petersen-Lab', 'data', subject_id)

    return data_folder


def get_subject_analysis_folder(subject_id):
    if subject_id == 'PB124':
        experimenter = 'Robin_Dard'
    else:
        experimenter = EXPERIMENTER_MAP[subject_id[:2]]
    # Map initials to experimenter to get analysis folder path.
    analysis_folder = os.path.join('\\\\sv2files.epfl.ch', 'Petersen-Lab', 'analysis',
                                   experimenter, 'data', subject_id)
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)

    return analysis_folder


def get_nwb_folder(subject_id):
    nwb_folder = os.path.join('\\\\sv2files.epfl.ch', 'Petersen-Lab', 'analysis',
                              EXPERIMENTER_MAP[subject_id[:2]], 'NWB')
    if not os.path.exists(nwb_folder):
        os.makedirs(nwb_folder)

    return nwb_folder


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
        os.makedirs(movies_path)
    movies = [os.path.join(movies_path, m) for m in os.listdir(movies_path) if os.path.splitext(m)[1] in ['.avi', '.mp4']]
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
        os.makedirs(tiff_path)
    tiff_file = [os.path.join(tiff_path, m) for m in os.listdir(tiff_path) if os.path.splitext(m)[1] == '.tif']
    if tiff_file:
        # Assuming there is a single tiff file.
        tiff_file = tiff_file[0]
    else:
        tiff_file = None

    return tiff_file


def get_suite2p_folder(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_analysis_folder(mouse_name)
    suite2p_path = os.path.join(data_folder, 'data', mouse_name, session_name, 'suite2p')
    if not os.path.exists(suite2p_path):
        print(f"No suite2p folder found for {session_name} session from {mouse_name}")
        return None
    else:
        return suite2p_path

