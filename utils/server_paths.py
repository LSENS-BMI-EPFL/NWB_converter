import os
import glob
import yaml
import platform

EXPERIMENTER_MAP = {
    'AR': 'Anthony_Renard',
    'RD': 'Robin_Dard',
    'AB': 'Axel_Bisi',
    'MP': 'Mauro_Pulin',
    'PB': 'Pol_Bech',
    'MM': 'Meriam_Malekzadeh',
    'MS': 'Lana_Smith',
    'GF': 'Anthony_Renard',
    'MI': 'Anthony_Renard',
    'JL': 'Jules_Lebert',
}


os_name = platform.system()
assert os_name in ['Windows', 'Darwin', 'Linux'], f'{os_name} not implemented' # TODO: add Linux, or a configurable server path?

if os_name == 'Windows':
    SERVER_PATH = '\\\\sv-nas1.rcp.epfl.ch'
elif os_name == 'Darwin': # MacOS
    SERVER_PATH = '/Volumes'
elif os_name == 'Linux':
    SERVER_PATH = '/mnt'

ON_HAAS = False
if platform.node() == 'haas056.ds-a3-r10.cct.rcp.epfl.ch':
    ON_HAAS = True

def get_data_root():
    if ON_HAAS:
        return os.path.join(SERVER_PATH, 'lsens-data')
    else:
        return os.path.join(SERVER_PATH, 'Petersen-Lab', 'data')

def get_analysis_root():
    if ON_HAAS:
        return os.path.join(SERVER_PATH, 'lsens-analysis')
    else:
        return os.path.join(SERVER_PATH, 'Petersen-Lab', 'analysis')

def get_subject_data_folder(subject_id):
    data_folder = os.path.join(get_data_root(), subject_id)
    return data_folder


def get_subject_analysis_folder(subject_id, experimenter=None):
    if experimenter is None:
        if subject_id == 'PB124':
            experimenter = 'Robin_Dard'
        else:
            experimenter = EXPERIMENTER_MAP[subject_id[:2]]
    analysis_folder = os.path.join(get_analysis_root(), experimenter, 'data', subject_id)
    if not os.path.exists(analysis_folder):
        os.makedirs(analysis_folder)
    return analysis_folder


def get_experimenter_analysis_folder(experimenter):
    analysis_folder = os.path.join(get_analysis_root(), EXPERIMENTER_MAP[experimenter])
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

    if initials == 'MI':
        initials = 'GF'

    return mouse_number, initials


def get_nwb_folder(subject_id, experimenter=None):
    if experimenter is not None:
        if subject_id in ['PB124']:
            experimenter = 'Robin_Dard'
        else:
            experimenter = EXPERIMENTER_MAP[subject_id[:2]]
    nwb_folder = os.path.join(get_analysis_root(), experimenter, 'NWB')
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

    ref_weight_folder = os.path.join(get_analysis_root(), EXPERIMENTER_MAP[experimenter], 'mice_info')
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

    if mouse_name[:2] in ['GF', 'MI']:
        if not os.path.exists(behavior_results_file):
            behavior_results_file = os.path.join(get_analysis_root(), 'Anthony_Renard', 'data', mouse_name, 'Recordings', 'BehaviourData',
                                                 session_name, 'performanceResults.json')
        # if not os.path.exists(behavior_results_file):
        #     behavior_results_file = os.path.join(data_folder, 'Recordings', 'BehaviourFiles',
        #                                          session_name, 'BehavResults.mat')

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
    log_continuous_cor = os.path.join(data_folder, 'Training', session_name, 'log_continuous_cor.bin')
    if os.path.exists(log_continuous_cor):
        log_continuous_file = log_continuous_cor
    return log_continuous_file


def get_movie_files(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_data_folder(mouse_name)
    movies_path = os.path.join(data_folder, 'Recording', 'Video', session_name)
    if not os.path.exists(movies_path):
        movies_path = os.path.join(data_folder, 'Recording', 'Filming', session_name)
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
    #analysis_folder = get_subject_analysis_folder(mouse_name)
    #movies_path = os.path.join(analysis_folder, session_name, 'Video')
    data_folder = get_subject_data_folder(mouse_name)
    movies_path = os.path.join(data_folder, 'Recording', session_name, 'Video')
    if not os.path.exists(movies_path):
        movies = None
        return movies
    movies = [os.path.join(movies_path, m) for m in os.listdir(movies_path) if
              os.path.splitext(m)[1] in ['.avi', '.mp4']]

    if not movies:
        movies = None

    return movies


def get_imaging_folder(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']

    analysis_folder = get_subject_analysis_folder(mouse_name)
    tiff_path = os.path.join(analysis_folder, session_name, 'suite2p', 'plane0', 'reg_tif')
    if not os.path.exists(tiff_path):
        data_folder = get_subject_data_folder(mouse_name)
        tiff_path = os.path.join(data_folder, 'Recording', 'Imaging', session_name)

    return tiff_path


def get_imaging_file(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']

    add_raw_movie = False
    analysis_folder = get_subject_analysis_folder(mouse_name)
    reg_tiff_path = os.path.join(analysis_folder, session_name, 'suite2p', 'plane0', 'reg_tif')
    if not os.path.exists(reg_tiff_path):
        add_raw_movie = True
    else:
        tiff_file = [os.path.join(reg_tiff_path, m) for m in os.listdir(reg_tiff_path)
                     if os.path.splitext(m)[1] in ['.tif', '.tiff']]
        # Sort this list
        f = lambda x: int(os.path.basename(x).split('_')[0][6:])
        tiff_file = sorted(tiff_file, key=f)    
        if not tiff_file:
            add_raw_movie = True
        else:
            print("Add registered tiff")
    
    if add_raw_movie:
        data_folder = get_subject_data_folder(mouse_name)
        tiff_path = os.path.join(data_folder, 'Recording', 'Imaging', session_name)
        if not os.path.exists(tiff_path):
            tiff_file = None
            return tiff_file
        tiff_file = [os.path.join(tiff_path, m) for m in os.listdir(tiff_path)
                     if os.path.splitext(m)[1] in ['.tif', '.tiff']]
        # f = lambda x: int(os.path.basename(x).split('_')[0][6:])
        # tiff_file = sorted(tiff_file, key=f)
        tiff_file = sorted(tiff_file)
        if not tiff_file:
            tiff_file = None
        else:
            print("Add raw tiff")

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
    run_name = [f for f in os.listdir(raw_ephys_path) if 'DS' not in f][0]
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
    """Returns the path to the processed ephys folder for a given session."""
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
    """ Get list of all processed imec probe folders for a given session."""
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_ephys_folder(config_file)
    imec_folder_list = [f for f in os.listdir(data_folder) if 'imec' in f]
    imec_folder_list = [os.path.join(data_folder, f) for f in imec_folder_list]
    if not imec_folder_list:
        print(f"No processed imec folder found for {session_name} session from {mouse_name}")
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


def get_anatomy_folder(config_file):
    """ Get path to raw data mouse anatomy folder."""
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    data_folder = get_subject_data_folder(mouse_name)
    anatomy_path = os.path.join(data_folder, 'Anatomy')

    return anatomy_path


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

def get_anat_probe_track_folder(config_file):
    """Returns path to folder with probe track estimates"""
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    experimenter = EXPERIMENTER_MAP[mouse_name[:2]]
    analysis_folder = os.path.join(get_analysis_root(), experimenter)
    if experimenter == 'Axel_Bisi':
        if int(mouse_name[2:]) < 90:
            probe_track_folder = os.path.join(analysis_folder, 'ImagedBrains', mouse_name, 'brainreg\\manual_segmentation')
        if int(mouse_name[2:]) < 102:
            probe_track_folder = os.path.join(analysis_folder, 'ImagedBrains', mouse_name, 'brainreg\\manual_segmentation')
        else:
            probe_track_folder = os.path.join(analysis_folder, 'ImagedBrains', experimenter, mouse_name, 'fused\\registered\\segmentation')
    else:
        probe_track_folder = os.path.join(analysis_folder, 'ImagedBrains', mouse_name, 'fused\\registered\\segmentation')
        print('Unspecified experimenter for probe track folder.')
        print('Default:', probe_track_folder)

    if not os.path.exists(probe_track_folder):
        print(f"No probe track folder found for {mouse_name}")
        return None

    return probe_track_folder

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
    data_folder = get_subject_data_folder(mouse_name)
    opto_config_file = os.path.join(data_folder, 'Training', session_name, 'opto_config.json')

    if not os.path.exists(opto_config_file):
        print(f"No opto config file found for {session_name} session from {mouse_name}")
        return None

    return opto_config_file


def get_widefield_file(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']
    data_folder = get_subject_data_folder(mouse_name)
    wf_folder = os.path.join(data_folder, 'Recording', 'Imaging', session_name)
    if not os.path.exists(wf_folder):
        return None
    mj2_file = [os.path.join(wf_folder, m) for m in os.listdir(wf_folder)
                   if os.path.splitext(m)[1] in ['.mj2']]
    
    if not mj2_file:
        mj2_file = None
    else:
        mj2_file = sorted(mj2_file)

    return mj2_file


def get_wf_fiji_rois_file(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']

    data_folder = get_subject_analysis_folder(mouse_name)
    wf_roi_folder = os.path.join(data_folder, session_name)

    if not os.path.exists(wf_roi_folder):
        print(f"No widefield rois folder found for {session_name} session from {mouse_name}")
        return None

    roi_file = [os.path.join(wf_roi_folder, m) for m in os.listdir(wf_roi_folder)
                if os.path.splitext(m)[1] in ['.zip']]

    if not roi_file:
        roi_file = None
    else:
        roi_file = sorted(roi_file)

    return roi_file


def get_rois_label_folder(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)
    mouse_name = config['subject_metadata']['subject_id']
    folder = get_subject_analysis_folder(mouse_name)
    folder = os.path.join(folder, 'projection_neurons')
    
    if os.path.exists(folder):
        return folder


def get_dlc_file_path(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    initials = config['session_metadata']['experimenter']
    session_id = config['session_metadata']['identifier']

    if initials == 'PB':
        experimenter = "Pol_Bech"
        dlc_folder = os.path.join(get_analysis_root(), experimenter, "data", session_id.split("_")[0], session_id).replace("\\", "/")
        dlc_file = glob.glob(dlc_folder + "/**/*view.csv")

    elif initials == 'RD':
        experimenter = "Robin_Dard"
        dlc_folder = os.path.join(get_analysis_root(), experimenter, "data", session_id.split("_")[0], session_id).replace("\\", "/")
        dlc_file = glob.glob(dlc_folder + "/**/*view.csv")

    elif initials == 'AB':
        experimenter = "Axel_Bisi"
        dlc_folder = os.path.join(get_analysis_root(), experimenter, "data", session_id.split("_")[0], session_id, 'Video').replace("\\", "/")
        dlc_file = glob.glob(dlc_folder + "/*filtered.h5")

    else:
        dlc_file = None

    if dlc_file is not None and len(dlc_file) == 0:
        print('No DLC file found for session {}'.format(session_id))
        dlc_file = None

    return dlc_file


def get_facemap_file_path(config_file):
    with open(config_file, 'r', encoding='utf8') as stream:
        config = yaml.safe_load(stream)

    mouse_name = config['subject_metadata']['subject_id']
    session_name = config['session_metadata']['session_id']

    data_folder = os.path.join(get_subject_analysis_folder(mouse_name), session_name)

    file_path = os.path.join(data_folder, f'{session_name}_sideview_proc.npy')
    if not os.path.exists(file_path):
        return None

    return file_path



