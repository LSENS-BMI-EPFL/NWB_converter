import yaml
import os
import ast
import pandas as pd
from utils.server_paths import (get_ref_weight_folder,
                                get_subject_analysis_folder,
                                get_subject_data_folder,
                                get_subject_mouse_number)


def add_metadata_to_config(mouse_id, session_id, experimenter):
    yaml_file_path = os.path.join(get_subject_analysis_folder(mouse_id), session_id)
    yaml_file = [os.path.join(yaml_file_path, m) for m in os.listdir(yaml_file_path)
                 if os.path.splitext(m)[1] in ['.yaml']][0]
    with open(yaml_file, 'r', encoding='utf8') as stream:
        yaml_dict = yaml.safe_load(stream)

    if experimenter == 'AR':
        metadata_xl = pd.read_excel('sv-nas1.rcp.epfl.ch/analysis/Anthony_Renard/mice_info/session_metadata.xlsx')
        session_metadata = metadata_xl.loc[metadata_xl.sesion_id == session_id]

        # Pharma metadata
        pharma_dict = {'pharma_tag': session_metadata.pharma_day.values[0],
                       'pharma_inactivation_type': session_metadata.pharma_inactivation_type.values[0],
                       'pharma_area': session_metadata.pharma_area.values[0],
                       'pharma_drug': session_metadata.pharma_drug.values[0],
                       'pharma_volume_per_site': session_metadata.pharma_volume_per_site.values[0],
                       'pharma_concentration': session_metadata.pharma_concentration.values[0],
                       'pharma_depth': session_metadata.pharma_depth.values[0]
                       }

        yaml_dict['session_metadata']['pharmacology'] = str(pharma_dict)

        # Opto metadata
        opto_dict = {'opto_tag': session_metadata.opto_day.values[0],
                     'opto_inactivation_type': session_metadata.opto_inactivation_type.values[0],
                     'opto_area': session_metadata.opto_area.values[0]
                     }
        exp_descr = ast.literal_eval(yaml_dict['session_metadata']['experiment_description'].replace("nan", "None"))
        exp_descr['opto_metadata'] = opto_dict
        yaml_dict['session_metadata']['experiment_description'] = str(exp_descr)

        # 2P metadata
        if str(session_metadata.setup_id.values[0]) in ['NaN', 'na', 'nan']:
            setup_id = '2P'
        else:
            setup_id = session_metadata.setup_id.values[0]

        if str(session_metadata.excitation.values[0]) in ['NaN', 'na', 'nan']:
            excitation_lambda = 940
        else:
            excitation_lambda = session_metadata.excitation.values[0]

        if str(session_metadata.fov_area.values[0]) and str(session_metadata.depth.values[0]) in ['nan', 'NaN', 'na']:
            img_plane_location = 'wS1-C2'
        else:
            if str(session_metadata.fov_area.values[0]) in ['nan', 'NaN', 'na']:
                img_plane_location = f"wS1-C2_{session_metadata.depth.values[0]}"
            elif str(session_metadata.depth.values[0]) in ['nan', 'NaN', 'na']:
                img_plane_location = session_metadata.fov_area.values[0]
            else:
                img_plane_location = f"{session_metadata.fov_area.values[0]}_{session_metadata.depth.values[0]}"

        if str(session_metadata.indicator.values[0]) in ['NaN', 'na', 'nan']:
            indicator = 'not specified'
        else:
            indicator = session_metadata.indicator.values[0]

        two_p_dict = {'device': f'2P microscope setup {setup_id}',
                      'emission_lambda': 510.0,
                      'excitation_lambda': excitation_lambda,
                      'image_plane_location': img_plane_location,
                      'indicator': indicator
                      }
        yaml_dict['two_photon_metadata'] = str(two_p_dict)

        # Stimulus notes:
        stim_parameters = {'Auditory background': 'white noise 80dB', 'Tone': 'Bilateral, 10kHz, 10ms, 74dB',
                           'Whisker': f"3ms, cosine pulse, {session_metadata.wh_amp.values[0]}mT"}
        yaml_dict['session_metadata']['stimulus notes'] = str(stim_parameters)

        with open(os.path.join(get_subject_analysis_folder(mouse_id), session_id, f"config_{session_id}.yaml"), 'w') as stream:
            yaml.dump(yaml_dict, stream, default_flow_style=False, explicit_start=True)



