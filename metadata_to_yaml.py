import yaml
import os
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
        pharma_dict = {'pharma_tag': session_metadata.pharma_day,
                       'pharma_inactivation_type': session_metadata.pharma_inactivation_type,
                       'pharma_area': session_metadata.pharma_area,
                       'pharma_drug': session_metadata.pharma_drug,
                       'pharma_volume_per_site': session_metadata.pharma_volume_per_site,
                       'pharma_concentration': session_metadata.pharma_concentration,
                       'pharma_depth': session_metadata.pharma_depth
                       }

        yaml_dict['session_metadata']['pharmacology'] = str(pharma_dict)

        opto_dict = {'opto_tag': session_metadata.opto_day,
                     'opto_inactivation_type': session_metadata.opto_inactivation_type,
                     'opto_area': session_metadata.opto_area
                     }
        yaml_dict['session_metadata']['optogenetics'] = str(opto_dict)

        two_p_dict = {'device': f'2P microscope setup {session_metadata.setup_id}',
                      'emission_lambda': 510.0,
                      'excitation_lambda': session_metadata.excitation,
                      'image_plane_location': f"{session_metadata.fov_area}_{session_metadata.depth}",
                      'indicator': session_metadata.indicator
                      }
        yaml_dict['two_photon_metadata'] = str(two_p_dict)

        with open(os.path.join(get_subject_analysis_folder(mouse_id), session_id, f"config_{session_id}.yaml"), 'w') as stream:
            yaml.dump(yaml_dict, stream, default_flow_style=False, explicit_start=True)



