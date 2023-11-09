# **NWB converter**

Ongoing work to convert neurophysology data to the NWB data format.

Dictionary of data entries:
https://docs.google.com/document/d/1VGH9Myq_kk0-qZW8uU_7FD9cMEc4-RvjtmP7y11hRlk/edit?usp=sharing

Code wiki:
https://github.com/LSENS-BMI-EPFL/NWB_converter/wiki

# **Installation**

Create environment 

```
conda create -n <env> python=3.9

conda activate <env>

pip install -r <petersenlab_to_nwb_env.txt>

```

# **How to use**

1. Create '.yaml' files for each session using 'make_yaml_config.py'
2. Create NWB files using 'NWB_conversion.py'


# Development
- Some implemented functions are experimenter-dependent (initials). They should be implemented for experimenter-dependent purposes.
