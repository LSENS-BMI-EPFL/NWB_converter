# **NWB converter**

Ongoing work to convert neurophysology data to the NWB data format

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
