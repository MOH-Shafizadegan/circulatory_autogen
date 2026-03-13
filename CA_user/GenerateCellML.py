# Core imports
#& 'C:\Program Files\OpenCOR\pythonshell.bat' 'C:\Users\jebollen\OneDrive - UGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\CircAutogen\FirstTry\circulatory_autogen\CA_user\GenerateCellML.py'
# 
from pathlib import Path
import os
import sys

sys.path.append(r"C:\Users\jebollen\OneDrive - UGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\CircAutogen\circulatory_autogen\src")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import opencor as oc
except:
    print('opencor not available, open this jupyter notebook with a python version that has opencor installed')
    exit()

print("Imports done")

new = True
# get dir where this file is. This should work locally or in Docker
this_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CA_root = r"C:\Users\jebollen\OneDriveUGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\CircAutogen\circulatory_autogen"
resources_dir = os.path.join(CA_root, "resources") 

src_path = os.path.join(CA_root, "src")
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Set up paths
CA_root = os.path.join(CA_root, "CA_user/BG_Brain")
generated_models_dir = os.path.join(CA_root, "generated_models")
param_id_output_dir = os.path.join(CA_root, "param_id_output")

print("Paths done")

from utilities.utility_funcs import get_default_inp_data_dict
import pprint
# Model identifiers
model_name = "cvs_model_0d"
input_param_file = f"{model_name}_parameters.csv"

# Base user inputs (this shows all the settings that can be changed)
inp_data_dict = get_default_inp_data_dict(model_name, input_param_file, resources_dir)

inp_data_dict["generated_models_dir"] = generated_models_dir
inp_data_dict["generated_models_subdir"] = os.path.join(generated_models_dir, model_name)
inp_data_dict["model_path"] = os.path.join(inp_data_dict["generated_models_subdir"], f"{model_name}.cellml")
# TEMPORARY FOR THIS TUTORIAL
inp_data_dict['DEBUG'] = True

print('inp_data_dict set')
pprint.pprint(inp_data_dict)

from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from solver_wrappers import get_simulation_helper_from_inp_data_dict

if new:
    success = generate_with_new_architecture(inp_data_dict=inp_data_dict)

    if not success:
        raise RuntimeError("Model generation failed")
    else:
        print('Model generation successful')
        print("Model generated at: ", inp_data_dict["model_path"])