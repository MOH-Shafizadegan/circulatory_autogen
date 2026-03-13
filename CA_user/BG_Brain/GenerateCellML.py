# This script generates a CellML file for a 0D circulatory model using the new architecture of the circulatory autogen project.
# It uses a set of input parameters defined in a CSV file and generates the model based on those parameters.
# The generated model is saved in the specified directory.
#
# Run the line below in the terminal to run this script. Make sure to change the paths if needed and include the & in front.
# & 'C:\Program Files\OpenCOR\pythonshell.bat' '<path_to_this_script>\GenerateCellML.py'
# for me:
# & 'C:\Program Files\OpenCOR\pythonshell.bat' 'C:\Users\jebollen\OneDriveUGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\CircAutogen\circulatory_autogen\CA_user\BG_Brain\GenerateCellML.py'
#
# Files required and their location
# - This script: CA_user/CVS0D/GenerateCellML.py
# - Parameter file: CA_user/CVS0D/resources/<model_name>_parameters.csv
# - Vessel array file: CA_user/CVS0D/resources/<model_name>_vessel_array.csv
#
# The output of this code are the CellML files required for the calibration in generated_models/<model_name>/


# ===================================================
#                  User setup 
# Define the model specific settings 
# Change the line below as needed to generate the model you want.
#  - model_name
# ===================================================

# Set the model_name 
model_name = "cerebral"

# ===================================================
#                  Imports and setup 
# Import the necassary libraries and define the dirs
# Nothing needs to be changed.
# ===================================================
from pathlib import Path
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import opencor as oc
except:
    print('opencor not available, open this jupyter notebook with a python version that has opencor installed')
    exit()
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))

print("First imports done")

CA_root = str(Path(__file__).resolve().parent.parent.parent)

src_path = os.path.join(CA_root, "src")
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

CA_user_dir = str(Path(__file__).resolve().parent)
generated_models_dir = os.path.join(CA_user_dir, "generated_models")
param_id_output_dir = os.path.join(CA_user_dir, "param_id_output")
resources_dir = os.path.join(CA_user_dir, "resources") 

print("Paths done")

from utilities.utility_funcs import get_default_inp_data_dict
import pprint
from scripts.script_generate_with_new_architecture import generate_with_new_architecture

print("Second imports done")

# ===================================================
#                  Model setup 
# Define the rest of the settings automatically
# Nothing needs to be changed. 
# ===================================================

input_param_file = f"{model_name}_parameters.csv"
# Base user inputs (this shows all the settings that can be changed)
inp_data_dict = get_default_inp_data_dict(model_name, input_param_file, resources_dir)
inp_data_dict["generated_models_dir"] = generated_models_dir
inp_data_dict["generated_models_subdir"] = os.path.join(generated_models_dir, model_name)
inp_data_dict["model_path"] = os.path.join(inp_data_dict["generated_models_subdir"], f"{model_name}.cellml")

print('inp_data_dict set: ')
pprint.pprint(inp_data_dict)

# ===================================================
#         Generate CellML file and other files 
# Nothing needs to be changed. 
# ===================================================

success = generate_with_new_architecture(inp_data_dict=inp_data_dict)

if not success:
    raise RuntimeError("Model generation failed")
else:
    print('Model generation successful')
    print("Model generated at: ", inp_data_dict["model_path"])