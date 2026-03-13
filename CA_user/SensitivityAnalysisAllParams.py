# Core imports
#Single core:  & 'C:\Program Files\OpenCOR\pythonshell.bat' 'C:\Users\jebollen\OneDrive - UGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\CircAutogen\FirstTry\circulatory_autogen\CA_user\SensitivityAnalysisAllParams.py'
#Multipel cores (writeh sa_agent.use_mp=True in the inp_data_dict): 
from pathlib import Path
import os
import sys

sys.path.append(r"C:\Users\jebollen\OneDrive - UGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\CircAutogen\FirstTry\circulatory_autogen\src")

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

def pretty_print_table(data, columns=None, max_rows=200, float_fmt="{:.6g}"):
    """
    Print list-of-dicts (or a single dict) as a table using pandas formatting.
    - columns: optional list of column names (order)
    - max_rows: maximum rows to print
    - float_fmt: format string for floats
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], dict):
        df = pd.DataFrame(data)
    else:
        # fallback to the original pretty printer for unsupported types
        pretty_print_dict(data)
        return

    if columns:
        cols = [c for c in columns if c in df.columns]
        df = df[cols]

    # format numeric columns for compact printing
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].map(lambda x: float_fmt.format(x) if pd.notnull(x) else "")

    # replace long lists/arrays in cells with short summaries
    def short_val(v):
        if isinstance(v, (list, tuple, set)):
            return f"{type(v).__name__}({len(v)})"
        if isinstance(v, np.ndarray):
            return f"array{v.shape}"
        if isinstance(v, pd.DataFrame):
            return f"DataFrame{v.shape}"
        return v

    df = df.applymap(short_val)

    # print using pandas for nice column alignment
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', None, 'display.width', 120):
        print(df.to_string(index=False))

new = True
# get dir where this file is. This should work locally or in Docker
this_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CA_root = r"C:\Users\jebollen\OneDriveUGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\CircAutogen\FirstTry\circulatory_autogen"

src_path = os.path.join(CA_root, "src")
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Set up paths
model_name = "smc_kapela"
CA_root = os.path.join(CA_root, f"CA_user/{model_name}")
resources_dir = os.path.join(CA_root, "resources") 
generated_models_dir = os.path.join(CA_root, "generated_models")
param_id_output_dir = os.path.join(CA_root, "param_id_output")

print("Paths done")

from utilities.utility_funcs import get_default_inp_data_dict
import pprint
# Model identifiers
model_name = "smc_kapela"
input_param_file = f"{model_name}_parameters.csv"

# Base user inputs (this shows all the settings that can be changed)
inp_data_dict = get_default_inp_data_dict(model_name, input_param_file, resources_dir)

inp_data_dict["generated_models_dir"] = generated_models_dir
inp_data_dict["generated_models_subdir"] = os.path.join(generated_models_dir, model_name)
inp_data_dict["model_path"] = os.path.join(inp_data_dict["generated_models_subdir"], f"{model_name}.cellml")
# TEMPORARY FOR THIS TUTORIAL
inp_data_dict['DEBUG'] = True
inp_data_dict["sim_time"] = 300


solver_info = {
    "dt_solver": 1e-3,
}
inp_data_dict['solver_info'] = solver_info

print('inp_data_dict set')
pprint.pprint(inp_data_dict)

from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from solver_wrappers import get_simulation_helper_from_inp_data_dict

gt = pd.read_csv(os.path.join(CA_root, "interpolated_500ms_cai.csv"))

time_col = list(gt.keys())[0] #"time_s"
var_col = list(gt.keys())[1] #"pressure_mmHg"

time_gt = gt[time_col].to_numpy()
cai_gt = gt[var_col].to_numpy()

from utilities.obs_data_helpers import ObsDataCreator

# now create the obs data creator object for creating a dictionary that contains the data you will fit towards
obs_data_creator = ObsDataCreator()

# Protocol info (this defines your times and changes of inputs in the experiment)
pre_times = [0] # the pre time for each experiment (period is 0.9s, so this is a multiple of the period
sim_times = [[300]] # the sim time for each subexperiment
obs_dt = (time_gt[1]-time_gt[0])/1000
params_to_change = {}
obs_data_creator.add_protocol_info(pre_times, sim_times, params_to_change)

# add an entry for fitting the Ca_i
entry = {
    "variable": "smc_kapela/Ca_i",
    "name_for_plotting": "Ca_i",
    "data_type": "constant",
    "unit": "mM",
    "value": np.average(cai_gt),
    "obs_dt": obs_dt,
    "plot_type": "horizontal",
    "std": 0.000001,
    "operands": ["smc_kapela/Ca_i"],
    "operation": "mean"
}
obs_data_creator.add_data_item(entry)

def last_sample(input, series_output=False):
    if series_output: # needed for plotting the series in automatic plots
        return input
    return input[-1]

def initial_sample(time, input, series_output=False):
    if series_output: # needed for plotting the series in automatic plots
        return input
    time = np.asarray(time)
    input = np.asarray(input)
    idx = int(np.abs(time - 100).argmin()) # get index of time closest to 100s (right before NE is added)
    return input[idx]

def end_of_constriction(time, input, series_output=False):
    if series_output: # needed for plotting the series in automatic plots
        return input
    time = np.asarray(time)
    input = np.asarray(input)
    idx = int(np.abs(time - 200).argmin()) # get index of time closest to 200s (right before NO is added)
    return input[idx]


def time_to_95NE(time,input, series_output=False):
    if series_output: # needed for plotting the series in automatic plots
        return input
    time = np.asarray(time)
    idx_NOadd = int(np.abs(time - 200).argmin())
    idx_NEadd = int(np.abs(time - 100).argmin())
    input = np.asarray(input)
    baseline = input[idx_NEadd]
    target = baseline + 0.95*(input[idx_NOadd]-baseline)
    idx_target = int(np.abs(input[idx_NEadd:idx_NOadd+1] - target).argmin()) + idx_NEadd # get index of time closest to 95% of the change
    return time[idx_target]


def time_to_63_2NE(time,input, series_output=False):
    if series_output: # needed for plotting the series in automatic plots
        return input
    time = np.asarray(time)
    idx_NOadd = int(np.abs(time - 200).argmin())
    idx_NEadd = int(np.abs(time - 100).argmin())
    input = np.asarray(input)
    baseline = input[idx_NEadd]
    target = baseline + 0.632*(input[idx_NOadd]-baseline)
    idx_target = int(np.abs(input[idx_NEadd:idx_NOadd+1] - target).argmin()) + idx_NEadd # get index of time closest to 95% of the change
    return time[idx_target]

def time_to_95NO(time,input, series_output=False):
    if series_output: # needed for plotting the series in automatic plots
        return input
    time = np.asarray(time)
    idx_NOadd = int(np.abs(time - 200).argmin())
    input = np.asarray(input)
    baseline = input[idx_NOadd]
    target = baseline + 0.95*(input[-1]-baseline)
    idx_target = int(np.abs(input[idx_NOadd:]-target).argmin()) + idx_NOadd # get index of time closest to 95% of the change
    return time[idx_target]


def time_to_63_2NO(time,input, series_output=False):
    if series_output: # needed for plotting the series in automatic plots
        return input
    time = np.asarray(time)
    idx_NOadd = int(np.abs(time - 200).argmin())
    input = np.asarray(input)
    baseline = input[idx_NOadd]
    target = baseline + 0.632*(input[-1]-baseline)
    idx_target = int(np.abs(input[idx_NOadd:]-target).argmin()) + idx_NOadd # get index of time closest to 95% of the change
    return time[idx_target]


last_sample.series_to_constant = True # this is needed for plotting the series in automatic plots

extra_entry1 = {
    "variable": "AM_beforeNE",
    "name_for_plotting": "AM_beforeNE",
    "operands": ["time","smc_kapela/AM"], # these need to correspond to the operands in my_extra_feature
    "operation": "initial_sample",
    "unit": "dimensionless",
    "plot_type": "horizontal", # don't plot this feature
    "value": 0,
    "std": 0.000001
}

extra_entry2 = {
    "variable": "AMp_beforeNE",
    "name_for_plotting": "AMp_beforeNE",
    "operands": ["time","smc_kapela/AMp"], # these need to correspond to the operands in my_extra_feature
    "operation": "initial_sample",
    "unit": "dimensionless",
    "plot_type": "horizontal",
    "value": 0,
    "std": 0.000001
}

extra_entry3 = {
    "variable": "time_95AM_NE",
    "name_for_plotting": "time_95AMNE",
    "operands": ["time","smc_kapela/AM"], # these need to correspond to the operands in my_extra_feature
    "operation": "time_to_95NE",
    "unit": "dimensionless",
    "plot_type": "horizontal",
    "value": 0,
    "std": 0.000001
}
extra_entry3a = {
    "variable": "time_95AMp_NE",
    "name_for_plotting": "time_95AMpNE",
    "operands": ["time","smc_kapela/AMp"], # these need to correspond to the operands in my_extra_feature
    "operation": "time_to_95NE",
    "unit": "dimensionless",
    "plot_type": "horizontal",
    "value": 0,
    "std": 0.000001
}
extra_entry3b = {
    "variable": "time_632AM_NE",
    "name_for_plotting": "time_632AMNE",
    "operands": ["time","smc_kapela/AM"], # these need to correspond to the operands in my_extra_feature
    "operation": "time_to_63_2NE",
    "unit": "dimensionless",
    "plot_type": "horizontal",
    "value": 0,
    "std": 0.000001
}
extra_entry3c = {
    "variable": "time_632AMp_NE",
    "name_for_plotting": "time_632AMpNE",
    "operands": ["time","smc_kapela/AMp"], # these need to correspond to the operands in my_extra_feature
    "operation": "time_to_63_2NE",
    "unit": "dimensionless",
    "plot_type": "horizontal",
    "value": 0,
    "std": 0.000001
}

extra_entry4 = {
    "variable": "AM_afterNE",
    "name_for_plotting": "AM_afterNE",
    "operands": ["time","smc_kapela/AM"],
    "operation": "end_of_constriction",
    "unit": "dimensionless",
    "plot_type": "horizontal",
    "value": 0,
    "std": 0.000001
}

extra_entry5 = {
    "variable": "AMp_afterNE",
    "name_for_plotting": "AMp_afterNE",
    "operands": ["time","smc_kapela/AMp"],
    "operation": "end_of_constriction",
    "unit": "dimensionless",
    "plot_type": "horizontal",
    "value": 0,
    "std": 0.000001
}

extra_entry6 = {
    "variable": "time_95AM_NO",
    "name_for_plotting": "time_95AMNO",
    "operands": ["time","smc_kapela/AM"], # these need to correspond to the operands in my_extra_feature
    "operation": "time_to_95NO",
    "unit": "dimensionless",
    "plot_type": "horizontal",
    "value": 0,
    "std": 0.000001
}
extra_entry6a = {
    "variable": "time_95AMp_NO",
    "name_for_plotting": "time_95AMpNO",
    "operands": ["time","smc_kapela/AMp"], # these need to correspond to the operands in my_extra_feature
    "operation": "time_to_95NO",
    "unit": "dimensionless",
    "plot_type": "horizontal",
    "value": 0,
    "std": 0.000001
}
extra_entry6b = {
    "variable": "time_632AM_NO",
    "name_for_plotting": "time_632AMNO",
    "operands": ["time","smc_kapela/AM"], # these need to correspond to the operands in my_extra_feature
    "operation": "time_to_63_2NO",
    "unit": "dimensionless",
    "plot_type": "horizontal",
    "value": 0,
    "std": 0.000001
}
extra_entry6c = {
    "variable": "time_632AMp_NO",
    "name_for_plotting": "time_632AMpNO",
    "operands": ["time","smc_kapela/AMp"], # these need to correspond to the operands in my_extra_feature
    "operation": "time_to_63_2NO",
    "unit": "dimensionless",
    "plot_type": "horizontal",
    "value": 0,
    "std": 0.000001
}

extra_entry7 = {
    "variable": "AM_afterNO",
    "name_for_plotting": "AM_afterNO",
    "operands": ["smc_kapela/AM"],
    "operation": "last_sample",
    "unit": "dimensionless",
    "plot_type": "horizontal",
    "value": 0,
    "std": 0.000001
}

extra_entry8 = {
    "variable": "AMp_afterNO",
    "name_for_plotting": "AMp_afterNO",
    "operands": ["smc_kapela/AMp"],
    "operation": "last_sample",
    "unit": "dimensionless",
    "plot_type": "horizontal",
    "value": 0,
    "std": 0.000001
}


""" extra_entry9 = {
    "variable": "Cai_afterNO",
    "name_for_plotting": "Cai_afterNO",
    "operands": ["smc_kapela/Ca_i"],
    "operation": "last_sample",
    "unit": "dimensionless",
    "plot_type": "horizontal",
    "value": 0,
    "std": 0.000001
} """


obs_data_creator.add_data_item(extra_entry1)
obs_data_creator.add_data_item(extra_entry2)
obs_data_creator.add_data_item(extra_entry3)
obs_data_creator.add_data_item(extra_entry3a)
obs_data_creator.add_data_item(extra_entry3b)
obs_data_creator.add_data_item(extra_entry3c)
obs_data_creator.add_data_item(extra_entry4)
obs_data_creator.add_data_item(extra_entry5)
obs_data_creator.add_data_item(extra_entry6)
obs_data_creator.add_data_item(extra_entry6a)
obs_data_creator.add_data_item(extra_entry6b)
obs_data_creator.add_data_item(extra_entry6c)
obs_data_creator.add_data_item(extra_entry7)
obs_data_creator.add_data_item(extra_entry8)

obs_data_dict = obs_data_creator.get_obs_data_dict()


from param_id.paramID import CVS0DParamID
param_id = CVS0DParamID.init_from_dict(inp_data_dict)

# now add the obs to the param id object
param_id.set_ground_truth_data(obs_data_dict)

###############  
params_file = os.path.join(resources_dir, f"{model_name}_parameters_forSensitivyHM.csv")  

def read_and_validate_params(filepath):
    all_params = None

    try:
        df = pd.read_csv(filepath)
        all_params = df.to_dict(orient="records")
    except Exception as e:
        raise RuntimeError(f"Unable to read params file '{filepath}': {e}")

    return all_params


all_params = read_and_validate_params(params_file)
print(f"Loaded {len(all_params)} parameters")
 
# Create params_for_id_dict with all parameters and reasonable bounds  
params_for_id_dict = []  
for param in all_params:  
    # Skip parameters that don't make sense to vary (e.g., fixed constants)  
    if param['variable_name'] in ['time', 'dt']:  # add any parameters to exclude  
        continue  
    name = param['variable_name'][:-len(model_name)-1]
    # Create entry with reasonable bounds (you may need to adjust these)  
    entry = {  
        "vessel_name": 'smc_kapela',  
        "param_name": name,  
        "min": param['value']*0.1,  # Use existing min or default  
        "max": param['value']*10,  # Use existing max or default  
        "name_for_plotting": name  
    }  

    if (param['value'] < 0):
        entry['min'] = param['value']*10
        entry['max'] = param['value']*0.1

    if (param['value'] == 0):
        entry['min'] = 0
        entry['max'] = 0.1
    params_for_id_dict.append(entry)  

print(f"Loaded {len(params_for_id_dict)} parameters for sensitivity analysis")

pretty_print_table(params_for_id_dict)

# now add the params to the param id object
param_id.set_params_for_id(params_for_id_dict)


from sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis
import shutil

sa_output_dir = Path(param_id_output_dir) / "sensitivity" 
if not sa_output_dir.exists():
    sa_output_dir.mkdir(parents=True, exist_ok=True)


sa_options = {
    "method": "sobol",
    "sample_type": "saltelli",
    "num_samples": 1024, # change to 256 for more accurate results
    "output_dir": str(sa_output_dir),
}

sa_agent = SensitivityAnalysis.init_from_dict(inp_data_dict)

sa_agent.add_user_operation_func(last_sample)
sa_agent.add_user_operation_func(initial_sample)
sa_agent.add_user_operation_func(end_of_constriction)
sa_agent.add_user_operation_func(time_to_95NE)
sa_agent.add_user_operation_func(time_to_63_2NE)
sa_agent.add_user_operation_func(time_to_95NO)
sa_agent.add_user_operation_func(time_to_63_2NO)

sa_agent.use_mpi = True

sa_agent.set_ground_truth_data(obs_data_dict)
sa_agent.set_params_for_id(params_for_id_dict)

sa_agent.set_sa_options(sa_options)

sa_agent.run_sensitivity_analysis(sa_options)
print("Sensitivity analysis completed. Results saved in: ", sa_output_dir)
