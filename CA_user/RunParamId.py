# Core imports
#& 'C:\Program Files\OpenCOR\pythonshell.bat' 'C:\Users\jebollen\OneDrive - UGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\CircAutogen\FirstTry\circulatory_autogen\CA_user\RunParamId.py'
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


new = True
# get dir where this file is. This should work locally or in Docker
this_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CA_root = r"C:\Users\jebollen\OneDriveUGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\CircAutogen\FirstTry\circulatory_autogen"

src_path = os.path.join(CA_root, "src")
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Set up paths
CA_root = os.path.join(CA_root, "CA_user/smc_kapela")
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
    "maximum_time_step": 300.0,
}

inp_data_dict['solver_info'] = solver_info

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
    # Simulation settings
    T = 300
    inp_data_dict["sim_time"] = T # the 2 periods we want to plot
    inp_data_dict["pre_time"] = 0 # simulate for 20 periods to get to periodic steady state
    inp_data_dict["dt"] = 1.0

    sim_helper = get_simulation_helper_from_inp_data_dict(inp_data_dict)

    # Run once and plot a few representative variables
    sim_helper.run()
    variables_to_plot = [
        "smc_kapela/Ca_i"
    ]

    y = sim_helper.get_results(variables_to_plot, flatten=True)
    t = sim_helper.get_time()

    ##### Plotting #####

    plot_dir = Path(param_id_output_dir) / "quicklooks"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(3, 2, sharex=True, figsize=(8, 6))
    axs = axs.flatten()
    for idx, (ax, series, name) in enumerate(zip(axs, y, variables_to_plot)):
        ax.plot(t, series)
        ax.set_ylabel(name)
        ax.set_xlabel("Time [s]")

    plt.tight_layout()
    plt.savefig(plot_dir / "uncalibrated_outputs.png")
    exit()

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
sim_times = [[time_gt[-1]/1000]] # the sim time for each subexperiment
obs_dt = (time_gt[1]-time_gt[0])/1000
params_to_change = {}
obs_data_creator.add_protocol_info(pre_times, sim_times, params_to_change)

# add an entry for fitting the Ca_i
entry = {
    "variable": "smc_kapela/Ca_i",
    "name_for_plotting": "Ca_i",
    "data_type": "constant",
    "unit": "mM",
    "value": np.max(cai_gt),
    "obs_dt": obs_dt,
    "plot_type": "horizontal",
    "std": 0.0000001,
    "operands": ["smc_kapela/Ca_i"],
    "operation": "max"
}
obs_data_creator.add_data_item(entry)
    

obs_data_dict = obs_data_creator.get_obs_data_dict()

# pprint.pprint(obs_data_dict)

from param_id.paramID import CVS0DParamID
param_id = CVS0DParamID.init_from_dict(inp_data_dict)

# now add the obs to the param id object
param_id.set_ground_truth_data(obs_data_dict)

params_for_id_dict = [
    {
        "vessel_name": "smc_kapela", 
        "param_name": "Ca_e", # we need to initialise blood volume somewhere, here we do it in the left ventricle for 
                                   # ease of implementation. This will be pumped around the body to 
                                   # give realistic volumes throughout the CVS.
                                   # Think of this as a parameter that shifts the total blood volume.
        "min": 0.5,
        "max": 5,
        "name_for_plotting": "Ca_e", 
    }
]

print(params_for_id_dict)

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
    "num_samples": 8, # change to 256 for more accurate results
    "output_dir": str(sa_output_dir),
}

sa_agent = SensitivityAnalysis.init_from_dict(inp_data_dict)

sa_agent.set_ground_truth_data(obs_data_dict)
sa_agent.set_params_for_id(params_for_id_dict)

sa_agent.set_sa_options(sa_options)

sa_agent.run_sensitivity_analysis(sa_options)
print("Sensitivity analysis completed. Results saved in: ", sa_output_dir)


inp_data_dict["param_id_method"] = "genetic_algorithm"
param_id.set_param_id_method(inp_data_dict["param_id_method"])

# Optimiser options (adjust as needed)
optimiser_options = {
    "num_calls_to_function": 1000,
    "cost_convergence": 0.001,
    "max_patience": 10,
    "cost_type": "MSE",
}

param_id.set_optimiser_options(optimiser_options)
param_id.set_ground_truth_data(obs_data_dict)

params_for_id_subset = [
    entry for entry in params_for_id_dict
]

param_id.set_params_for_id(params_for_id_subset)

# Run calibration
param_id.run()

param_id.simulate_with_best_param_vals()
param_id.plot_outputs()
