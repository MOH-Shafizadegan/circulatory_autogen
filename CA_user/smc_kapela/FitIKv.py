# Core imports
#& 'C:\Program Files\OpenCOR\pythonshell.bat' 'C:\Users\jebollen\OneDrive - UGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\CircAutogen\circulatory_autogen\CA_user\smc_kapela\FitIKv.py'
from datetime import datetime
import json
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

from utilities.obs_data_helpers import ObsDataCreator
from utilities.utility_funcs import get_default_inp_data_dict
import pprint
from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from solver_wrappers import get_simulation_helper_from_inp_data_dict
from param_id.paramID import CVS0DParamID
from sensitivity_analysis.sensitivityAnalysis import SensitivityAnalysis

sys.path.append(r"C:\Users\jebollen\OneDrive - UGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\CircAutogen\circulatory_autogen\CA_user")


import generate_modules_files

print("Imports done")

new = True

this_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
CA_root = r"C:\Users\jebollen\OneDrive - UGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\CircAutogen\circulatory_autogen"
src_path = os.path.join(CA_root, "src")
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))
CA_root = os.path.join(CA_root, "CA_user", "smc_kapela")
resources_dir = os.path.join(CA_root, "resources") 
generated_models_dir = os.path.join(CA_root, "generated_models")
param_id_output_dir = r"C:\Users\jebollen\OneDrive - UGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\CircAutogen\circulatory_autogen\param_id_output"
print("Paths done")

# Model identifiers
model_name = "smc_kapela_VoltageClamp"
input_param_file = f"{model_name}_parameters.csv"

if new:
    generate_modules_files.create_modules_files(dir_model=CA_root, dir_output=CA_root, name_prefix=model_name, data_ref=input_param_file, comp_name="all")

# Base user inputs (this shows all the settings that can be changed)
inp_data_dict = get_default_inp_data_dict(model_name, input_param_file, resources_dir)

inp_data_dict["generated_models_dir"] = generated_models_dir
inp_data_dict["generated_models_subdir"] = os.path.join(generated_models_dir, model_name)
inp_data_dict["model_path"] = os.path.join(inp_data_dict["generated_models_subdir"], f"{model_name}.cellml")
inp_data_dict["param_id_output_dir"] = param_id_output_dir
# TEMPORARY FOR THIS TUTORIAL
inp_data_dict['DEBUG'] = True
inp_data_dict["sim_time"] = 10


solver_info = {
    "dt_solver": 2e-3,
}

inp_data_dict['solver_info'] = solver_info

print('inp_data_dict set')
pprint.pprint(inp_data_dict)

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
        "smc_kapela_VoltageClamp/I_Kv"
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

# now create the obs data creator object for creating a dictionary that contains the data you will fit towards
obs_data_creator = ObsDataCreator()

# Protocol info (this defines your times and changes of inputs in the experiment)
pre_times = [10] # the pre time for each experiment (period is 0.9s, so this is a multiple of the period
sim_times =[[0.300]] # the sim time for each subexperiment


data_dir = r"C:\Users\jebollen\OneDrive - UGent\Documents\Research\1-WP\CellModels\0_ABI_SMC\Data_rats"
gt = np.loadtxt(os.path.join(data_dir, "I_Kv_corrected.csv"), delimiter=",")
time_gt = gt[:,0].flatten()
I_Kv_gt = gt[:,1].flatten()

obs_dt = float((time_gt[1] - time_gt[0]) / 1000.0) # convert from ms to s

entry = {
    "variable": "smc_kapela_VoltageClamp/I_Kv",
    "name_for_plotting": r"I\_Kv",
    "data_type": "constant",
    "unit": "pA",
    "value": 142.783,
    "obs_dt": obs_dt,
    "plot_type": "horizontal",
    "std": 0.0001,
    "operands": ["smc_kapela_VoltageClamp/I_Kv"],
    "operation": "max"
}

entry1 = {
    "variable": "smc_kapela_VoltageClamp/I_Kv",
    "name_for_plotting": r"I\_Kv",
    "data_type": "constant",
    "unit": "pA",
    "value": -0.00001,
    "obs_dt": obs_dt,
    "plot_type": "horizontal",
    "std": 0.0001,
    "operands": ["time","smc_kapela_VoltageClamp/I_Kv"],
    "operation": "change_after_Peak"
}

entry2 = {
    "variable": "smc_kapela_VoltageClamp/I_Kv",
    "name_for_plotting": r"I\_Kv",
    "data_type": "constant",
    "unit": "ms",
    "value": 17.383,
    "obs_dt": obs_dt,
    "plot_type": "vertical",
    "std": 0.0001,
    "operands": ["time","smc_kapela_VoltageClamp/I_Kv"],
    "operation": "time_to_63_percent_peak"
}

entry3 = {  
  "variable": "I_Kv",  
  "name_for_plotting": r"I\_Kv",  
  "data_type": "series",  
  "operation": "null",  
  "operands": ["smc_kapela_VoltageClamp/I_Kv"],  
  "unit": "pA",  
  "weight": float(1.0),  
  "value": I_Kv_gt.tolist(),  
  "std": np.ones(len(I_Kv_gt)).tolist(),  
  "obs_dt": obs_dt,  
  "experiment_idx": 0,  
  "subexperiment_idx": 0  
}

inp_data_dict["param_id_method"] = "genetic_algorithm"

inp_data_dict['do_mcmc'] = True  
inp_data_dict['mcmc_options'] = {  
    'num_steps': 1000,  
    'num_walkers': 64,  
    'cost_type': 'MSE',  
    'cost_convergence': 0.001  
}

#obs_data_creator.add_data_item(entry)
#obs_data_creator.add_data_item(entry1)
#obs_data_creator.add_data_item(entry2)
obs_data_creator.add_data_item(entry3)
obs_data_dict = obs_data_creator.get_obs_data_dict()
param_id = CVS0DParamID.init_from_dict(inp_data_dict)

params_file = os.path.join(resources_dir, f"{model_name}_parameters_IKv.csv")  

def change_after_Peak(time, input, series_output=False):
    """  
    Calculates the change in the signal after it reaches its peak value.  
    """  
    if series_output:  
        return input  
    peak_index = np.argmax(input)
    end_time_idx = np.argmin(np.abs(time - (time[peak_index] + 10)))  # look for the value 10s after the peak
    if peak_index == len(input) - 1:  
        return 0.0  
    return input[end_time_idx] - input[peak_index]

def time_to_63_percent_peak(time, input, series_output=False):  
    """  
    Calculates the time when the signal first reaches 63% of its peak value.  
    """  
    if series_output:  
        return input  
    peak_value = np.max(input)  
    threshold = 0.632 * peak_value  
    cross_indices = np.where(input >= threshold)[0]  
    if len(cross_indices) == 0:  
        return time[-1]  
    return time[cross_indices[0]]


param_id.add_user_operation_func(time_to_63_percent_peak)
param_id.add_user_operation_func(change_after_Peak)

def read_params(filepath):
    all_params = None
    df = pd.read_csv(filepath)
    all_params = df.to_dict(orient="records")
    return all_params

all_params = read_params(params_file)
print(f"Loaded {len(all_params)} parameters")
 
params_for_id_dict = []  

for param in all_params:  
    name = param['variable_name'][:-len(model_name)-1]
    # Create entry with reasonable bounds (you may need to adjust these)  
    entry = {  
        "vessel_name": 'smc_kapela_VoltageClamp',  
        "param_name": name,  
        "min": param['min'],
        "max": param['max'],  # Use existing max or default  
        "name_for_plotting": name.replace("_", r"\_")  # Escape underscores for plotting
    }  
    params_for_id_dict.append(entry)  


print(f"Loaded {len(params_for_id_dict)} parameters for sensitivity analysis")

df = pd.read_csv(os.path.join(resources_dir, "experimental_protocol.csv"))     
params_to_change = {}    
for _, row in df.iterrows():    
    exp_idx = int(row['Experiment'])    
    sub_idx = int(row['subexperiment'])    
    param_name = model_name + "/" + row['variable']    
    value = row['value']    
    if pd.isna(value):    
        continue    
    if param_name not in params_to_change:    
        max_exp = df['Experiment'].max()    
        max_sub = df.groupby('Experiment')['subexperiment'].max().max()    
        params_to_change[param_name] = [[None] * (max_sub + 1) for _ in range(max_exp + 1)]    
    # Convert to native Python type before storing  
    params_to_change[param_name][exp_idx][sub_idx] = float(value) if not pd.isna(value) else None  
  
obs_data_creator.add_protocol_info(pre_times, sim_times, params_to_change, experiment_labels=["exp0"])  
obs_data_dict = obs_data_creator.get_obs_data_dict()

# Optimiser options (adjust as needed)
optimiser_options = {
    "num_calls_to_function": 20000,
    "cost_convergence": 0.001,
    "max_patience": 10,
    "cost_type": "MSE",
}

param_id.set_optimiser_options(optimiser_options)
param_id.set_ground_truth_data(obs_data_dict)
param_id.set_param_id_method(inp_data_dict["param_id_method"])

# now add the params to the param id object
param_id.set_params_for_id(params_for_id_dict)
params_for_id_subset = [ entry for entry in params_for_id_dict]
param_id.set_params_for_id(params_for_id_subset)

# Run calibration
param_id.run()

# Shorten file_prefix to avoid Windows MAX_PATH limit
param_id.file_prefix = f"smc_IKv_{datetime.today().strftime('%m%d')}"

print("STARTING WITH MCMC")
sys.stdout.flush()
# Run MCMC after GA — MUST happen BEFORE simulate_with_best_param_vals()
# because simulate_once() calls exit() on worker ranks!
if inp_data_dict.get('do_mcmc', False):
    import param_id.paramID as paramID_module
    from param_id.paramID import OpencorMCMC
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Get best GA params (broadcast from rank 0 to ensure consistency)
    if rank == 0:
        best_param_vals = param_id.get_best_param_vals()
        print(f"GA best params: {best_param_vals}")
        sys.stdout.flush()
    else:
        best_param_vals = None
    best_param_vals = comm.bcast(best_param_vals, root=0)

    # Broadcast output_dir from rank 0 (it's None on workers)
    if rank == 0:
        output_dir_for_mcmc = param_id.output_dir
    else:
        output_dir_for_mcmc = None
    output_dir_for_mcmc = comm.bcast(output_dir_for_mcmc, root=0)

    # Get the internal OpencorParamID that already has a working simulation
    internal = param_id.param_id

    # Create an OpencorMCMC WITHOUT calling __init__ (avoids re-initializing OpenCOR)
    mcmc_obj = object.__new__(OpencorMCMC)
    # Copy all attributes from the existing OpencorParamID
    mcmc_obj.__dict__.update(internal.__dict__)

    # Set MCMC-specific attributes that OpencorMCMC.__init__ would set
    mcmc_options = inp_data_dict.get('mcmc_options', {
        'num_steps': 10000,
        'num_walkers': 64,
    })
    mcmc_obj.sampler = None
    mcmc_obj.mcmc_options = mcmc_options
    mcmc_obj.set_best_param_vals(best_param_vals)
    mcmc_obj.output_dir = output_dir_for_mcmc

    # Set the global mcmc_object on ALL ranks before run()
    paramID_module.mcmc_object = mcmc_obj

    if rank == 0:
        print(f"Starting MCMC: {mcmc_obj.mcmc_options['num_walkers']} walkers x {mcmc_obj.mcmc_options['num_steps']} steps")
        print(f"num_params: {mcmc_obj.num_params}")
        sys.stdout.flush()

    try:
        mcmc_obj.run()
        if rank == 0:
            print("MCMC completed successfully")
            sys.stdout.flush()
    except Exception as e:
        import traceback
        print(f"MCMC run failed on rank {rank}:")
        traceback.print_exc()
        sys.stdout.flush()

# Now do simulate_with_best_param_vals and plotting
# (this calls exit() on worker ranks, so nothing can use MPI after this)
param_id.simulate_with_best_param_vals()
plot_dir = Path(param_id_output_dir) / "param_id_plots"
plot_dir.mkdir(parents=True, exist_ok=True)
param_id.plot_dir = str(plot_dir)
# Ensure plots_param_id subfolder exists under output_dir (used by plot_mcmc chain plot)
os.makedirs(os.path.join(param_id.output_dir, 'plots_param_id'), exist_ok=True)
param_id.plot_outputs()

# Plot MCMC results (only rank 0 survives to here)
if inp_data_dict.get('do_mcmc', False):
    try:
        samples_result = param_id.get_mcmc_samples()
        if samples_result is not None:
            param_id.plot_mcmc()
            print(f"MCMC plots saved successfully in {plot_dir}")
        else:
            print("MCMC samples not available — skipping MCMC plots.")
    except Exception as e:
        import traceback
        print(f"MCMC plotting failed:")
        traceback.print_exc()
    sys.stdout.flush()

print("All done!")
sys.stdout.flush()
os._exit(0)  # Clean exit without MPI finalize error