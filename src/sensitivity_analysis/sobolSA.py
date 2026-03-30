'''
@author: Mohammad H. Shafieizadegan
@ reference: https://salib.readthedocs.io/en/latest/index.html
'''

import json
import os
import sys
from sys import exit
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
import math as math
try:
    import opencor as oc
    opencor_available = True
except:
    opencor_available = False
    pass
if opencor_available:
    from solver_wrappers.opencor_helper import SimulationHelper as OpenCORSimulationHelper
else:
    from solver_wrappers.python_solver_helper import SimulationHelper as PythonSimulationHelper
from SALib.sample import saltelli
import pandas as pd
from SALib.analyze import sobol
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from parsers.PrimitiveParsers import scriptFunctionParser
from mpi4py import MPI
from parsers.PrimitiveParsers import CSVFileParser
import csv
from tqdm import tqdm  # make sure tqdm is installed

class sobol_SA():

    """
        A class for performing sensitivity analysis
        How to use:
        1. Initialize the class with the model path, output names, solver info, sensitivity analysis configuration, protocol info, time step, and save path.
        2. Call the `run` method with a feature extractor function and any additional arguments needed by that function.
        3. The 'run' method will generate sobol indices
        4. You can plot the results using the `plot_sobol_first_order_idx` and `plot_sobol_S2_idx` methods.
    """

    def _is_rank0(self):
        try:
            return MPI.COMM_WORLD.Get_rank() == 0
        except Exception:
            return True

    def _rank0_print(self, *args, **kwargs):
        if self._is_rank0():
            print(*args, **kwargs)

    def __init__(self, model_path, model_out_names, solver_info, SA_cfg, dt, save_path, 
                 param_id_path = None, params_for_id_path=None, use_MPI = False, verbose=False, ga_options=None,
                 sim_time=2.0, pre_time=20.0):

        """
        Initializes the Sensitivity_analysis class.
        Parameters:
            model_path (str): Path to the model file.
            model_out_names (list): Names of the model outputs to be analyzed.
            solver_info (dict): Solver configuration parameters.
            SA_cfg (dict): Configuration for sensitivity analysis, including sample type, number of samples,
                           parameter names, and their bounds.
            protocol_info (dict): Information about the simulation protocol, including simulation times and pre-times.
            dt (float): Time step for the simulation.
            save_path (str): Directory where results will be saved.
            verbose (bool): If True, prints additional information during execution.
        """

        self.model_path = model_path
        self.output_dir = None
        self.verbose = verbose
        self.save_path = save_path

        self.solver_info = solver_info
        self.sample_type = SA_cfg["sample_type"]
        self.num_params = None
        self.model_output_names = model_out_names
        # For backwards compatibility, accept both ga_options and optimiser_options
        # optimiser_options takes precedence
        self.ga_options = ga_options  # Keep for backwards compatibility
        self.optimiser_options = None  # Will be set if passed
        self.protocol_info = None
        self.dt = dt
        
        # set up observables functions
        sfp = scriptFunctionParser()
        self.operation_funcs_dict = sfp.get_operation_funcs_dict()
        self.__set_obs_names_and_df(param_id_path, sim_time=sim_time, pre_time=pre_time)

        # set up opencor simulation
        if self.protocol_info['sim_times'][0][0] is not None:
            self.sim_time = self.protocol_info['sim_times'][0][0]
        else:
            # set temporary sim time, just to initialise the sim_helper
            self.sim_time = 0.001
        if self.protocol_info['pre_times'][0] is not None:
            self.pre_time = self.protocol_info['pre_times'][0]
        else:
            # set temporary pre time, just to initialise the sim_helper
            self.pre_time = 0.001

        self.sim_helper = self.initialise_sim_helper()
        self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)
        self.n_steps = int(self.sim_time/self.dt)

        self.set_output_dir(save_path)

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()
        self.use_mpi = use_MPI

        self.params_for_id_path = params_for_id_path
        if self.params_for_id_path:
            self.__set_and_save_param_names()
        self.SA_cfg = self.create_SA_cfg(self.sample_type, SA_cfg["num_samples"])

        self.series_indices = None  # Initialize tracking structure

    def create_SA_cfg(self, sample_type, num_samples):
        
        # Use param_id_info to build SA_cfg dynamically
        if not hasattr(self, "param_id_info") or not self.param_id_info:
            raise ValueError("param_id_info is not set. Please run __set_and_save_param_names() first.")

        SA_cfg = {
            "sample_type": sample_type,
            "param_names": [name[0] if isinstance(name, list) else name for name in self.param_id_info["param_names"]],
            "num_samples": num_samples,
            "param_mins": list(self.param_id_info["param_mins"]),
            "param_maxs": list(self.param_id_info["param_maxs"])
        }

        # if self.verbose:
        #     print("Sensitivity Analysis Configuration:")
        #     print(json.dumps(SA_cfg, indent=4))

        self.num_params = len(SA_cfg["param_names"])

        return SA_cfg

    def __set_and_save_param_names(self, idxs_to_ignore=None):
        # This should also be a function under parsers.

        # Each entry in param_names is a name or list of names that gets modified by one parameter
        self.param_id_info = {}
        if self.params_for_id_path:
            csv_parser = CSVFileParser()
            input_params = csv_parser.get_data_as_dataframe_multistrings(self.params_for_id_path)
            self.param_id_info["param_names"] = []
            param_names_for_gen = []
            for II in range(input_params.shape[0]):
                if idxs_to_ignore is not None:
                    if II in idxs_to_ignore:
                        continue
                self.param_id_info["param_names"].append([input_params["vessel_name"][II][JJ] + '/' +
                                               input_params["param_name"][II]for JJ in
                                               range(len(input_params["vessel_name"][II]))])

                if input_params["vessel_name"][II][0] == 'global':
                    param_names_for_gen.append([input_params["param_name"][II]])

                else:
                    param_names_for_gen.append([input_params["param_name"][II] + '_' +
                                                input_params["vessel_name"][II][JJ] # re.sub('_T$', '', input_params["vessel_name"][II][JJ])
                                                for JJ in range(len(input_params["vessel_name"][II]))])

            # set param ranges from file and strings for plotting parameter names
            if idxs_to_ignore is not None:
                self.param_id_info["param_mins"] = np.array([float(input_params["min"][JJ]) for JJ in range(input_params.shape[0])
                                            if JJ not in idxs_to_ignore])
                self.param_id_info["param_maxs"] = np.array([float(input_params["max"][JJ]) for JJ in range(input_params.shape[0])
                                            if JJ not in idxs_to_ignore])
                if "name_for_plotting" in input_params.columns:
                    self.param_id_info["param_names_for_plotting"] = np.array([input_params["name_for_plotting"][JJ]
                                                            for JJ in range(input_params.shape[0])
                                                            if JJ not in idxs_to_ignore])
                else:
                    self.param_id_info["param_names_for_plotting"] = np.array([self.param_id_info["param_names"][JJ][0]
                                                            for JJ in range(len(self.param_id_info["param_names"]))
                                                            if JJ not in idxs_to_ignore])
            else:
                self.param_id_info["param_mins"] = np.array([float(input_params["min"][JJ]) for JJ in range(input_params.shape[0])])
                self.param_id_info["param_maxs"] = np.array([float(input_params["max"][JJ]) for JJ in range(input_params.shape[0])])
                if "name_for_plotting" in input_params.columns:
                    self.param_id_info["param_names_for_plotting"] = np.array([input_params["name_for_plotting"][JJ]
                                                            for JJ in range(input_params.shape[0])])
                else:
                    self.param_id_info["param_names_for_plotting"] = np.array([param_name[0] for param_name in self.param_id_info["param_names"]])

            # set param_priors
            if "prior" in input_params.columns:
                self.param_id_info["param_prior_types"] = np.array([input_params["prior"][JJ] for JJ in range(input_params.shape[0])])
            else:
                self.param_id_info["param_prior_types"] = np.array(["uniform" for JJ in range(input_params.shape[0])])


        else:
            self._rank0_print(f'params_for_id_path cannot be None, exiting')

        if self.rank == 0:
            with open(os.path.join(self.output_dir, 'param_names.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(self.param_id_info["param_names"])
            with open(os.path.join(self.output_dir, 'param_names_for_gen.csv'), 'w') as f:
                wr = csv.writer(f)
                wr.writerows(param_names_for_gen)
        return

    def initialise_sim_helper(self):
        if opencor_available:
            return OpenCORSimulationHelper(self.model_path, self.dt, self.sim_time,
                                solver_info=self.solver_info, pre_time=self.pre_time)
        else:
            return PythonSimulationHelper(self.model_path, self.dt, self.sim_time,
                                solver_info=self.solver_info, pre_time=self.pre_time)

    def __set_obs_names_and_df(self, param_id_obs_path, pre_time=None, sim_time=None):
        # TODO this function should be in the parsing section. as it parses the 
        # ground truth data.
        # TODO it should also be cleaned up substantially.
        """_summary_

        Args:
            param_id_obs_path (_type_): _description_
            pre_time (_type_): _description_
            sim_time (_type_): _description_
        """
        with open(param_id_obs_path, encoding='utf-8-sig') as rf:
            json_obj = json.load(rf)
        if type(json_obj) == list:
            self.gt_df = pd.DataFrame(json_obj)
            self.protocol_info = {"pre_times": [pre_time], 
                                    "sim_times": [[sim_time]],
                                    "params_to_change": [[None]]}
            self.prediction_info = {'names': [],
                                    'units': [],
                                    'names_for_plotting': [],
                                    'experiment_idxs': []}
        elif type(json_obj) == dict:
            if 'data_items' in json_obj.keys():
                self.gt_df = pd.DataFrame(json_obj['data_items'])
            elif 'data_item' in json_obj.keys():
                self.gt_df = pd.DataFrame(json_obj['data_item']) # should be data_items but accept this
            else:
                self._rank0_print("data_items not found in json object. ",
                      "Please check that data_items is the key for the list of data items")
            if 'protocol_info' in json_obj.keys():
                self.protocol_info = json_obj['protocol_info']
                if "sim_times" not in self.protocol_info.keys():
                    self.protocol_info["sim_times"] = [[sim_time]]
                if "pre_times" not in self.protocol_info.keys():
                    self.protocol_info["pre_times"] = [pre_time]
            else:
                if pre_time is None or sim_time is None:
                    self._rank0_print("protocol_info not found in json object. ",
                          "If this is the case sim_time and pre_time must be set",
                          "in the user_inputs.yaml file")
                    exit()

                self.protocol_info = {"pre_times": [pre_time], 
                                      "sim_times": [[sim_time]],
                                      "params_to_change": [[None]]}
            if 'prediction_items' in json_obj.keys():
                self.prediction_info = {'names': [],
                                        'units': [],
                                        'names_for_plotting': [],
                                        'experiment_idxs': []}

                for entry in json_obj['prediction_items']:
                    if 'variable' in entry.keys():
                        self.prediction_info['names'].append(entry['variable'])
                    else:
                        self._rank0_print('"variable" not found in prediction item in obs_data.json file, ',
                              'exitiing') 
                        exit()
                    if 'unit' in entry.keys():
                        self.prediction_info['units'].append(entry['unit'])
                    else:
                        self._rank0_print('"unit" not found in prediction item in obs_data.json file, ',
                              'exitiing') 
                        exit()
                    if 'name_for_plotting' in entry.keys():
                        self.prediction_info['names_for_plotting'].append(entry['name_for_plotting'])
                    else:
                        self.prediction_info['names_for_plotting'].append(entry['variable'])
                    if 'experiment_idx' in entry.keys():
                        self.prediction_info['experiment_idxs'].append(entry['experiment_idx'])
                    else:
                        self.prediction_info['experiment_idxs'].append(0)
            else:
                self.prediction_info = None
        else:
            self._rank0_print(f"unknown data type for imported json object of {type(json_obj)}")
        
        self.obs_info = {}
        self.obs_info["obs_names"] = [self.gt_df.iloc[II]["variable"] for II in range(self.gt_df.shape[0])]

        # OBSOLETE self.obs_types = [self.gt_df.iloc[II]["obs_type"] for II in range(self.gt_df.shape[0])]
        self.obs_info["data_types"] = [self.gt_df.iloc[II]["data_type"] for II in range(self.gt_df.shape[0])]
        self.obs_info["units"] = [self.gt_df.iloc[II]["unit"] for II in range(self.gt_df.shape[0])]
        self.obs_info["experiment_idxs"] = [self.gt_df.iloc[II]["experiment_idx"] if "experiment_idx" in 
                                            self.gt_df.iloc[II].keys() else 0 for II in range(self.gt_df.shape[0])]
        self.obs_info["subexperiment_idxs"] = [self.gt_df.iloc[II]["subexperiment_idx"] if "subexperiment_idx" in
                                               self.gt_df.iloc[II].keys() else 0 for II in range(self.gt_df.shape[0])]

        # get plotting color, asign to randomish color if not defined
        # list of all possible colors
        possible_colors = ['b', 'g', 'c', 'm', 'y', 
                           'tab:brown', 'tab:pink', 'tab:olive', 'tab:orange'] # don't include red or black, 
                                                    # because they are used for plotting the series
        self.obs_info["plot_colors"] = [self.gt_df.iloc[II]["plot_color"] if "plot_color" in 
                                        self.gt_df.iloc[II].keys() else possible_colors[II%len(possible_colors)] 
                                        for II in range(self.gt_df.shape[0])]
        self.obs_info["plot_type"] = []

        dt_list = []  
        for II in range(self.gt_df.shape[0]):  
            if self.gt_df.iloc[II]["data_type"] == "series":  
                if "obs_dt" not in self.gt_df.iloc[II].keys():  
                    self._rank0_print("dt not found in obs_data.json for series data, exiting")  
                    exit()  
                dt_list.append(self.gt_df.iloc[II]["obs_dt"])  
        
        self.obs_info["obs_dt"] = np.array(dt_list)  
        
        # Validate obs_dt against simulation dt  
        if len(self.obs_info["obs_dt"]) > 0:  
            if min(self.obs_info["obs_dt"]) < self.dt:  
                self._rank0_print("one of the dt in obs_data.json is less than the dt in user_inputs.yaml, "  
                                "the output timestep defined in user_inputs.yaml must be less than the "  
                                "smallest dt for your data. Exiting")  
                exit()

        # get plotting type
        # TODO make the plot_types operation_funcs so the user can defined how they are plotted.
        warning_printed = False
        for II in range(self.gt_df.shape[0]):
            if "plot_type" not in self.gt_df.iloc[II].keys():
                if self.gt_df.iloc[II]["data_type"] == "constant":
                    if not warning_printed:
                        self._rank0_print('constant data types plot type defaults to horizontal lines',
                            'change "plot_type" in obs_data.json to change this')
                        warning_printed = True
                    self.obs_info["plot_type"].append("horizontal")
                elif self.gt_df.iloc[II]["data_type"] == "prob_dist":
                    if not warning_printed:
                        self._rank0_print('prob_dist data types plot type defaults to horizontal lines',
                            'change "plot_type" in obs_data.json to change this')
                        warning_printed = True
                    self.obs_info["plot_type"].append("horizontal")
                elif self.gt_df.iloc[II]["data_type"] == "series":
                    self.obs_info["plot_type"].append("series")
                elif self.gt_df.iloc[II]["data_type"] == "frequency":
                    self.obs_info["plot_type"].append("frequency")
                elif self.gt_df.iloc[II]["data_type"] == "plot_dist":
                    self.obs_info["plot_type"].append("horizontal")
                else:
                    self._rank0_print(f'data type {self.gt_df.iloc[II]["data_type"]} not recognised')
            else:
                self.obs_info["plot_type"].append(self.gt_df.iloc[II]["plot_type"])
                if self.obs_info["plot_type"][II] in ["None", "null", "Null", "none", "NONE"]:
                    self.obs_info["plot_type"][II] = None

        self.obs_info["operations"] = []
        self.obs_info["names_for_plotting"] = []
        self.obs_info["operands"] = []
        self.obs_info["freqs"] = []
        self.obs_info["operation_kwargs"] = []
        # below we remove the need for obs_types, but keep it backwards compatible so 
        # previous specifications of obs_type = mean etc should still work
        for II in range(self.gt_df.shape[0]):
            if "operation" not in self.gt_df.iloc[II].keys() or \
                    self.gt_df.iloc[II]["operation"] in ["Null", "None", "null", "none", "", "nan", np.nan]:
                if "obs_type" in self.gt_df.iloc[II].keys():
                    if self.gt_df.iloc[II]["obs_type"] == "series":
                        self.obs_info["operations"].append(None)
                        if "operands" in self.gt_df.iloc[II].keys():
                            self.obs_info["operands"].append(self.gt_df.iloc[II]["operands"])
                        else:
                            self.obs_info["operands"].append(None)
                    elif self.gt_df.iloc[II]["obs_type"] == "frequency":
                        self.obs_info["operations"].append(None)
                        if "operands" in self.gt_df.iloc[II].keys():
                            self.obs_info["operands"].append(self.gt_df.iloc[II]["operands"])
                        else:
                            self.obs_info["operands"].append(None)
                    # TODO remove these eventually when I get rid of obs_type
                    elif self.gt_df.iloc[II]["obs_type"] == "min":
                        self.obs_info["operations"].append("min")
                        self.obs_info["operands"].append([self.gt_df.iloc[II]["variable"]])
                    elif self.gt_df.iloc[II]["obs_type"] == "max":
                        self.obs_info["operations"].append("max")
                        self.obs_info["operands"].append([self.gt_df.iloc[II]["variable"]])
                    elif self.gt_df.iloc[II]["obs_type"] == "mean":
                        self.obs_info["operations"].append("mean")
                        self.obs_info["operands"].append([self.gt_df.iloc[II]["variable"]])
                else:
                    self.obs_info["operations"].append(None)
                    if "operands" in self.gt_df.iloc[II].keys():
                        self.obs_info["operands"].append(self.gt_df.iloc[II]["operands"])
                    else:
                        self.obs_info["operands"].append(None)
            elif self.gt_df.iloc[II]["operation"] in ["Null", "None", "null", "none", ""]:
                self.obs_info["operations"].append(None)
                self.obs_info["operands"].append(None)
            else:
                self.obs_info["operations"].append(self.gt_df.iloc[II]["operation"])
                self.obs_info["operands"].append(self.gt_df.iloc[II]["operands"])

            if "frequencies" not in self.gt_df.iloc[II].keys():
                self.obs_info["freqs"].append(None)
            else:
                self.obs_info["freqs"].append(self.gt_df.iloc[II]["frequencies"])

            if "name_for_plotting" in self.gt_df.iloc[II].keys():
                self.obs_info['names_for_plotting'].append(self.gt_df.iloc[II]["name_for_plotting"])
            else:
                self.obs_info['names_for_plotting'].append(self.obs_info["obs_names"][II])

            if "operation_kwargs" in self.gt_df.iloc[II].keys() and self.gt_df.iloc[II]["operation_kwargs"] \
                    not in ["Null", "None", "null", "none", "", np.nan]:
                self.obs_info["operation_kwargs"].append(self.gt_df.iloc[II]["operation_kwargs"])
            else:
                self.obs_info["operation_kwargs"].append({})

        self.obs_info["num_obs"] = len(self.obs_info["obs_names"])

        # how much to weight the different observable errors by
        self.obs_info["weight_const_vec"] = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                          if self.gt_df.iloc[II]["data_type"] == "constant"])

        self.obs_info["weight_series_vec"] = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                           if self.gt_df.iloc[II]["data_type"] == "series"])

        self.obs_info["weight_amp_vec"] = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                           if self.gt_df.iloc[II]["data_type"] == "frequency"])
        
        self.obs_info["weight_prob_dist_vec"] = np.array([self.gt_df.iloc[II]["weight"] for II in range(self.gt_df.shape[0])
                                          if self.gt_df.iloc[II]["data_type"] == "prob_dist"])

        weight_phase_list = [] 
        for II in range(self.gt_df.shape[0]):
            if self.gt_df.iloc[II]["data_type"] == "frequency":
                if "phase_weight" not in self.gt_df.iloc[II].keys():
                    weight_phase_list.append(1)
                else:
                    weight_phase_list.append(self.gt_df.iloc[II]["phase_weight"])
        self.obs_info["weight_phase_vec"] = np.array(weight_phase_list)

        # set the cost type for each observable
        self.obs_info["cost_type"] = []
        for II in range(self.gt_df.shape[0]):
            if "cost_type" in self.gt_df.iloc[II].keys() and self.gt_df.iloc[II]["cost_type"] not in [np.nan, None, "None", ""]:
                self.obs_info["cost_type"].append(self.gt_df.iloc[II]["cost_type"])
            else:
                # Check optimiser_options first, then ga_options for backwards compatibility
                options_dict = self.optimiser_options if self.optimiser_options is not None else self.ga_options
                if options_dict is not None:
                    if "cost_type" in options_dict.keys():
                        self.obs_info["cost_type"].append(options_dict["cost_type"]) # default to cost type in options
                    else:
                        self.obs_info["cost_type"].append("MSE") # default to mean squared error
                elif self.mcmc_options is not None:
                    if "cost_type" in self.mcmc_options.keys():
                        self.obs_info["cost_type"].append(self.mcmc_options["cost_type"]) # default to cost type in mcmc_options
                    else:
                        self.obs_info["cost_type"].append("MSE") # default to mean squared error
                else:
                    self._rank0_print("cost_type not found in obs_data.json, ga_options, or mcmc_options, exiting")
                    exit()



        # preprocess information in the protocol_info dataframe
        self.protocol_info['num_experiments'] = len(self.protocol_info["sim_times"])
        self.protocol_info['num_sub_per_exp'] = [len(self.protocol_info["sim_times"][II]) for II in range(self.protocol_info["num_experiments"])]
        self.protocol_info['num_sub_total'] = sum(self.protocol_info['num_sub_per_exp'])

        # calculate total experiment sim times
        self.protocol_info["total_sim_times_per_exp"] = []
        self.protocol_info["tSims_per_exp"] = []
        self.protocol_info["num_steps_total_per_exp"] = []

        for exp_idx in range(self.protocol_info['num_experiments']):
            total_sim_time = np.sum([self.protocol_info["sim_times"][exp_idx][II] for
                            II in range(self.protocol_info["num_sub_per_exp"][exp_idx])])
            num_steps_total = int(total_sim_time/self.dt)
            tSim_per_exp = np.linspace(0.0, total_sim_time, num_steps_total + 1)
            self.protocol_info["total_sim_times_per_exp"].append(total_sim_time)
            self.protocol_info["tSims_per_exp"].append(tSim_per_exp)
            self.protocol_info["num_steps_total_per_exp"].append(num_steps_total)
            

        if "experiment_colors" not in self.protocol_info.keys():
            self.protocol_info["experiment_colors"] = ['r']
            if self.protocol_info['num_experiments'] > 1:
                self.protocol_info["experiment_colors"] = ['r']*self.protocol_info['num_experiments']
        else:
            if len(self.protocol_info["experiment_colors"]) != self.protocol_info['num_experiments']:
                self._rank0_print('experiment_colors in obs_data.json not the same length as num_experiments, exiting')
                exit()

        if "experiment_labels" in self.protocol_info.keys():
            if len(self.protocol_info["experiment_labels"]) != self.protocol_info['num_experiments']:
                self._rank0_print('experiment_labels in obs_data.json not the same length as num_experiments, exiting')
                exit()
        else:
            self.protocol_info["experiment_labels"] = [None]
            if self.protocol_info['num_experiments'] > 1:
                self.protocol_info["experiment_labels"] = [None]*self.protocol_info['num_experiments']
        
        # set experiment and subexperiment idxs to 0 if they are not defined. print warning if multiple subexperiments
        for II in range(self.gt_df.shape[0]):
            if "experiment_idx" not in self.gt_df.iloc[II].keys():
                self.gt_df["experiment_idx"] = 0
                if self.protocol_info['num_sub_total'] > 1:
                    self._rank0_print(f'experiment_idx not found in obs_data.json entry {self.gt_df.iloc[II]["variable"]}, '
                          'but multiple experiments are defined.',
                          'Setting experiment_idx to 0 for all data points')
            if "subexperiment_idx" not in self.gt_df.iloc[II].keys():
                self.gt_df["subexperiment_idx"] = 0
                if self.protocol_info['num_sub_total'] > 1:
                    self._rank0_print(f'subexperiment_idx not found in obs_data.json entry {self.gt_df.iloc[II]["variable"]}, '
                          'but multiple subexperiments are defined.',
                          'Setting subexperiment_idx to 0 for all data points')
        
        # calculate the mapping from sub and experiment idx to the weight of the observable for that subexperiment
        const_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]
        series_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]
        amp_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]
        phase_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]
        prob_dist_map = [[[] for sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx])]
                     for exp_idx in range(self.protocol_info['num_experiments'])]

        for exp_idx in range(self.protocol_info['num_experiments']):
            for this_sub_idx in range(self.protocol_info['num_sub_per_exp'][exp_idx]):

                for II in range(self.gt_df.shape[0]):
                    if self.gt_df.iloc[II]["data_type"] == "constant":
                        if self.gt_df.iloc[II]["experiment_idx"] == exp_idx and \
                            self.gt_df.iloc[II]["subexperiment_idx"] == this_sub_idx:
                            const_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                        else:
                            # if the data point is not in assigned to this experiment/subexperiment, 
                            # set the weight mapping to 0, so it doesn't influence the cost in this 
                            # subexperiment
                            const_map[exp_idx][this_sub_idx].append(0.0)
                    if self.gt_df.iloc[II]["data_type"] == "series":
                        if self.gt_df.iloc[II]["experiment_idx"] == exp_idx and \
                            self.gt_df.iloc[II]["subexperiment_idx"] == this_sub_idx:
                            series_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                        else:
                            series_map[exp_idx][this_sub_idx].append(0.0)

                    if self.gt_df.iloc[II]["data_type"] == "frequency":
                        if self.gt_df.iloc[II]["experiment_idx"] == exp_idx and \
                            self.gt_df.iloc[II]["subexperiment_idx"] == this_sub_idx:
                            amp_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                            if "phase_weight" not in self.gt_df.iloc[II].keys():
                                # if there is no phase weight, weight it the same as the amplitude
                                phase_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                            else:
                                phase_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["phase_weight"])
                        else:
                            amp_map[exp_idx][this_sub_idx].append(0.0)
                            phase_map[exp_idx][this_sub_idx].append(0.0)

                    if self.gt_df.iloc[II]["data_type"] == "prob_dist":
                        if self.gt_df.iloc[II]["experiment_idx"] == exp_idx and \
                            self.gt_df.iloc[II]["subexperiment_idx"] == this_sub_idx:
                            prob_dist_map[exp_idx][this_sub_idx].append(self.gt_df.iloc[II]["weight"])
                        else:
                            prob_dist_map[exp_idx][this_sub_idx].append(0.0)

                # make each weight vector a numpy array
                const_map[exp_idx][this_sub_idx] = np.array(const_map[exp_idx][this_sub_idx])
                series_map[exp_idx][this_sub_idx] = np.array(series_map[exp_idx][this_sub_idx])
                amp_map[exp_idx][this_sub_idx] = np.array(amp_map[exp_idx][this_sub_idx])
                phase_map[exp_idx][this_sub_idx] = np.array(phase_map[exp_idx][this_sub_idx])
                prob_dist_map[exp_idx][this_sub_idx] = np.array(prob_dist_map[exp_idx][this_sub_idx])

        self.protocol_info["scaled_weight_const_from_exp_sub"] = const_map
        self.protocol_info["scaled_weight_series_from_exp_sub"] = series_map
        self.protocol_info["scaled_weight_amp_from_exp_sub"] = amp_map
        self.protocol_info["scaled_weight_phase_from_exp_sub"] = phase_map
        self.protocol_info["scaled_weight_prob_dist_from_exp_sub"] = prob_dist_map
        return
    
    def set_output_dir(self, path):
        
        self.output_dir = path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_samples(self):

        problem = {
            'num_vars': self.num_params,
            'names': self.SA_cfg["param_names"],
            'bounds': list(zip(self.SA_cfg["param_mins"], self.SA_cfg["param_maxs"]))
        }
        self.problem = problem

        self.num_samples = self.SA_cfg["num_samples"]

        if self.SA_cfg["sample_type"] == "saltelli":
            samples = saltelli.sample(problem, self.num_samples, calc_second_order=True)  # Enable second-order interactions
        elif self.SA_cfg["sample_type"] == "sobol":
            samples = sobol.sample(problem, self.num_samples, calc_second_order=True)  # Enable second-order interactions
        else:
            raise ValueError(f"Unsupported sample type: {self.SA_cfg['sample_type']}")
        
        return samples
    
    def run_model_and_get_results(self, param_vals):
        self.sim_helper.set_param_vals(self.SA_cfg["param_names"], param_vals)
        self.sim_helper.reset_states()
        self.sim_helper.run()

        operands = self.sim_helper.get_results(self.obs_info["operands"])

        self.sim_helper.reset_and_clear()
        # y = self.sim_helper.get_results(self.model_output_names)
        # t = self.sim_helper.tSim - self.pre_time
        # return y, t
        return operands
    
    def generate_outputs_mpi(self, samples):
        # Split samples across ranks
        n_samples = len(samples)
        n_invalid_local = 0

        samples_per_rank = n_samples // self.num_procs
        remainder = n_samples % self.num_procs

        if self.rank < remainder:
            start = self.rank * (samples_per_rank + 1)
            end = start + samples_per_rank + 1
        else:
            start = self.rank * samples_per_rank + remainder
            end = start + samples_per_rank

        local_samples = samples[start:end]

        self._rank0_print(f"[MPI Rank {self.rank}] Starting samples {start}:{end} (total {len(local_samples)})")

        local_outputs = []

        # Create a single progress bar for rank 0 only to avoid noisy output from all ranks
        with tqdm(total=len(local_samples), desc=f"Rank {self.rank}", position=self.rank, leave=True, disable=self.rank != 0) as pbar:
            for param_vals in local_samples:

                sample_invalid = False

                # --- handle single vs multi subexperiment ---
                if self.protocol_info["num_sub_total"] == 1:
                    # simple case (one experiment only)
                    self.sim_helper.set_param_vals(self.param_id_info["param_names"], param_vals)
                    self.sim_helper.reset_states()
                    success = self.sim_helper.run()

                    operands_outputs_dict = {}

                    retry_count = 0
                    max_retries = 5
                    original_MaximumStep = self.solver_info.get("MaximumStep", None)
                    original_MaximumNumberOfSteps = self.solver_info.get("MaximumNumberOfSteps", None)

                    while not success and retry_count < max_retries:

                        original_MaximumStep = self.solver_info.get("MaximumStep", None)
                        reduced_MaximumStep = original_MaximumStep / 2 if original_MaximumStep else 0.001
                        increased_MaximumNumberOfSteps = original_MaximumNumberOfSteps * 2 if original_MaximumNumberOfSteps else 1000000
                        retry_count += 1
                        # Reduce max_dt for retry
                        self.solver_info["MaximumStep"] = reduced_MaximumStep
                        self.solver_info["MaximumNumberOfSteps"] = increased_MaximumNumberOfSteps
                                
                        self.sim_helper.set_param_vals(self.param_id_info["param_names"], param_vals)
                        self.sim_helper.reset_states()
                        success = self.sim_helper.run()

                        # Restore original max_dt after retries
                        self.solver_info["MaximumStep"] = original_MaximumStep
                        self.solver_info["MaximumNumberOfSteps"] = original_MaximumNumberOfSteps
                    
                    if success:
                        operands_outputs = self.sim_helper.get_results(self.obs_info["operands"])
                        operands_outputs_dict[(0, 0)] = operands_outputs

                    else:
                        print(f"[MPI Rank {self.rank}] Simulation failed for params: {param_vals}, after {retry_count} retries")
                        # Set a flag in operands_outputs_dict to indicate failure
                        operands_outputs_dict[(0, 0)] = {"failed": True}
                        sample_invalid = True

                    # reset at the end of each experiment
                    self.sim_helper.reset_and_clear()

                else:
                    # multiple subexperiments
                    current_time = 0
                    operands_outputs_dict = {}
                    for exp_idx in range(self.protocol_info["num_experiments"]):
                        self.sim_helper.set_param_vals(self.param_id_info["param_names"], param_vals)
                        self.sim_helper.reset_states()

                        for this_sub_idx in range(self.protocol_info["num_sub_per_exp"][exp_idx]):
                            subexp_count = int(np.sum(
                                [num_sub for num_sub in self.protocol_info["num_sub_per_exp"][:exp_idx]]
                            ) + this_sub_idx)

                            self.sim_time = self.protocol_info["sim_times"][exp_idx][this_sub_idx]
                            self.pre_time = self.protocol_info["pre_times"][exp_idx]

                            if self.protocol_info["num_sub_total"] > 1:
                                if this_sub_idx == 0:
                                    self.sim_helper.update_times(self.dt, 0.0, self.sim_time, self.pre_time)
                                    current_time += self.pre_time
                                else:
                                    self.sim_helper.update_times(self.dt, current_time, self.sim_time, 0.0)

                            # set subexperiment-specific parameters
                            self.sim_helper.set_param_vals(
                                list(self.protocol_info["params_to_change"].keys()),
                                [
                                    self.protocol_info["params_to_change"][param_name][exp_idx][this_sub_idx]
                                    for param_name in self.protocol_info["params_to_change"].keys()
                                ]
                            )

                            success = self.sim_helper.run()

                            retry_count = 0
                            max_retries = 5
                            original_MaximumStep = self.solver_info.get("MaximumStep", None)
                            original_MaximumNumberOfSteps = self.solver_info.get("MaximumNumberOfSteps", None)

                            while not success and retry_count < max_retries:

                                original_MaximumStep = self.solver_info.get("MaximumStep", None)
                                reduced_MaximumStep = original_MaximumStep / 2 if original_MaximumStep else 0.001
                                increased_MaximumNumberOfSteps = original_MaximumNumberOfSteps * 2 if original_MaximumNumberOfSteps else 1000000
                                retry_count += 1
                                # Reduce max_dt for retry
                                self.solver_info["MaximumStep"] = reduced_MaximumStep
                                self.solver_info["MaximumNumberOfSteps"] = increased_MaximumNumberOfSteps
                                
                                self.sim_helper.set_param_vals(self.param_id_info["param_names"], param_vals)
                                self.sim_helper.reset_states()
                                success = self.sim_helper.run()

                            # Restore original max_dt after retries
                            self.solver_info["MaximumStep"] = original_MaximumStep
                            self.solver_info["MaximumNumberOfSteps"] = original_MaximumNumberOfSteps
                            if success:
                                current_time += self.sim_time
                                operands_outputs = self.sim_helper.get_results(self.obs_info["operands"])
                                operands_outputs_dict[(exp_idx, this_sub_idx)] = operands_outputs

                                # reset at the end of each experiment
                                if this_sub_idx == self.protocol_info["num_sub_per_exp"][exp_idx] - 1:
                                    self.sim_helper.reset_and_clear()
                            else:
                                self._rank0_print(f"[MPI Rank {self.rank}] Simulation failed for params: {param_vals}, subexp={subexp_count} after {retry_count} retries")
                                # Set a flag in operands_outputs_dict to indicate failure
                                operands_outputs_dict[(exp_idx, this_sub_idx)] = {"failed": True}
                                sample_invalid = True

                                # reset at the end of each experiment
                                if this_sub_idx == self.protocol_info["num_sub_per_exp"][exp_idx] - 1:
                                    self.sim_helper.reset_and_clear()

                features = []
                series_indices = []
                for j in range(len(self.obs_info["operations"])):
                    
                    exp_idx = self.obs_info["experiment_idxs"][j]  
                    subexp_idx = self.obs_info["subexperiment_idxs"][j]  
                    operands_outputs = operands_outputs_dict.get((exp_idx, subexp_idx), None)  
                    
                    if operands_outputs is not None and not (isinstance(operands_outputs, dict) and operands_outputs == {"failed": True}):  
                        if self.obs_info["data_types"][j] == "series" and self.obs_info["operations"][j] is None:  
                            # Expand series into individual time points  
                            series_data = operands_outputs[j][0]  
                            for t_idx in range(len(series_data)):  
                                features.append(series_data[t_idx])  
                                series_indices.append((j, t_idx))  # Track original obs index and time index  
                        else:  
                            # Handle regular operations  
                            if self.obs_info["operations"][j] is None:  
                                feature = operands_outputs[j][0]  
                            else:  
                                func = self.operation_funcs_dict[self.obs_info["operations"][j]]  
                                feature = func(*operands_outputs[j], **self.obs_info["operation_kwargs"][j])  
    
                                if feature is None or (isinstance(feature, (list, np.ndarray)) and 
                                                       np.any([f is None or (isinstance(f, (float, int)) and 
                                                                             np.isnan(f)) for f in feature])):
                                    feature = np.nanmean(features) if not np.all(np.isnan(features)) else 0.0
                                    sample_invalid = True

                            features.append(feature)  
                            series_indices.append((j, None))  # Not a series point  
                    else:  
                        features.append(np.mean(features) if features else 0)  
                        series_indices.append((j, None)) 

                if sample_invalid:
                    n_invalid_local += 1

                local_outputs.append(features)
                pbar.update(1)

        self._rank0_print(f"[MPI Rank {self.rank}] Finished processing samples {start}:{end}")

        # Gather results at rank 0
        n_invalid_samples = self.comm.reduce(n_invalid_local, op=MPI.SUM, root=0)
        all_outputs = self.comm.gather(local_outputs, root=0)

        if self.rank == 0:
            outputs = [item for sublist in all_outputs for item in sublist]
            outputs = np.array(outputs)
            self._rank0_print(f"[MPI Rank 0] Gathered and flattened all outputs. Total outputs: {outputs.shape}")
            self.series_indices = series_indices

            n_valid_samples = n_samples - n_invalid_samples
            self._rank0_print(f"[MPI Rank 0] Outputs shape: {outputs.shape}")
            self._rank0_print(f"[INFO] Valid samples: {n_valid_samples}/{n_samples}")
            self._rank0_print(f"[WARNING] Invalid samples: {n_invalid_samples}/{n_samples}")

            return outputs
        else:
            return None

    def sobol_index(self, outputs):

        if self.rank !=0:
            return None, None, None
        
        outputs = np.array(outputs)
    
        if outputs.ndim == 1:
            outputs = outputs[:, np.newaxis]  # convert to (n_samples, 1)

        n_outputs = outputs.shape[1]

        # Group features by original observable  
        obs_groups = {}  
        for idx, (obs_idx, t_idx) in enumerate(self.series_indices):  
            if obs_idx not in obs_groups:  
                obs_groups[obs_idx] = []  
            obs_groups[obs_idx].append(idx) 

        # Calculate Sobol indices for each observable  
        S1_all_dict = {}  
        ST_all_dict = {}  
        S2_all_dict = {}
        
        for obs_idx, feature_indices in obs_groups.items():  
            obs_outputs = outputs[:, feature_indices]  
            n_obs_outputs = obs_outputs.shape[1]  
            
            S1_obs = np.zeros((n_obs_outputs, self.num_params))  
            ST_obs = np.zeros((n_obs_outputs, self.num_params))  
            S2_obs = np.zeros((n_obs_outputs, self.num_params, self.num_params))  
            
            for i in range(n_obs_outputs):  
                Si = sobol.analyze(self.problem, obs_outputs[:, i], print_to_console=self.verbose)  
                S1_obs[i, :] = Si['S1']  
                ST_obs[i, :] = Si['ST']  
                S2_obs[i, :] = np.array(Si['S2'])  
            
            S1_all_dict[obs_idx] = S1_obs  
            ST_all_dict[obs_idx] = ST_obs  
            S2_all_dict[obs_idx] = S2_obs 

        return S1_all_dict, ST_all_dict, S2_all_dict

    def plot_sobol_first_order_idx_old(self, S1_all, ST_all):

        if self.rank !=0:
            return
        
        """
        Plot first-order and total-order Sobol indices for multiple outputs.

        Parameters:
            S1_all (np.ndarray): First-order Sobol indices, shape (n_outputs, n_params)
            ST_all (np.ndarray): Total-order Sobol indices, shape (n_outputs, n_params)
        """
        n_outputs = S1_all.shape[0]
        x = np.arange(self.num_params)

        for i in range(n_outputs):
            S1 = S1_all[i]
            ST = ST_all[i]
            output_name = rf"${self.obs_info['names_for_plotting'][i]}$ - experiment{self.obs_info['experiment_idxs'][i]}, subexperiment{self.obs_info['subexperiment_idxs'][i]}"
            # output_name = self.obs_info["names_for_plotting"][i] if hasattr(self, "obs_info") else f"Output_{i}"

            # Set figure width adaptively based on number of parameters (xticks)
            fig_width = max(12, 1.0 * len(self.SA_cfg["param_names"]))
            plt.figure(figsize=(fig_width, 5))
            plt.bar(x - 0.2, S1, width=0.4, label='First-order', color='blue', alpha=0.7)
            plt.bar(x + 0.2, ST, width=0.4, label='Total-order', color='red', alpha=0.7)

            plt.xticks(x, self.SA_cfg["param_names"], rotation=45, fontsize=8)
            plt.ylabel('Sensitivity Index')
            plt.title(rf'Sobol Sensitivity - {output_name}')
            plt.legend()
            plt.tight_layout()

            file_name = f"{output_name}_n{self.num_samples}_First_order_idx.png"
            plt.savefig(os.path.join(self.save_path, file_name))
            plt.clf()
            plt.close()

    def plot_sobol_first_order_idx(self, S1_all_dict, ST_all_dict):  
        """  
        Plot first-order and total-order Sobol indices for multiple outputs.  
        Handles both regular observables and time-series data with overlaid parameters.  
    
        Parameters:  
            S1_all_dict (dict): Dictionary mapping obs_idx to first-order Sobol indices  
            ST_all_dict (dict): Dictionary mapping obs_idx to total-order Sobol indices  
        """  
        if self.rank != 0:  
            return  
        
        for obs_idx, S1_obs in S1_all_dict.items():  
            obs_name = self.obs_info['names_for_plotting'][obs_idx]  
            
            # Check if this is a series with null operation  
            if self.obs_info["data_types"][obs_idx] == "series" and self.obs_info["operations"][obs_idx] is None:  
                # Time-series plotting with overlaid parameters  
                n_time_points = S1_obs.shape[0]  
                obs_dt = self.obs_info["obs_dt"][obs_idx] if "obs_dt" in self.obs_info else 0.01  
                time_array = np.arange(n_time_points) * obs_dt  
                
                # Create single figure with all parameters overlaid  
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))  
                
                # Plot first-order indices  
                for param_idx in range(self.num_params):  
                    ax1.plot(time_array, S1_obs[:, param_idx],   
                            label=f'{self.SA_cfg["param_names"][param_idx]} (S1)',   
                            linewidth=2, alpha=0.8)  
                ax1.set_ylabel('First-Order Sensitivity')  
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
                ax1.grid(True, alpha=0.3)  
                
                max_S1 = np.nanmax(S1_obs)
                if np.isfinite(max_S1) and max_S1 > 0:
                    ax1.set_ylim([-0.5, max_S1 * 1.1])
                else:
                    ax1.set_ylim([-0.5, 1])  # or another default range

                # Plot total-order indices  
                for param_idx in range(self.num_params):  
                    ax2.plot(time_array, ST_all_dict[obs_idx][:, param_idx],   
                            label=f'{self.SA_cfg["param_names"][param_idx]} (ST)',   
                            linewidth=2, alpha=0.8)  
                ax2.set_ylabel('Total-Order Sensitivity')  
                ax2.set_xlabel('Time (s)')  
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
                ax2.grid(True, alpha=0.3)  
                
                max_ST = np.nanmax(ST_all_dict[obs_idx])
                if np.isfinite(max_ST) and max_ST > 0:
                    ax2.set_ylim([-0.5, max_ST * 1.1])
                else:
                    ax2.set_ylim([-0.5, 1])  # or another default range
                
                plt.suptitle(f'Time-Series Sobol Sensitivity - {obs_name}', fontsize=14, fontweight='bold')  
                plt.tight_layout()  
                
                filename = f"{obs_name}_time_series_sensitivity_n{self.num_samples}.png"  
                plt.savefig(os.path.join(self.save_path, filename), dpi=150, bbox_inches='tight')  
                plt.clf()  
                plt.close()  
                
            else:  
                # Regular bar plot for non-series observables  
                S1 = S1_obs[0] if S1_obs.ndim > 1 else S1_obs  
                ST = ST_all_dict[obs_idx][0] if ST_all_dict[obs_idx].ndim > 1 else ST_all_dict[obs_idx]  
                
                x = np.arange(self.num_params)  
                fig_width = max(12, 1.0 * len(self.SA_cfg["param_names"]))  
                plt.figure(figsize=(fig_width, 6))  
                
                plt.bar(x - 0.2, S1, width=0.4, label='First-order', color='blue', alpha=0.7)  
                plt.bar(x + 0.2, ST, width=0.4, label='Total-order', color='red', alpha=0.7)  
                
                plt.xticks(x, self.SA_cfg["param_names"], rotation=45, fontsize=10)  
                plt.ylabel('Sensitivity Index')  
                plt.title(f'Sobol Sensitivity - {obs_name}')  
                plt.legend()  
                plt.grid(True, alpha=0.3)  
                plt.tight_layout()  
                
                filename = f"{obs_name}_n{self.num_samples}_First_order_idx.png"  
                plt.savefig(os.path.join(self.save_path, filename), dpi=150, bbox_inches='tight')  
                plt.clf()  
                plt.close()
    
    def plot_sobol_S2_idx_old(self, S2_all):
        """
        Plot second-order Sobol interaction indices for multiple outputs.

        Parameters:
            S2_all (np.ndarray): Second-order indices, shape (n_outputs, n_params, n_params)
        """

        if self.rank !=0:
            return
        
        n_outputs = S2_all.shape[0]
        for i in range(n_outputs):
            S2 = S2_all[i]
            output_name = rf"${self.obs_info['names_for_plotting'][i]}$ - experiment{self.obs_info['experiment_idxs'][i]}, subexperiment{self.obs_info['subexperiment_idxs'][i]}"

            # plt.figure(figsize=(6, 5))
            fig_width = max(6, 1.0 * len(self.SA_cfg["param_names"]))
            plt.figure(figsize=(fig_width, fig_width))
            sns.heatmap(S2, annot=True, fmt=".2f", xticklabels=self.SA_cfg["param_names"], yticklabels=self.SA_cfg["param_names"], cmap="coolwarm")
            plt.title(rf"2nd order Sobol Indices - {output_name}")
            plt.tight_layout()

            filename = f"{output_name}_n{self.num_samples}_2nd_order_idx.png"
            plt.savefig(os.path.join(self.save_path, filename))
            plt.clf()
            plt.close()
    
    def plot_sobol_S2_idx(self, S2_all_dict):  
        """  
        Plot second-order Sobol interaction indices for multiple outputs.  
        Handles both regular observables and time-series data.  
    
        Parameters:  
            S2_all_dict (dict): Dictionary mapping obs_idx to second-order Sobol indices  
        """  
        if self.rank != 0:  
            return  
        
        for obs_idx, S2_obs in S2_all_dict.items():  
            obs_name = self.obs_info['names_for_plotting'][obs_idx]  
            
            # Check if this is a series with null operation  
            if self.obs_info["data_types"][obs_idx] == "series" and self.obs_info["operations"][obs_idx] is None:  
                # Time-series second-order plotting  
                n_time_points = S2_obs.shape[0]  
                obs_dt = self.obs_info["obs_dt"][obs_idx] if "obs_dt" in self.obs_info else 0.01  
                time_array = np.arange(n_time_points) * obs_dt  
                
                # Create heatmap animation or multiple subplots for key time points  
                # Option 1: Plot at selected time points  
                n_plots = min(4, n_time_points)  # Plot up to 4 time points  
                time_indices = np.linspace(0, n_time_points-1, n_plots, dtype=int)  
                
                fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))  
                if n_plots == 1:  
                    axes = [axes]  
                
                for plot_idx, t_idx in enumerate(time_indices):  
                    S2_at_t = S2_obs[t_idx]  
                    sns.heatmap(S2_at_t, annot=True, fmt=".2f",   
                            xticklabels=self.SA_cfg["param_names"],   
                            yticklabels=self.SA_cfg["param_names"],   
                            cmap="coolwarm", center=0,  
                            ax=axes[plot_idx], vmin=-0.1, vmax=0.5)  
                    axes[plot_idx].set_title(f't = {time_array[t_idx]:.2f}s')  
                
                plt.suptitle(f'Time-Series 2nd Order Sobol Indices - {obs_name}', fontsize=14, fontweight='bold')  
                plt.tight_layout()  
                
                filename = f"{obs_name}_time_series_2nd_order_n{self.num_samples}.png"  
                plt.savefig(os.path.join(self.save_path, filename), dpi=150, bbox_inches='tight')  
                plt.clf()  
                plt.close()  
                
            else:  
                # Regular heatmap for non-series observables  
                S2 = S2_obs[0] if S2_obs.ndim > 2 else S2_obs  
                
                fig_width = max(8, 1.0 * len(self.SA_cfg["param_names"]))  
                plt.figure(figsize=(fig_width, fig_width))  
                
                sns.heatmap(S2, annot=True, fmt=".2f",   
                        xticklabels=self.SA_cfg["param_names"],   
                        yticklabels=self.SA_cfg["param_names"],   
                        cmap="coolwarm", center=0, vmin=-0.1, vmax=0.5)  
                
                plt.title(f"2nd Order Sobol Indices - {obs_name}")  
                plt.tight_layout()  
                
                filename = f"{obs_name}_n{self.num_samples}_2nd_order_idx.png"  
                plt.savefig(os.path.join(self.save_path, filename), dpi=150, bbox_inches='tight')  
                plt.clf()  
                plt.close()

    def plot_sobol_heatmap_old(self, S1_all, ST_all):
        
        if self.rank != 0:
            return
        
        """
        Generates 2D heatmaps for first-order (S1) and total-order (ST) Sobol indices.
        
        The heatmaps show:
        Y-axis: Input Parameters (self.SA_cfg["param_names"])
        X-axis: Model Outputs (concatenated names from self.obs_info)
        Color: Sobol Index Value
        
        Parameters:
            S1_all (np.ndarray): First-order Sobol indices, shape (n_outputs, n_params)
            ST_all (np.ndarray): Total-order Sobol indices, shape (n_outputs, n_params)
        """
        
        print("\nGenerating Sobol Index Heatmaps...")
        
        # 1. Define Axis Labels
        output_labels = self.generate_output_labels()

        param_labels = [rf"${name}$" for name in self.param_id_info["param_names_for_plotting"]]

        # Current shape: (n_outputs, n_params) -> Desired shape: (n_params, n_outputs)
        S1_heatmap_data = S1_all.T
        ST_heatmap_data = ST_all.T
        
        # Define the title prefix using the total sample count (N * (D+2))
        total_samples = S1_all.shape[1] * (S1_all.shape[0] + 2) if hasattr(self, 'num_params') else 'N/A'
        title_prefix = f"Sobol Indices (N={self.num_samples*(self.num_params+2)})"
        
        def create_heatmap(data, index_type):
            
            df_data = pd.DataFrame(data, index=param_labels, columns=output_labels)
            
            fig_width = max(10, len(output_labels) * 0.5) 
            fig_height = max(6, len(param_labels) * 0.5)
            
            plt.figure(figsize=(fig_width, fig_height))
            
            sns.heatmap(
                df_data,
                annot=True,               # Annotate with the index values
                fmt=".2f",                # Format annotations to 2 decimal places
                cmap="viridis",           # Good colormap for continuous data
                linewidths=0.5,           # Lines between cells
                linecolor='lightgray',
                cbar_kws={'label': f'{index_type} Index Value'}
            )

            plt.title(f'{title_prefix} - {index_type}', fontsize=14)
            plt.xlabel('Model Output', fontsize=12)
            plt.ylabel('Input Parameter', fontsize=12)
            
            plt.xticks(rotation=45, ha='right', fontsize=8) 
            plt.yticks(rotation=0, fontsize=8) 
            
            plt.tight_layout()
            
            file_name = f"{index_type.replace('-', '_')}_Sobol_Heatmap.png"
            save_path = os.path.join(self.save_path, file_name)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved {index_type} heatmap to {save_path}")

        create_heatmap(S1_heatmap_data, 'First-Order ($S_1$)')
        create_heatmap(ST_heatmap_data, 'Total-Order ($S_T$)')

    def plot_sobol_heatmap(self, S1_all_dict, ST_all_dict):  
        """  
        Generates 2D heatmaps for first-order (S1) and total-order (ST) Sobol indices  
        for NON-SERIES outputs only, using dictionary inputs.  
        
        The heatmaps show:  
        Y-axis: Input Parameters (self.SA_cfg["param_names"])  
        X-axis: Model Outputs (all non-series features concatenated)  
        Color: Sobol Index Value  
        
        Parameters:  
            S1_all_dict (dict): Dictionary mapping obs_idx to first-order Sobol indices  
            ST_all_dict (dict): Dictionary mapping obs_idx to total-order Sobol indices  
        """  
        
        if self.rank != 0:  
            return  
        
        print("\nGenerating Sobol Index Heatmaps for Non-Series Outputs...")  
        
        # Filter out series data items and collect non-series features  
        non_series_indices = []  
        non_series_output_labels = []  
        non_series_S1_list = []  
        non_series_ST_list = []  
        
        for obs_idx in S1_all_dict.keys():  
            # Skip series data with null operation  
            if not (self.obs_info["data_types"][obs_idx] == "series" and   
                    self.obs_info["operations"][obs_idx] is None):  
                
                # Get the Sobol indices for this observable  
                S1_obs = S1_all_dict[obs_idx]  
                ST_obs = ST_all_dict[obs_idx]  
                
                # For non-series data, we expect single values (not time series)  
                if S1_obs.ndim > 1:  
                    # If it's multi-dimensional, take the first (or mean) value  
                    S1_values = S1_obs[0] if S1_obs.shape[0] == 1 else np.mean(S1_obs, axis=0)  
                    ST_values = ST_obs[0] if ST_obs.shape[0] == 1 else np.mean(ST_obs, axis=0)  
                else:  
                    S1_values = S1_obs  
                    ST_values = ST_obs  
                
                non_series_indices.append(obs_idx)  
                non_series_S1_list.append(S1_values)  
                non_series_ST_list.append(ST_values)  
                
                # Create output label with operation info  
                obs_name = self.obs_info['names_for_plotting'][obs_idx]  
                exp_idx = self.obs_info["experiment_idxs"][obs_idx]  
                subexp_idx = self.obs_info["subexperiment_idxs"][obs_idx]  
                
                operation = self.obs_info["operations"][obs_idx]  
                if operation is None:  
                    feature_label = f"{obs_name}_exp{exp_idx}_sub{subexp_idx}"  
                else:  
                    feature_label = f"{obs_name}({operation})_exp{exp_idx}_sub{subexp_idx}"  
                
                non_series_output_labels.append(feature_label)  
        
        if len(non_series_indices) == 0:  
            print("No non-series outputs found for heatmap generation.")  
            return  
        
        # Concatenate all non-series features  
        S1_non_series = np.vstack(non_series_S1_list)  
        ST_non_series = np.vstack(non_series_ST_list)  
        
        # Define axis labels  
        param_labels = [rf"${name}$" for name in self.param_id_info["param_names_for_plotting"]]  
        
        # Current shape: (n_outputs, n_params) -> Desired shape: (n_params, n_outputs)  
        S1_heatmap_data = S1_non_series.T  
        ST_heatmap_data = ST_non_series.T  
        
        # Define the title prefix  
        title_prefix = f"Sobol Indices (N={self.num_samples*(self.num_params+2)})"  
        
        def create_heatmap(data, index_type, output_labels):  
            df_data = pd.DataFrame(data, index=param_labels, columns=output_labels)  
            
            fig_width = max(10, len(output_labels) * 0.5)   
            fig_height = max(6, len(param_labels) * 0.5)  
            
            plt.figure(figsize=(fig_width, fig_height))  
            
            sns.heatmap(  
                df_data,  
                annot=True,  
                fmt=".2f",  
                cmap="viridis",  
                linewidths=0.5,  
                linecolor='lightgray',  
                cbar_kws={'label': f'{index_type} Index Value'}  
            )  
            
            plt.title(f'{title_prefix} - {index_type} (Non-Series Only)', fontsize=14)  
            plt.xlabel('Model Output Features', fontsize=12)  
            plt.ylabel('Input Parameters', fontsize=12)  
            
            plt.xticks(rotation=45, ha='right', fontsize=8)   
            plt.yticks(rotation=0, fontsize=8)   
            
            plt.tight_layout()  
            
            file_name = f"{index_type.replace('-', '_')}_Non_Series_Sobol_Heatmap.png"  
            save_path = os.path.join(self.save_path, file_name)  
            plt.savefig(save_path, bbox_inches='tight', dpi=300)  
            plt.close()  
            print(f"Saved {index_type} non-series heatmap to {save_path}")  
        
        create_heatmap(S1_heatmap_data, 'First-Order ($S_1$)', non_series_output_labels)  
        create_heatmap(ST_heatmap_data, 'Total-Order ($S_T$)', non_series_output_labels)
        
    def plot_time_series_summary(self, S1_all_dict, ST_all_dict):  
        """  
        Create summary plots showing sensitivity over time for all series observables  
        with all parameters overlaid in single figures.  
        
        Parameters:  
            S1_all_dict (dict): Dictionary mapping obs_idx to first-order Sobol indices  
            ST_all_dict (dict): Dictionary mapping obs_idx to total-order Sobol indices  
        """  
        if self.rank != 0:  
            return  
        
        # Find all series observables  
        series_obs = [idx for idx in S1_all_dict.keys()   
                    if self.obs_info["data_types"][idx] == "series"   
                    and self.obs_info["operations"][idx] is None]  
        
        if not series_obs:  
            return  
        
        # Create summary plots - one for first-order, one for total-order  
        fig1, ax1 = plt.subplots(figsize=(14, 8))  
        fig2, ax2 = plt.subplots(figsize=(14, 8))  
        
        # Plot first-order sensitivities  
        for obs_idx in series_obs:  
            obs_name = self.obs_info['names_for_plotting'][obs_idx]  
            S1_obs = S1_all_dict[obs_idx]  
            
            obs_dt = self.obs_info["obs_dt"][obs_idx] if "obs_dt" in self.obs_info else 0.01  
            time_array = np.arange(S1_obs.shape[0]) * obs_dt  
            
            # Plot average sensitivity across all parameters for this observable  
            avg_sensitivity = np.mean(S1_obs, axis=1)  
            ax1.plot(time_array, avg_sensitivity,   
                    label=f'{obs_name} (avg)', linewidth=2, alpha=0.8)  
        
        ax1.set_xlabel('Time (s)')  
        ax1.set_ylabel('Average First-Order Sensitivity')  
        ax1.set_title('Time-Series First-Order Sensitivity Summary (All Observables)')  
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
        ax1.grid(True, alpha=0.3)  
        plt.tight_layout()  
        
        filename = f"time_series_first_order_summary_n{self.num_samples}.png"  
        plt.savefig(os.path.join(self.save_path, filename), dpi=150, bbox_inches='tight')  
        plt.clf()  
        plt.close()  
        
        # Plot total-order sensitivities  
        for obs_idx in series_obs:  
            obs_name = self.obs_info['names_for_plotting'][obs_idx]  
            ST_obs = ST_all_dict[obs_idx]  
            
            obs_dt = self.obs_info["obs_dt"][obs_idx] if "obs_dt" in self.obs_info else 0.01  
            time_array = np.arange(ST_obs.shape[0]) * obs_dt  
            
            # Plot average sensitivity across all parameters for this observable  
            avg_sensitivity = np.mean(ST_obs, axis=1)  
            ax2.plot(time_array, avg_sensitivity,   
                    label=f'{obs_name} (avg)', linewidth=2, alpha=0.8)  
        
        ax2.set_xlabel('Time (s)')  
        ax2.set_ylabel('Average Total-Order Sensitivity')  
        ax2.set_title('Time-Series Total-Order Sensitivity Summary (All Observables)')  
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
        ax2.grid(True, alpha=0.3)  
        plt.tight_layout()  
        
        filename = f"time_series_total_order_summary_n{self.num_samples}.png"  
        plt.savefig(os.path.join(self.save_path, filename), dpi=150, bbox_inches='tight')  
        plt.clf()  
        plt.close()
            
    def save_sobol_indices_old(self, S1_all, ST_all, S2_all):
        if self.rank != 0:
            return

        """
        Save all Sobol indices to single CSV files (one for S1/ST, one for S2).

        Parameters:
            S1_all (np.ndarray): First-order Sobol indices, shape (n_outputs, n_params)
            ST_all (np.ndarray): Total-order Sobol indices, shape (n_outputs, n_params)
            S2_all (np.ndarray): Second-order Sobol indices, shape (n_outputs, n_params, n_params)
        """
        n_outputs = S1_all.shape[0]
        param_names = self.SA_cfg["param_names"]

        # Prepare output/feature names
        if n_outputs <= len(self.obs_info['names_for_plotting']):
            output_names = [
                f"{self.obs_info['names_for_plotting'][i]} (Exp{self.obs_info['experiment_idxs'][i]}, Sub{self.obs_info['subexperiment_idxs'][i]})"
                for i in range(n_outputs)
            ]
        else:
            output_names = [
                f"{self.obs_info['names_for_plotting'][i]} (Exp{self.obs_info['experiment_idxs'][i]}, Sub{self.obs_info['subexperiment_idxs'][i]})"
                for i in range(n_outputs-1)
            ]
            output_names.append("Cost")

        # --- Save S1/ST indices ---
        df_Sobol = pd.DataFrame({'Parameter': param_names})
        for i, out_name in enumerate(output_names):
            df_Sobol[f"S1_{out_name}"] = S1_all[i]
            df_Sobol[f"ST_{out_name}"] = ST_all[i]
        file_name = f"all_outputs_n{self.num_samples}_Sobol_indices.csv"
        df_Sobol.to_csv(os.path.join(self.save_path, file_name), index=False)

        # --- Save S2 indices ---
        # For each output, flatten S2 into a DataFrame with MultiIndex columns
        s2_dict = {}
        for i, out_name in enumerate(output_names):
            # S2_all[i]: (n_params, n_params)
            s2_flat = pd.DataFrame(
                S2_all[i],
                index=param_names,
                columns=param_names
            )
            # Rename columns to include output name
            s2_flat.columns = [f"{out_name}__{col}" for col in s2_flat.columns]
            s2_dict[out_name] = s2_flat

        # Concatenate all S2 DataFrames horizontally
        df_S2 = pd.concat([s2_dict[out_name] for out_name in output_names], axis=1)
        df_S2.index.name = "Parameter"
        file_name_S2 = f"all_outputs_n{self.num_samples}_Sobol_2nd_order_indices.csv"
        df_S2.to_csv(os.path.join(self.save_path, file_name_S2))

    def save_sobol_indices(self, S1_all_dict, ST_all_dict, S2_all_dict):  
        """  
        Save all Sobol indices to single CSV files (one for S1/ST, one for S2).  
        Filters out series data and concatenates non-series features.  
        
        Parameters:  
            S1_all_dict (dict): Dictionary mapping obs_idx to first-order Sobol indices  
            ST_all_dict (dict): Dictionary mapping obs_idx to total-order Sobol indices  
            S2_all_dict (dict): Dictionary mapping obs_idx to second-order Sobol indices  
        """  
        
        if self.rank != 0:  
            return  
        
        print("\nSaving Sobol Indices for Non-Series Outputs...")  
        
        # Filter out series data items and collect non-series features  
        non_series_indices = []  
        non_series_output_labels = []  
        non_series_S1_list = []  
        non_series_ST_list = []  
        non_series_S2_list = []  
        
        for obs_idx in S1_all_dict.keys():  
            # Skip series data with null operation  
            if not (self.obs_info["data_types"][obs_idx] == "series" and   
                    self.obs_info["operations"][obs_idx] is None):  
                
                # Get the Sobol indices for this observable  
                S1_obs = S1_all_dict[obs_idx]  
                ST_obs = ST_all_dict[obs_idx]  
                S2_obs = S2_all_dict[obs_idx]  
                
                # For non-series data, we expect single values (not time series)  
                if S1_obs.ndim > 1:  
                    # If it's multi-dimensional, take the first (or mean) value  
                    S1_values = S1_obs[0] if S1_obs.shape[0] == 1 else np.mean(S1_obs, axis=0)  
                    ST_values = ST_obs[0] if ST_obs.shape[0] == 1 else np.mean(ST_obs, axis=0)  
                    S2_values = S2_obs[0] if S2_obs.shape[0] == 1 else np.mean(S2_obs, axis=0)  
                else:  
                    S1_values = S1_obs  
                    ST_values = ST_obs  
                    S2_values = S2_obs  
                
                non_series_indices.append(obs_idx)  
                non_series_S1_list.append(S1_values)  
                non_series_ST_list.append(ST_values)  
                non_series_S2_list.append(S2_values)  
                
                # Create output label with operation info  
                obs_name = self.obs_info['names_for_plotting'][obs_idx]  
                exp_idx = self.obs_info["experiment_idxs"][obs_idx]  
                subexp_idx = self.obs_info["subexperiment_idxs"][obs_idx]  
                
                operation = self.obs_info["operations"][obs_idx]  
                if operation is None:  
                    feature_label = f"{obs_name}_exp{exp_idx}_sub{subexp_idx}"  
                else:  
                    feature_label = f"{obs_name}({operation})_exp{exp_idx}_sub{subexp_idx}"  
                
                non_series_output_labels.append(feature_label)  
        
        if len(non_series_indices) == 0:  
            print("No non-series outputs found for saving Sobol indices.")  
            return  
        
        # Concatenate all non-series features  
        S1_non_series = np.vstack(non_series_S1_list)  
        ST_non_series = np.vstack(non_series_ST_list)  
        S2_non_series = np.stack(non_series_S2_list, axis=0)  
        
        n_outputs = len(non_series_output_labels)  
        param_names = self.SA_cfg["param_names"]  
        
        # --- Save S1/ST indices ---  
        df_Sobol = pd.DataFrame({'Parameter': param_names})  
        for i, out_name in enumerate(non_series_output_labels):  
            df_Sobol[f"S1_{out_name}"] = S1_non_series[i]  
            df_Sobol[f"ST_{out_name}"] = ST_non_series[i]  
        
        file_name = f"non_series_outputs_n{self.num_samples}_Sobol_indices.csv"  
        save_path = os.path.join(self.save_path, file_name)  
        df_Sobol.to_csv(save_path, index=False)  
        print(f"Saved S1/ST indices to {save_path}")  
        
        # --- Save S2 indices ---  
        # For each output, flatten S2 into a DataFrame with MultiIndex columns  
        s2_dict = {}  
        for i, out_name in enumerate(non_series_output_labels):  
            # S2_non_series[i]: (n_params, n_params)  
            s2_flat = pd.DataFrame(  
                S2_non_series[i],  
                index=param_names,  
                columns=param_names  
            )  
            # Rename columns to include output name  
            s2_flat.columns = [f"{out_name}__{col}" for col in s2_flat.columns]  
            s2_dict[out_name] = s2_flat  
        
        # Concatenate all S2 DataFrames horizontally  
        df_S2 = pd.concat([s2_dict[out_name] for out_name in non_series_output_labels], axis=1)  
        df_S2.index.name = "Parameter"  
        
        file_name_S2 = f"non_series_outputs_n{self.num_samples}_Sobol_2nd_order_indices.csv"  
        save_path_S2 = os.path.join(self.save_path, file_name_S2)  
        df_S2.to_csv(save_path_S2)  
        print(f"Saved S2 indices to {save_path_S2}")

    def save_sobol_indices_series(self, S1_all_dict, ST_all_dict, S2_all_dict):  
        """  
        Save Sobol indices for series outputs to separate files.  
        Each series observable gets its own set of files with time evolution.  
        
        Parameters:  
            S1_all_dict (dict): Dictionary mapping obs_idx to first-order Sobol indices  
            ST_all_dict (dict): Dictionary mapping obs_idx to total-order Sobol indices    
            S2_all_dict (dict): Dictionary mapping obs_idx to second-order Sobol indices  
        """  
        
        if self.rank != 0:  
            return  
        
        print("\nSaving Sobol Indices for Series Outputs...")  
        
        # Filter series data items (data_type == "series" and operation is None)  
        series_indices = []  
        series_output_labels = []  
        
        for obs_idx in S1_all_dict.keys():  
            if self.obs_info["data_types"][obs_idx] == "series" and \
            self.obs_info["operations"][obs_idx] is None:  
                
                series_indices.append(obs_idx)  
                
                # Create output label  
                obs_name = self.obs_info['names_for_plotting'][obs_idx]  
                exp_idx = self.obs_info["experiment_idxs"][obs_idx]  
                subexp_idx = self.obs_info["subexperiment_idxs"][obs_idx]  
                label = f"{obs_name}_exp{exp_idx}_sub{subexp_idx}"  
                series_output_labels.append(label)  
        
        if len(series_indices) == 0:  
            print("No series outputs found for saving Sobol indices.")  
            return  
        
        # Save each series observable separately  
        for i, obs_idx in enumerate(series_indices):  
            obs_label = series_output_labels[i]  
            
            # Get indices for this observable  
            S1_obs = S1_all_dict[obs_idx]  
            ST_obs = ST_all_dict[obs_idx]  
            S2_obs = S2_all_dict[obs_idx]  
            
            # Create time array  
            obs_dt = self.obs_info["obs_dt"][obs_idx] if "obs_dt" in self.obs_info else 0.01  
            n_time_points = S1_obs.shape[0]  
            time_array = np.arange(n_time_points) * obs_dt  
            
            # --- Save S1/ST indices with time ---  
            df_s1_st = pd.DataFrame({  
                'Time_s': time_array,  
            })  
            
            # Add S1 and ST columns for each parameter  
            param_names = self.SA_cfg["param_names"]  
            for j, param_name in enumerate(param_names):  
                df_s1_st[f'S1_{param_name}'] = S1_obs[:, j]  
                df_s1_st[f'ST_{param_name}'] = ST_obs[:, j]  
            
            file_name_s1_st = f"{obs_label}_series_n{self.num_samples}_Sobol_indices.csv"  
            save_path = os.path.join(self.save_path, file_name_s1_st)  
            df_s1_st.to_csv(save_path, index=False)  
            print(f"Saved S1/ST indices for {obs_label} to {save_path}")  
            
            # --- Save S2 indices (save as separate files for each time point or summary) ---  
            # Option 1: Save S2 at key time points  
            n_save_points = min(5, n_time_points)  # Save up to 5 time points  
            time_indices = np.linspace(0, n_time_points-1, n_save_points, dtype=int)  
            
            for t_idx in time_indices:  
                S2_at_t = S2_obs[t_idx]  
                df_s2 = pd.DataFrame(S2_at_t, index=param_names, columns=param_names)  
                df_s2.index.name = "Parameter"  
                
                time_str = f"t{time_array[t_idx]:.2f}s"  
                file_name_s2 = f"{obs_label}_series_{time_str}_n{self.num_samples}_Sobol_2nd_order.csv"  
                save_path_s2 = os.path.join(self.save_path, file_name_s2)  
                df_s2.to_csv(save_path_s2)  
            
            print(f"Saved S2 indices for {obs_label} at {n_save_points} time points")  
            
            # Option 2: Save summary statistics across time  
            S2_mean = np.mean(S2_obs, axis=0)  
            S2_std = np.std(S2_obs, axis=0)  
            
            df_s2_summary = pd.DataFrame({  
                'Parameter_1': [p1 for p1 in param_names for p2 in param_names],  
                'Parameter_2': [p2 for p1 in param_names for p2 in param_names],  
                'S2_Mean': S2_mean.flatten(),  
                'S2_Std': S2_std.flatten()  
            })  
            
            file_name_s2_summary = f"{obs_label}_series_summary_n{self.num_samples}_Sobol_2nd_order.csv"  
            save_path_s2_summary = os.path.join(self.save_path, file_name_s2_summary)  
            df_s2_summary.to_csv(save_path_s2_summary, index=False)  
            print(f"Saved S2 summary for {obs_label} to {save_path_s2_summary}")

    def generate_output_labels(self, num_outputs=None):
            """
            Generate output labels for plots, handling cases where the number of outputs
            exceeds the number of names_for_plotting in obs_info.

            Args:
                num_outputs (int, optional): Number of outputs to generate labels for.
                                            If None, uses length of names_for_plotting.

            Returns:
                list: List of output labels.
            """
            # Determine how many labels to generate
            if num_outputs is None:
                if hasattr(self, "obs_info") and "names_for_plotting" in self.obs_info:
                    num_outputs = len(self.obs_info["names_for_plotting"])
                else:
                    num_outputs = 0

            labels = []
            for i in range(num_outputs):
                if hasattr(self, "obs_info") and "names_for_plotting" in self.obs_info:
                    # Use available names, else fallback to generic
                    if i < len(self.obs_info["names_for_plotting"]):
                        name = self.obs_info["names_for_plotting"][i]
                        exp_idx = self.obs_info.get("experiment_idxs", [None]*num_outputs)[i] if "experiment_idxs" in self.obs_info else None
                        sub_idx = self.obs_info.get("subexperiment_idxs", [None]*num_outputs)[i] if "subexperiment_idxs" in self.obs_info else None
                        label = f"{name} (Exp{exp_idx}, Sub{sub_idx})"
                    else:
                        label = f"Output_{i}"
                else:
                    label = f"Output_{i}"
                labels.append(label)
            return labels    

    def plot_sobol_param_trends(self, S1_all):
        """
        Plot line trends: x-axis = outputs, each line = parameter (showing S1 index vs. output).
        Downsamples outputs to a maximum of 300 for clarity.
        """
        if self.rank != 0:
            return

        n_outputs, n_params = S1_all.shape
        output_labels = self.generate_output_labels(n_outputs)
        param_names = self.SA_cfg["param_names"]

        # Downsample if too many outputs
        max_outputs = 50
        if n_outputs > max_outputs:
            idxs = np.linspace(0, n_outputs - 1, max_outputs, dtype=int)
            S1_all = S1_all[idxs, :]
            output_labels = [output_labels[i] for i in idxs]
            n_outputs = max_outputs

        plt.figure(figsize=(min(64, max(16, n_outputs * 0.5)), 8))
        for j in range(n_params):
            plt.plot(output_labels, S1_all[:, j], marker='o', label=param_names[j], alpha=0.8)

        plt.xlabel("Model Output")
        plt.ylabel("First-order Sobol Index")
        plt.title("Parameter Sensitivity Trends Across Outputs")
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "Sobol_ParamTrends_vs_Outputs.png"), dpi=300)
        plt.close()

    def run(self):
        samples = self.generate_samples()
        if self.use_mpi:
            outputs = self.generate_outputs_mpi(samples)
            if self.rank == 0:
                S1_all, ST_all, S2_all = self.sobol_index(outputs)
                # print(f">>>>>>>>>>  {S1_all}, {ST_all}, {S2_all}")
                return S1_all, ST_all, S2_all
            else:
                return None, None, None
        else:
            outputs = self.generate_outputs(samples)
            S1_all, ST_all, S2_all = self.sobol_index(outputs)
            return S1_all, ST_all, S2_all

