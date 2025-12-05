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
import opencor as oc
from opencor_helper import SimulationHelper
from SALib.sample import saltelli
import pandas as pd
from SALib.analyze import sobol
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from parsers.PrimitiveParsers import scriptFunctionParser
from mpi4py import MPI
from parsers.PrimitiveParsers import CSVFileParser, JSONFileParser
import csv
from tqdm import tqdm  # make sure tqdm is installed
import pandas as pd
import datetime
import corner
from matplotlib.ticker import MaxNLocator, ScalarFormatter # Import necessary tools

class sobol_SA():

    """
        A class for performing sensitivity analysis
        How to use:
        1. Initialize the class with the model path, output names, solver info, sensitivity analysis configuration, protocol info, time step, and save path.
        2. Call the `run` method with a feature extractor function and any additional arguments needed by that function.
        3. The 'run' method will generate sobol indices
        4. You can plot the results using the `plot_sobol_first_order_idx` and `plot_sobol_S2_idx` methods.
    """
        
    def __init__(self, model_path, model_out_names, solver_info, SA_cfg, dt, save_path, 
                 param_id_path = None, params_for_id_path=None, use_MPI = False, verbose=False, ga_options=None,
                 sim_time=2.0, pre_time=20.0, feature_lookup_ranges=None, param_id=None):

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
        self.param_id = param_id

        self.model_path = model_path
        self.output_dir = None
        self.verbose = verbose
        self.save_path = save_path

        self.solver_info = solver_info
        self.sample_type = SA_cfg["sample_type"]
        self.num_params = None
        self.model_output_names = model_out_names
        self.ga_options = ga_options
        self.protocol_info = None
        self.dt = dt
        
        # set up observables functions
        sfp = scriptFunctionParser()
        self.operation_funcs_dict = sfp.get_operation_funcs_dict()
        
        # self.__set_obs_names_and_df(param_id_path, sim_time=sim_time, pre_time=pre_time)
        json_parser = JSONFileParser()
        parsed_data = json_parser._parse_json_data(
            param_id_obs_path=param_id_path,
            pre_time=pre_time,
            sim_time=sim_time
        )
        self.gt_df = parsed_data["gt_df"]
        self.protocol_info = parsed_data["protocol_info"]
        self.prediction_info = parsed_data["prediction_info"]

        self.obs_info = json_parser._process_obs_info(gt_df=self.gt_df)
        self.protocol_info = json_parser._process_protocol_and_weights(
            gt_df=self.gt_df,
            protocol_info=self.protocol_info,
            dt=self.dt
        )

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
            self.csv_parser = CSVFileParser()
            self.param_id_info = self.csv_parser.get_param_id_info(self.params_for_id_path)
            self.csv_parser.save_param_names(self.param_id_info, self.output_dir, self.rank)
            # self.__set_and_save_param_names()

        self.SA_cfg = self.create_SA_cfg(self.sample_type, SA_cfg["num_samples"])

        self.set_feature_lookup_ranges()

    def set_feature_lookup_ranges(self):
        feature_lookup_ranges = {}
        for i, lookup_range in enumerate(self.obs_info.get("faeture_range", [{}]*len(self.obs_info["obs_names"]))):
            if lookup_range and isinstance(lookup_range, dict) and "min" in lookup_range and "max" in lookup_range:
                feature_lookup_ranges[str(i)] = (lookup_range["min"], lookup_range["max"])
            else:
                feature_lookup_ranges[str(i)] = (-np.inf, np.inf)
        
        feature_lookup_ranges[str(i+1)] = (-np.inf, np.inf)
        self.feature_lookup_ranges = feature_lookup_ranges

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

    def initialise_sim_helper(self):
        return SimulationHelper(self.model_path, self.dt, self.sim_time,
                                solver_info=self.solver_info, pre_time=self.pre_time)

<<<<<<< HEAD
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
                print("data_items not found in json object. ",
                      "Please check that data_items is the key for the list of data items")
            if 'protocol_info' in json_obj.keys():
                self.protocol_info = json_obj['protocol_info']
                if "sim_times" not in self.protocol_info.keys():
                    self.protocol_info["sim_times"] = [[sim_time]]
                if "pre_times" not in self.protocol_info.keys():
                    self.protocol_info["pre_times"] = [pre_time]
            else:
                if pre_time is None or sim_time is None:
                    print("protocol_info not found in json object. ",
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
                        print('"variable" not found in prediction item in obs_data.json file, ',
                              'exitiing') 
                        exit()
                    if 'unit' in entry.keys():
                        self.prediction_info['units'].append(entry['unit'])
                    else:
                        print('"unit" not found in prediction item in obs_data.json file, ',
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
            print(f"unknown data type for imported json object of {type(json_obj)}")
        
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

        # get plotting type
        # TODO make the plot_types operation_funcs so the user can defined how they are plotted.
        warning_printed = False
        for II in range(self.gt_df.shape[0]):
            if "plot_type" not in self.gt_df.iloc[II].keys():
                if self.gt_df.iloc[II]["data_type"] == "constant":
                    if not warning_printed:
                        print('constant data types plot type defaults to horizontal lines',
                            'change "plot_type" in obs_data.json to change this')
                        warning_printed = True
                    self.obs_info["plot_type"].append("horizontal")
                elif self.gt_df.iloc[II]["data_type"] == "prob_dist":
                    if not warning_printed:
                        print('prob_dist data types plot type defaults to horizontal lines',
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
                    print(f'data type {self.gt_df.iloc[II]["data_type"]} not recognised')
            else:
                self.obs_info["plot_type"].append(self.gt_df.iloc[II]["plot_type"])
                if self.obs_info["plot_type"][II] in ["None", "null", "Null", "none", "NONE"]:
                    self.obs_info["plot_type"][II] = None

        self.obs_info["operations"] = []
        self.obs_info["names_for_plotting"] = []
        self.obs_info["operands"] = []
        self.obs_info["freqs"] = []
        self.obs_info["operation_kwargs"] = []
        self.obs_info["faeture_range"] = []
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

            if "lookup_range" in self.gt_df.iloc[II].keys():
                self.obs_info["faeture_range"].append(self.gt_df.iloc[II]["lookup_range"])
            else:
                self.obs_info["faeture_range"].append({})

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
                if self.ga_options is not None:
                    if "cost_type" in self.ga_options.keys():
                        self.obs_info["cost_type"].append(self.ga_options["cost_type"]) # default to cost type in ga_options
                    else:
                        self.obs_info["cost_type"].append("MSE") # default to mean squared error
                elif self.mcmc_options is not None:
                    if "cost_type" in self.mcmc_options.keys():
                        self.obs_info["cost_type"].append(self.mcmc_options["cost_type"]) # default to cost type in mcmc_options
                    else:
                        self.obs_info["cost_type"].append("MSE") # default to mean squared error
                else:
                    print("cost_type not found in obs_data.json, ga_options, or mcmc_options, exiting")
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
                print('experiment_colors in obs_data.json not the same length as num_experiments, exiting')
                exit()

        if "experiment_labels" in self.protocol_info.keys():
            if len(self.protocol_info["experiment_labels"]) != self.protocol_info['num_experiments']:
                print('experiment_labels in obs_data.json not the same length as num_experiments, exiting')
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
                    print(f'experiment_idx not found in obs_data.json entry {self.gt_df.iloc[II]["variable"]}, '
                          'but multiple experiments are defined.',
                          'Setting experiment_idx to 0 for all data points')
            if "subexperiment_idx" not in self.gt_df.iloc[II].keys():
                self.gt_df["subexperiment_idx"] = 0
                if self.protocol_info['num_sub_total'] > 1:
                    print(f'subexperiment_idx not found in obs_data.json entry {self.gt_df.iloc[II]["variable"]}, '
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
    
=======
>>>>>>> dev_parser
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

        failed_params = []     # 1. Simulation failures
        warning_params = []    # 2. Feature extraction warnings
        lookup_range_params = []   # 3. Outputs within provided range

        # Split samples across ranks
        n_samples = len(samples)
        samples_per_rank = n_samples // self.num_procs
        remainder = n_samples % self.num_procs

        if self.rank < remainder:
            start = self.rank * (samples_per_rank + 1)
            end = start + samples_per_rank + 1
        else:
            start = self.rank * samples_per_rank + remainder
            end = start + samples_per_rank

        local_samples = samples[start:end]

        print(f"[MPI Rank {self.rank}] Starting samples {start}:{end} (total {len(local_samples)})")

        local_outputs = []

        # Create a single progress bar for this rank
        with tqdm(total=len(local_samples), desc=f"Rank {self.rank}", position=self.rank, leave=True) as pbar:
            for param_vals in local_samples:

                sim_failed = False      # internal tracking
                warn_flag = False
                lookup_range_flag = False    # if provided

                # --- handle single vs multi subexperiment ---
                if self.protocol_info["num_sub_total"] == 1:
                    # simple case (one experiment only)
                    self.sim_helper.set_param_vals(self.param_id_info["param_names"], param_vals)
                    self.sim_helper.reset_states()
                    success = self.sim_helper.run()

                    operands_outputs_dict = {}

                    retry_count = 0
                    max_retries = 0
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

                        self.sim_helper.reset_and_clear()
                    else:
                        print(f"[MPI Rank {self.rank}] Simulation failed for params: {param_vals}, subexp={subexp_count} after {retry_count} retries")
                        # Set a flag in operands_outputs_dict to indicate failure
                        operands_outputs_dict[(0, 0)] = {"failed": True}

                        sim_failed = True
                        failed_params.append(param_vals)

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
                            max_retries = 0
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
                                print(f"[MPI Rank {self.rank}] Simulation failed for params: {param_vals}, subexp={subexp_count} after {retry_count} retries")
                                # Set a flag in operands_outputs_dict to indicate failure
                                operands_outputs_dict[(exp_idx, this_sub_idx)] = {"failed": True}
                                sim_failed = True
                                failed_params.append(param_vals)
                                
                                # reset at the end of each experiment
                                if this_sub_idx == self.protocol_info["num_sub_per_exp"][exp_idx] - 1:
                                    self.sim_helper.reset_and_clear()

                features = []
                cost = 0.0
                for j in range(len(self.obs_info["operations"])):
                    func = self.operation_funcs_dict[self.obs_info["operations"][j]]
                    exp_idx = self.obs_info["experiment_idxs"][j]
                    subexp_idx = self.obs_info["subexperiment_idxs"][j]
                    operands_outputs = operands_outputs_dict.get((exp_idx, subexp_idx), None)
                    if operands_outputs is not None and not (isinstance(operands_outputs, dict) and operands_outputs == {"failed": True}):
                        feature = func(*operands_outputs[j], **self.obs_info["operation_kwargs"][j])

                        if self.param_id is not None:                        
                            sub_cost = self.param_id.param_id.get_cost_from_operands(operands_outputs, 
                                                                exp_idx=exp_idx, sub_idx=subexp_idx)
                            cost += sub_cost

                        # If function returns (value, warning)
                        if isinstance(feature, tuple):
                            val, flag = feature
                            features.append(val)
                            if flag:
                                warn_flag = True
                        else:
                            features.append(feature)

                    else:
                        # WARNING: using mean biases variance estimates (shrinks variance), underestimates sensitivity
                        # TODO: come up with a better way to impute missing features
                        # Append the mean of the current features (ignoring None) -> reduces variance and bias induces toward zero
                        features.append(np.mean(features))

                # adding cost as extra feature
                if self.param_id is not None:
                    features.append(cost)
                
                if warn_flag:
                    warning_params.append((param_vals, features))

                if hasattr(self, "feature_lookup_ranges"):
                    print(">>>>>>>>")
                    for i, f in enumerate(features):
                        f_min, f_max = self.feature_lookup_ranges[f"{i}"]
                        print(f"Feature {i}: {f}, Range: ({f_min}, {f_max})")
                        if not (f_min <= f <= f_max):
                            print("break")
                            lookup_range_flag = True
                            break
                    if lookup_range_flag:
                        lookup_range_params.append((param_vals, features))

                local_outputs.append(features)
                pbar.update(1)

        print(f"[MPI Rank {self.rank}] Finished processing samples {start}:{end}")

        # Gather results at rank 0
        all_outputs = self.comm.gather(local_outputs, root=0)

        if self.rank == 0:
            outputs = [item for sublist in all_outputs for item in sublist]
            outputs = np.array(outputs)
            print(f"[MPI Rank 0] Gathered and flattened all outputs. Total outputs: {outputs.shape}")

            # Convert input samples to np.array
            samples_arr = np.array(samples)

            # Call the new function
            self.save_output_results(
                samples=samples_arr,
                outputs=outputs,
                failed_params=failed_params,
                warning_params=warning_params,
                lookup_range_params=lookup_range_params
            )

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
        S1_all = np.zeros((n_outputs, self.num_params))
        ST_all = np.zeros((n_outputs, self.num_params))
        S2_all = np.zeros((n_outputs, self.num_params, self.num_params))

        for i in range(n_outputs):
            Si = sobol.analyze(self.problem, outputs[:,i], print_to_console=self.verbose)
            S1_all[i, :] = Si['S1']
            ST_all[i, :] = Si['ST']
            S2_all[i, :] = np.array(Si['S2'])

        return S1_all, ST_all, S2_all

    def save_sobol_indices(self, S1_all, ST_all, S2_all):
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

    def plot_sobol_first_order_idx(self, S1_all, ST_all):

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
            
            if i >= len(self.obs_info['names_for_plotting']):
                output_name = rf"Cost"
            else:
                output_name = rf"${self.obs_info['names_for_plotting'][i]}$ - experiment{self.obs_info["experiment_idxs"][i]}, subexperiment{self.obs_info["subexperiment_idxs"][i]}"
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
            plt.savefig(os.path.join(self.save_path, file_name), dpi=300)
            plt.clf()
            plt.close()

    def plot_sobol_S2_idx(self, S2_all):
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
            
            # output_name = rf"${self.obs_info['names_for_plotting'][i]}$ - experiment{self.obs_info["experiment_idxs"][i]}, subexperiment{self.obs_info["subexperiment_idxs"][i]}"
            if i >= len(self.obs_info['names_for_plotting']):
                output_name = rf"Cost"
            else:
                output_name = rf"${self.obs_info['names_for_plotting'][i]}$ - experiment{self.obs_info["experiment_idxs"][i]}, subexperiment{self.obs_info["subexperiment_idxs"][i]}"

            # plt.figure(figsize=(6, 5))
            fig_width = max(6, 1.0 * len(self.SA_cfg["param_names"]))
            plt.figure(figsize=(fig_width, fig_width))
            sns.heatmap(S2, annot=True, fmt=".2f", xticklabels=self.SA_cfg["param_names"], yticklabels=self.SA_cfg["param_names"], cmap="coolwarm")
            plt.title(rf"2nd order Sobol Indices - {output_name}")
            plt.tight_layout()

            filename = f"{output_name}_n{self.num_samples}_2nd_order_idx.png"
            plt.savefig(os.path.join(self.save_path, filename), dpi=300)
            plt.clf()
            plt.close()

    def plot_sobol_heatmap(self, S1_all, ST_all):
        
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
        if S1_all.shape[0] < len(self.obs_info['names_for_plotting']):
            output_labels = [
                rf"{self.obs_info['names_for_plotting'][i]} (Exp{self.obs_info['experiment_idxs'][i]}, Sub{self.obs_info['subexperiment_idxs'][i]})"
                for i in range(S1_all.shape[0])
            ]
        else:
            output_labels = [
                rf"{self.obs_info['names_for_plotting'][i]} (Exp{self.obs_info['experiment_idxs'][i]}, Sub{self.obs_info['subexperiment_idxs'][i]})"
                for i in range(S1_all.shape[0]-1)
            ]
            output_labels.append(rf"Cost")

        param_labels = self.SA_cfg["param_names"]

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

    def plot_sobol_bubble_plot(self, S1_all, ST_all, index_type='First-Order'):
    
        if self.rank != 0:
            return
        
        """
        Generates a bubble plot (dot plot) for either first-order (S1) or total-order (ST) Sobol indices,
        similar to the provided image example.
        
        Y-axis: Model Outputs
        X-axis: Input Parameters
        Bubble Size: Magnitude of the Sobol Index
        Bubble Color: Unique color for each input parameter
        
        Parameters:
            S1_all (np.ndarray): First-order Sobol indices, shape (n_outputs, n_params)
            ST_all (np.ndarray): Total-order Sobol indices, shape (n_outputs, n_params)
            index_type (str): 'First-Order' for S1 or 'Total-Order' for ST. Determines which matrix to plot.
        """
        
        print(f"\nGenerating Sobol {index_type} Bubble Plot...")
        
        sobol_indices = S1_all if index_type == 'First-Order' else ST_all

        # Output labels need to be created by combining the name, experiment, and subexperiment
        # output_labels = [
        #     self.obs_info['names_for_plotting'][i] 
        #     for i in range(sobol_indices.shape[0])
        # ]
        if sobol_indices.shape[0] < len(self.obs_info['names_for_plotting']):
            output_labels = [
                rf"{self.obs_info['names_for_plotting'][i]} (Exp{self.obs_info['experiment_idxs'][i]}, Sub{self.obs_info['subexperiment_idxs'][i]})"
                for i in range(S1_all.shape[0])
            ]
        else:
            output_labels = [
                rf"{self.obs_info['names_for_plotting'][i]} (Exp{self.obs_info['experiment_idxs'][i]}, Sub{self.obs_info['subexperiment_idxs'][i]})"
                for i in range(S1_all.shape[0]-1)
            ]
            output_labels.append(rf"Cost")
        
        param_labels = self.SA_cfg["param_names"]
        
        n_outputs = sobol_indices.shape[0]
        n_params = sobol_indices.shape[1]

        sns.set_theme(style="white", context="notebook") # Use a clean white background
        
        fig_width = max(10, n_params * 0.7) # Adjust based on number of parameters
        fig_height = max(6, n_outputs * 0.7) # Adjust based on number of outputs        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        param_colors = sns.color_palette("hsv", n_params) 

        # Max bubble size scaling - adjust as needed
        # A base size, and then scale by a factor related to the max index
        max_sobol_val = np.nanmax(sobol_indices) 
        if np.isclose(max_sobol_val, 0): # Avoid division by zero if all values are zero
            max_sobol_val = 1
        
        # Scale bubble sizes: small values get small bubbles, large values get large ones
        # A common way is to scale by value / max_value * some_max_display_size
        BASE_BUBBLE_SIZE = 10 
        MAX_DISPLAY_BUBBLE_SIZE = 800 # Max size in points, adjust as necessary

        for i in range(n_outputs):  
            for j in range(n_params): 
                sobol_val = sobol_indices[i, j]
                
                # Only plot if the sensitivity is non-zero/non-negligible
                if not np.isnan(sobol_val) and sobol_val > 0.005:
                    
                    # Use a specific color for each parameter, cycling if n_params > len(palette)
                    bubble_color = param_colors[j % len(param_colors)]
                    
                    bubble_size = BASE_BUBBLE_SIZE + (sobol_val / max_sobol_val) * (MAX_DISPLAY_BUBBLE_SIZE - BASE_BUBBLE_SIZE)
                    
                    ax.scatter(j, i, 
                            s=bubble_size,      # Size of the marker
                            color=bubble_color,
                            edgecolor='black',  # Black outline for bubbles
                            linewidth=0.5,
                            alpha=0.8,
                            zorder=3) # Ensure bubbles are on top of grid

        ax.set_yticks(np.arange(n_outputs))
        ax.set_yticklabels(output_labels, fontsize=12)
        ax.set_ylabel('Model Output', fontsize=14, labelpad=20)
        ax.set_xticks(np.arange(n_params))
        
        ax.set_xticklabels(param_labels, rotation=45, ha='left', fontsize=12) 
        for tick_label, color in zip(ax.get_xticklabels(), param_colors):
            tick_label.set_color(color)
            tick_label.set_weight('bold') # Make parameter names bold

        ax.set_xlabel('Input Parameter', fontsize=14, labelpad=20)
        
        ax.grid(False) 
        ax.set_facecolor('white') 

        # Add light grey horizontal bands for outputs for better readability (like in image)
        for i in range(0, n_outputs, 2):
            ax.axhspan(i - 0.5, i + 0.5, facecolor='lightgray', alpha=0.3, zorder=0)

        ax.set_xlim(-0.5, n_params - 0.5)
        ax.set_ylim(-0.5, n_outputs - 0.5)

        total_samples = self.num_samples * (2*self.num_params + 2) if hasattr(self, 'num_params') else 'N/A'
        plt.title(f'Sobol {index_type} Sensitivity (N={total_samples})', fontsize=16, pad=20)        
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for title

        # Save Figure
        file_name = f"{index_type.replace('-', '_').replace(' ', '_')}_Sobol_Bubble_Plot.png"
        save_path = os.path.join(self.save_path, file_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {index_type} bubble plot to {save_path}")

    def save_output_results(self, samples, outputs, failed_params=None, warning_params=None, lookup_range_params=None):
        """
        Saves parameters and outputs, including tracking issues, into CSV files.
        """

        # 1) Save all samples + outputs
        try:
            param_labels = self.SA_cfg["param_names"]  # from sensitivity setup
        except:
            param_labels = [f"param_{i}" for i in range(samples.shape[1])]

        try:
            # output_labels = self.obs_info['names_for_plotting']  # user-defined names
            if outputs.shape[1] > len(self.obs_info['names_for_plotting']):
                output_labels = [
                    rf"{self.obs_info['names_for_plotting'][i]} (Exp{self.obs_info['experiment_idxs'][i]}, Sub{self.obs_info['subexperiment_idxs'][i]})"
                    for i in range(outputs.shape[1])
                ]
            else:
                output_labels = [
                    rf"{self.obs_info['names_for_plotting'][i]} (Exp{self.obs_info['experiment_idxs'][i]}, Sub{self.obs_info['subexperiment_idxs'][i]})"
                    for i in range(outputs.shape[1]-1)
                ]
                output_labels.append(rf"Cost")
        except:
            output_labels = [f"feature_{i}" for i in range(outputs.shape[1])]
        
        df = pd.DataFrame(
            np.hstack((samples, outputs)),
            columns=param_labels + output_labels
        )
        file_name = "all_outputs.csv"
        save_path = os.path.join(self.save_path, file_name)
        df.to_csv(save_path, index=False)

        # 2) Save failed simulation parameters
        
        if failed_params:
            print(failed_params)
            file_name = "failed_simulation.csv"
            save_path = os.path.join(self.save_path, file_name)
            pd.DataFrame(failed_params, columns=param_labels).to_csv(save_path, index=False)

        # 3) Save feature extraction warnings
        if warning_params:
            file_name = "warnings.csv"
            save_path = os.path.join(self.save_path, file_name)
            pd.DataFrame([
                {**{p: val for p, val in zip(param_labels, params)},
                **{f: val for f, val in zip(output_labels, outputs_)},
                "warning_flags": flags}
                for params, outputs_, flags in warning_params
            ]).to_csv(save_path, index=False)

        # 4) Save outputs that meet desired range criteria
        if lookup_range_params:
            file_name = "lookup_range.csv"
            save_path = os.path.join(self.save_path, file_name)
            pd.DataFrame([
                {**{p: val for p, val in zip(param_labels, params)},
                **{f: val for f, val in zip(output_labels, outputs_)}}
                for params, outputs_ in lookup_range_params
            ]).to_csv(save_path, index=False)

    def load_category_data(self):
        """
        Loads parameter CSVs for failed, warning, and lookup_range categories if they exist.

        Returns:
            df_all (pd.DataFrame): Combined dataframe with parameters and Category label.
        """
        category_files = {
            "all": "all_outputs.csv",
            "failed": "failed.csv",
            "warning": "warning.csv",
            "lookup_range": "lookup_range.csv"
        }

        df_list = []
        param_labels = self.SA_cfg["param_names"]

        for cat, filename in category_files.items():
            file_path = os.path.join(self.save_path, filename)
            if os.path.isfile(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df["Category"] = cat
                    df_list.append(df[param_labels + ["Category"]])
                    print(f"Loaded {filename}")
                except Exception as e:
                    print(f"⚠️ Error loading {filename}: {e}")
            else:
                print(f"ℹ️ {filename} not found, skipping.")

        if not df_list:
            raise ValueError("❌ No category CSV files were found.")

        df_all = pd.concat(df_list, ignore_index=True)
        return df_all
    
    def plot_corner_overlay_old(self, df: pd.DataFrame, param_names: list[str]):
        """
        Overlay all categories in a corner plot.
        """
        unique_categories = df["Category"].unique()
        fig = None

        for i, cat in enumerate(unique_categories):
            data_cat = df[df["Category"] == cat][param_names].values
            
            print(data_cat)

            fig = corner.corner(
                data_cat,
                fig=fig,
                labels=param_names,
                color=f"C{i}",
                plot_datapoints=True,
                plot_density=True,
                hist_kwargs={"alpha": 0.4},
                plot_contours=False,
                use_math_text=True
            )

        # Legend
        fig.axes[0].plot([], [], label="Categories:")
        for i, cat in enumerate(unique_categories):
            fig.axes[0].plot([], [], color=f"C{i}", label=str(cat))
        fig.axes[0].legend(loc="upper right", fontsize=10)

        file_name = f"corner_plots.png"
        save_path = os.path.join(self.save_path, file_name)
        if save_path is not None:
            fig.savefig(save_path, dpi=300)
            print(f"📌 Corner plot saved to: {save_path}")

    def plot_corner_overlay_old2(self, df: pd.DataFrame, param_names: list[str]):
        """
        Creates a corner-style plot using pure Matplotlib, suitable for overlaying
        categories, even those with very few samples (bypassing corner library limitations).
        """
        N_dims = len(param_names)
        unique_categories = df["Category"].unique()
        print(f"Unique categories found: {unique_categories}")
        
        # 1. Initialize the Plotting Grid
        # Create an N_dims x N_dims figure with shared axes for aligning the scatter plots
        fig, axes = plt.subplots(N_dims, N_dims, figsize=(12, 12))
        
        # Adjust spacing for a tighter fit like a standard corner plot
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        # 2. Iterate through Categories and Plot Data
        for i, cat in enumerate(unique_categories):
            data_cat = df[df["Category"] == cat][param_names].values
            N_samples = data_cat.shape[0]
            print(f"Category '{cat}' has {N_samples} samples.")
            color = f"C{i}"
            
            if N_samples == 0:
                print(f"Skipping category '{cat}': No samples found.")
                continue
                
            print(f"Plotting category '{cat}' with {N_samples} samples.")

            # Iterate over all possible pairs of dimensions (i.e., subplots)
            for row in range(N_dims):
                for col in range(N_dims):
                    ax = axes[row, col]
                    
                    # --- A. Diagonal Plots (i == j): Use for Parameter Labels ---
                    if row == col:
                        # Clear the plotting area and just place the label
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if row == 0:
                            ax.text(0.5, 0.5, param_names[row], transform=ax.transAxes, 
                                    fontsize=14, ha='center', va='center')
                        
                    # --- B. Upper Triangle (i < j): Skip (Corner plots are symmetric) ---
                    elif col > row:
                        ax.set_visible(False)
                        
                    # --- C. Lower Triangle Plots (i > j): Scatter Plots ---
                    elif col < row:
                        # x-axis corresponds to the column index (col)
                        # y-axis corresponds to the row index (row)
                        
                        # Scatter plot the data for this category
                        ax.scatter(data_cat[:, col], data_cat[:, row], 
                                c=color, 
                                s=20,          # Marker size
                                alpha=0.7,     # Transparency
                                label=str(cat) if (row == N_dims-1 and col == 0) else None)
                        
                        # Clean up axis limits and labels
                        ax.tick_params(axis='both', which='major', labelsize=8)
                        
                        # Remove y-tick labels for inner columns
                        if col != 0:
                            ax.set_yticklabels([])
                        # Remove x-tick labels for inner rows
                        if row != N_dims - 1:
                            ax.set_xticklabels([])
                            
                        # Add X-axis label only to the bottom row
                        if row == N_dims - 1:
                            ax.set_xlabel(param_names[col], fontsize=10)
                            
                        # Add Y-axis label only to the first column
                        if col == 0:
                            ax.set_ylabel(param_names[row], fontsize=10)

        # 3. Add Global Legend and Clean Up Axes
        # Use the bottom-left axis (last row, first column) for the legend
        # Find all unique labels from the scatter plots to generate the legend
        handles, labels = axes[N_dims-1, 0].get_legend_handles_labels()
        
        if handles:
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95), 
                    title="Categories", fontsize=10)

        # 4. Save Figure
        file_name = f"corner_style_plots_matplotlib.png"
        save_path = os.path.join(self.save_path, file_name)
        
        # Ensure save_path exists (you might need to create the directory if it doesn't exist)
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 

        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        print(f"📌 Corner-style plot (Matplotlib) saved to: {save_path}")

    def plot_corner_overlay_old3(self, df: pd.DataFrame, param_names: list[str]):
        """
        Creates a corner-style plot using pure Matplotlib with improved axis
        and label alignment for visual appeal.
        """
        N_dims = len(param_names)
        unique_categories = df["Category"].unique()
        
        # 1. Initialize the Plotting Grid
        # Use plt.figure and manually set subplots to prevent the diagonal from
        # automatically taking up the full width/height of the row/column.
        fig, axes = plt.subplots(N_dims, N_dims, figsize=(12, 12))
        
        # Adjust spacing for a tighter fit like a standard corner plot
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        # 2. Iterate through Categories and Plot Data
        for i, cat in enumerate(unique_categories):
            data_cat = df[df["Category"] == cat][param_names].values
            N_samples = data_cat.shape[0]
            color = f"C{i}"
            
            if N_samples == 0:
                continue
                
            # Iterate over all possible pairs of dimensions (i.e., subplots)
            for row in range(N_dims):
                for col in range(N_dims):
                    ax = axes[row, col]
                    
                    # --- A. Upper Triangle (col > row): Remove entirely ---
                    if col > row:
                        ax.set_visible(False)
                        continue # Skip to the next subplot
                    
                    # --- B. Diagonal Plots (row == col): Use for Parameter Labels/Names ---
                    elif row == col:
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # Remove the box frame around the diagonal plot
                        ax.axis('off') 
                        
                        # Place the parameter name in the center
                        ax.text(0.5, 0.5, param_names[row], transform=ax.transAxes, 
                                fontsize=14, ha='center', va='center')
                    
                    # --- C. Lower Triangle Plots (col < row): Scatter Plots ---
                    elif col < row:
                        # Scatter plot the data for this category
                        ax.scatter(data_cat[:, col], data_cat[:, row], 
                                c=color, 
                                s=20,          
                                alpha=0.7,     
                                # Only add a label for the legend in the bottom-left plot
                                label=str(cat) if (row == N_dims-1 and col == 0) else None)
                        
                        # --- AXIS CLEANUP AND ALIGNMENT ---
                        
                        # 1. Ticks and Labels for X-axis (Columns)
                        if row == N_dims - 1:
                            # Only show x-ticks/labels on the bottom row
                            ax.set_xlabel(param_names[col], fontsize=10)
                            # Rotate ticks for better visual separation
                            ax.tick_params(axis='x', which='major', rotation=45, labelsize=8)
                        else:
                            # Hide x-ticks/labels on all inner rows
                            ax.set_xticklabels([])
                            ax.tick_params(axis='x', which='major', length=0) # Remove ticks themselves
                            
                        # 2. Ticks and Labels for Y-axis (Rows)
                        if col == 0:
                            # Only show y-ticks/labels on the first column
                            ax.set_ylabel(param_names[row], fontsize=10)
                            ax.tick_params(axis='y', which='major', labelsize=8)
                        else:
                            # Hide y-ticks/labels on all inner columns
                            ax.set_yticklabels([])
                            ax.tick_params(axis='y', which='major', length=0) # Remove ticks themselves

        # 3. Add Global Legend
        # The legend handles and labels come from the bottom-left axis (0, N_dims-1)
        handles, labels = axes[N_dims-1, 0].get_legend_handles_labels()
        
        if handles:
            # Place the legend outside the main plotting area
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                    title="Categories", fontsize=10)

        # 4. Save Figure
        file_name = f"corner_style_plots_aligned.png"
        save_path = os.path.join(self.save_path, file_name)
        
        # Ensure save_path exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 

        # Use bbox_inches='tight' to ensure labels and legend are not cut off
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_corner_overlay(self, df: pd.DataFrame, param_names: list[str]):
        """
        Creates a corner-style plot using pure Matplotlib with optimized
        axis formatting and alignment for high visual appeal and minimal overlap.
        """
        N_dims = len(param_names)
        unique_categories = df["Category"].unique()
        
        # 1. Initialize the Plotting Grid
        fig, axes = plt.subplots(N_dims, N_dims, figsize=(12, 12))
        
        fig.subplots_adjust(hspace=0.05, wspace=0.05)

        # 2. Iterate through Categories and Plot Data
        for i, cat in enumerate(unique_categories):
            data_cat = df[df["Category"] == cat][param_names].values
            N_samples = data_cat.shape[0]
            color = f"C{i}"
            
            if N_samples == 0:
                continue
                
            # Iterate over all possible pairs of dimensions (i.e., subplots)
            for row in range(N_dims):
                for col in range(N_dims):
                    ax = axes[row, col]
                    
                    # --- A. Upper Triangle (col > row): Remove entirely ---
                    if col > row:
                        ax.set_visible(False)
                        continue
                    
                    # --- B. Diagonal Plots (row == col): Parameter Names ---
                    elif row == col:
                        ax.axis('off') 
                        # Place the parameter name in the center
                        ax.text(0.5, 0.5, param_names[row], transform=ax.transAxes, 
                                fontsize=14, ha='center', va='center')
                    
                    # --- C. Lower Triangle Plots (col < row): Scatter Plots ---
                    elif col < row:
                        
                        # 1. Plot the Data
                        ax.scatter(data_cat[:, col], data_cat[:, row], 
                                c=color, 
                                s=20,          
                                alpha=0.7,     
                                label=str(cat) if (row == N_dims-1 and col == 0) else None)
                        
                        # --- 2. AXIS ALIGNMENT AND FORMATTING ---
                        
                        # Set max number of ticks and use scientific notation (which takes less space)
                        # Use a MaxNLocator to ensure a maximum of 5 ticks to prevent crowding
                        ax.xaxis.set_major_locator(MaxNLocator(5))
                        ax.yaxis.set_major_locator(MaxNLocator(5))
                        
                        # Use ScalarFormatter for clean scientific notation where appropriate
                        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
                        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=True))
                        
                        # 3. Label Hiding and Placement (to prevent overlap)
                        
                        # X-Axis Labels: Only on the bottom row
                        if row == N_dims - 1:
                            ax.set_xlabel(param_names[col], fontsize=10)
                            ax.tick_params(axis='x', which='major', rotation=45, labelsize=8)
                        else:
                            ax.set_xticklabels([])
                            
                        # Y-Axis Labels: Only on the first column
                        if col == 0:
                            ax.set_ylabel(param_names[row], fontsize=10)
                            ax.tick_params(axis='y', which='major', labelsize=8)
                        else:
                            ax.set_yticklabels([])
                            
                        # Y-axis tick parameter to keep the labels external to the plotting box
                        # This often requires Matplotlib to be smart enough about space, 
                        # which is why ScalarFormatter helps by reducing the label width.
                        ax.tick_params(axis='y', which='major', pad=5)


        # 3. Add Global Legend
        handles, labels = axes[N_dims-1, 0].get_legend_handles_labels()
        
        if handles:
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                    title="Categories", fontsize=10)

        # 4. Save Figure
        file_name = f"corner_style_plots_final_aligned.png"
        save_path = os.path.join(self.save_path, file_name)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 

        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

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

