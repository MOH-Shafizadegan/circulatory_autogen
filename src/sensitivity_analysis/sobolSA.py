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

        self.SA_cfg = self.create_SA_cfg(self.sample_type, SA_cfg["num_samples"])

        self.set_feature_lookup_ranges()

    def set_feature_lookup_ranges(self):
        feature_lookup_ranges = {}
        for i, lookup_range in enumerate(self.obs_info.get("feature_range", [{}]*len(self.obs_info["obs_names"]))):
            if lookup_range and isinstance(lookup_range, dict) and "min" in lookup_range and "max" in lookup_range:
                feature_lookup_ranges[str(i)] = (lookup_range["min"], lookup_range["max"])
            else:
                mean = self.gt_df["value"][i]
                std = self.gt_df["std"][i]
                feature_lookup_ranges[str(i)] = (mean - std, mean + std)
            # else:
            #     feature_lookup_ranges[str(i)] = (None, None)
        
        feature_lookup_ranges[str(i+1)] = (None, None)
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
        local_sim_flags = []

        if self.rank == 0:
            # Initialize tqdm only if the current rank is 0
            pbar = tqdm(total=len(local_samples), desc=f"Rank {self.rank}", position=self.rank, leave=True)
        else:
            # Create a simple context manager that does nothing for other ranks
            class DummyPbar:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    pass
                def close(self):
                    pass
                def update(self, n=1):
                    pass # No operation
            pbar = DummyPbar()

        with pbar:
            for param_vals in local_samples:

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

                        sim_flag = f"success_after_{retry_count}tries"
                        self.sim_helper.reset_and_clear()
                    else:
                        print(f"[MPI Rank {self.rank}] Simulation failed for params: {param_vals}, subexp={subexp_count} after {retry_count} retries")
                        # Set a flag in operands_outputs_dict to indicate failure
                        operands_outputs_dict[(0, 0)] = {"failed": True}

                        sim_flag = "failed"

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

                                sim_flag = f"success_after_{retry_count}tries"
                                # reset at the end of each experiment
                                if this_sub_idx == self.protocol_info["num_sub_per_exp"][exp_idx] - 1:
                                    self.sim_helper.reset_and_clear()
                            else:
                                print(f"[MPI Rank {self.rank}] Simulation failed for params: {param_vals}, subexp={subexp_count} after {retry_count} retries")
                                # Set a flag in operands_outputs_dict to indicate failure
                                operands_outputs_dict[(exp_idx, this_sub_idx)] = {"failed": True}
                                sim_flag = "failed"
                                
                                # reset at the end of each experiment
                                if this_sub_idx == self.protocol_info["num_sub_per_exp"][exp_idx] - 1:
                                    self.sim_helper.reset_and_clear()

                features = []
                sim_flags = []
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
                            sim_flag = f"warning_{flag}"
                        else:
                            features.append(feature)

                        if hasattr(self, "feature_lookup_ranges"):
                            for i, f in enumerate(features):
                                f_min, f_max = self.feature_lookup_ranges[f"{i}"]
                                # print(f"Feature {i}: {features[i]} - ({f_min}, {f_max})")
                                if f_max != None or f_min != None:
                                    if (f_min <= f <= f_max):
                                        sim_flag = "in_lookup_range"
                                        break

                    else:
                        # WARNING: using mean biases variance estimates (shrinks variance), underestimates sensitivity
                        # TODO: come up with a better way to impute missing features
                        # Append the mean of the current features (ignoring None) -> reduces variance and bias induces toward zero
                        features.append(np.mean(features))

                    sim_flags.append(sim_flag)

                # adding cost as extra feature
                if self.param_id is not None:
                    features.append(cost)

                local_outputs.append(features)
                local_sim_flags.append(sim_flags)
                pbar.update(1)

        pbar.close()
        print(f"[MPI Rank {self.rank}] Finished processing samples {start}:{end}")

        # Gather results at rank 0
        all_outputs = self.comm.gather(local_outputs, root=0)
        all_sim_flags = self.comm.gather(local_sim_flags, root=0)

        if self.rank == 0:
            outputs = [item for sublist in all_outputs for item in sublist]
            outputs_sim_flags = [item for sublist in all_sim_flags for item in sublist]

            outputs = np.array(outputs)
            print(f"[MPI Rank 0] Gathered and flattened all outputs. Total outputs: {outputs.shape}")

            # Convert input samples to np.array
            samples_arr = np.array(samples)

            # Call the new function
            self.save_output_results(
                samples=samples_arr,
                outputs=outputs,
                outputs_sim_flags = outputs_sim_flags
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
                output_name = rf"Cost Function"
            else:
                output_name = rf"${self.obs_info['names_for_plotting'][i]}$ - experiment{self.obs_info["experiment_idxs"][i]}, subexperiment{self.obs_info["subexperiment_idxs"][i]}"
            # output_name = self.obs_info["names_for_plotting"][i] if hasattr(self, "obs_info") else f"Output_{i}"

            # Set figure width adaptively based on number of parameters (xticks)
            fig_width = max(12, 1.0 * len(self.SA_cfg["param_names"]))
            plt.figure(figsize=(fig_width, 5))
            plt.bar(x - 0.2, S1, width=0.4, label='First-order', color='blue', alpha=0.7)
            plt.bar(x + 0.2, ST, width=0.4, label='Total-order', color='red', alpha=0.7)

            param_labels = [rf"${name}$" for name in self.param_id_info["param_names_for_plotting"]]
            plt.xticks(x, param_labels, rotation=45, fontsize=8)
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
            
            if i >= len(self.obs_info['names_for_plotting']):
                output_name = rf"Cost Function"
            else:
                output_name = rf"${self.obs_info['names_for_plotting'][i]}$ - experiment{self.obs_info["experiment_idxs"][i]}, subexperiment{self.obs_info["subexperiment_idxs"][i]}"

            fig_width = max(6, 1.0 * len(self.SA_cfg["param_names"]))
            plt.figure(figsize=(fig_width, fig_width))
            param_labels = [rf"${name}$" for name in self.param_id_info["param_names_for_plotting"]]
            sns.heatmap(S2, annot=True, fmt=".2f", xticklabels=param_labels, yticklabels=param_labels, cmap="coolwarm")
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
        output_labels = self.get_sobol_output_labels(S1_all.shape[0])

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

        output_labels = self.get_sobol_output_labels(sobol_indices.shape[0])
        param_labels = [rf"${name}$" for name in self.param_id_info["param_names_for_plotting"]]
        
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

    def save_output_results(self, samples, outputs, outputs_sim_flags=None):
        """
        Saves parameters and outputs, including tracking issues, into CSV files.
        """

        # 1) Save all samples + outputs
        try:
            param_labels = self.SA_cfg["param_names"]  # from sensitivity setup
        except:
            param_labels = [f"param_{i}" for i in range(samples.shape[1])]

        output_labels = self.get_sobol_output_labels(outputs.shape[1])
        
        # Prepare DataFrame including output_sim_flags (which may be strings)
        if outputs_sim_flags is not None:
            # Ensure outputs_sim_flags is a numpy array of dtype object (to allow strings)
            outputs_sim_flags = np.array(outputs_sim_flags, dtype=object)
            # Create column names for flags
            if outputs.shape[1] > len(self.obs_info['names_for_plotting']):
                flag_labels = [
                    f"{label}_flag"
                    for label in output_labels
                    if label != "Cost Function" and label != f"feature_{outputs.shape[1]-1}"
                ]
            else:
                flag_labels = [
                    f"{label}_flag"
                    for label in output_labels
                ]
            # Concatenate samples, outputs, and flags horizontally
            df = pd.DataFrame(
            np.hstack((samples, outputs, outputs_sim_flags)),
            columns=param_labels + output_labels + flag_labels
            )
        else:
            df = pd.DataFrame(
            np.hstack((samples, outputs)),
            columns=param_labels + output_labels
            )
        file_name = "all_outputs.csv"
        save_path = os.path.join(self.save_path, file_name)
        df.to_csv(save_path, index=False)

    def load_category_data(self):
        """
        Loads parameter CSVs for failed, warning, and lookup_range categories if they exist.

        Returns:
            df_all (pd.DataFrame): Combined dataframe with parameters and Category label.
        """

        filename = "all_outputs.csv"
        file_path = os.path.join(self.save_path, filename)
        if os.path.isfile(file_path):
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")
        else:
                print(f"ℹ️ {filename} not found, skipping.")                

        return df
    
    def plot_corner_overlay(self, df: pd.DataFrame):
        """
        Creates a corner-style plot using pure Matplotlib with optimized
        axis formatting and alignment for high visual appeal and minimal overlap.
        """
        param_names = self.SA_cfg["param_names"]
        N_dims = len(param_names)
        # Find all columns ending with '_flag'
        flag_cols = [col for col in df.columns if col.endswith('_flag')]

        # Define a function to classify each row
        def classify_category(row):
            flags = row[flag_cols].astype(str).str.lower()
            if any('fail' in flag for flag in flags):
                return 'fail'
            elif any('warning' in flag for flag in flags):
                return 'warning'
            elif any('lookup' in flag for flag in flags):
                return 'lookup'
            else:
                return 'success'

        # Create a new 'Category' column based on flags
        df["Category"] = df.apply(classify_category, axis=1)
        unique_categories = ['success', 'fail', 'warning', 'lookup']
        
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
            param_labels = self.param_id_info["param_names_for_plotting"]
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
                        ax.text(0.5, 0.5, rf"${param_labels[row]}$", transform=ax.transAxes, 
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
                            ax.set_xlabel(rf"${param_labels[col]}$", fontsize=10)
                            ax.tick_params(axis='x', which='major', rotation=45, labelsize=8)
                        else:
                            ax.set_xticklabels([])
                            
                        # Y-Axis Labels: Only on the first column
                        if col == 0:
                            ax.set_ylabel(rf"${param_labels[col]}$", fontsize=10)
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

    import numpy as np

    def get_sobol_output_labels(self, num_labels):
        """
        Generates a list of output labels for Sobol sensitivity analysis plots.

        Labels are generated based on whether plotting information exists in self.obs_info
        and whether a 'Cost Function' label needs to be appended based on array shapes.

        Args:
            self (object): The instance containing the obs_info dictionary.
            sobol_indices (np.ndarray): Array used for determining the number of labels.
            S1_all (np.ndarray): Array used for determining the number of labels (often has same shape as sobol_indices).

        Returns:
            list: A list of formatted label strings.
        """
        
        names_for_plotting = self.obs_info.get('names_for_plotting', [])
        should_append_cost_function = (num_labels >= len(names_for_plotting))
        
        # Define the ending index for the list comprehension (exclusive)
        end_range = num_labels - 1 if should_append_cost_function else num_labels

        print(end_range)

        has_plotting_info = (
            hasattr(self, "obs_info") and 
            self.obs_info and 
            "names_for_plotting" in self.obs_info
        )
        
        if has_plotting_info:
            # Use a rich label format with experimental details
            def generate_label(i):
                name = self.obs_info['names_for_plotting'][i]
                # Use .get() with a default for slightly more robustness
                exp_idx = self.obs_info.get('experiment_idxs', ['?'])[i]
                sub_idx = self.obs_info.get('subexperiment_idxs', ['?'])[i]
                # The rf"..." is used to render text as LaTeX/Math Text
                return rf"{name} (Exp{exp_idx}, Sub{sub_idx})"
        else:
            # Use a generic label format
            def generate_label(i):
                return f"feature_{i}"

        output_labels = [generate_label(i) for i in range(end_range)]
        
        if should_append_cost_function:
            output_labels.append(rf"Cost Function")
            
        return output_labels

    def run(self):
        samples = self.generate_samples()
        if self.use_mpi:
            outputs = self.generate_outputs_mpi(samples)
            if self.rank == 0:
                S1_all, ST_all, S2_all = self.sobol_index(outputs)
                return S1_all, ST_all, S2_all
            else:
                return None, None, None
        else:
            outputs = self.generate_outputs(samples)
            S1_all, ST_all, S2_all = self.sobol_index(outputs)
            return S1_all, ST_all, S2_all

