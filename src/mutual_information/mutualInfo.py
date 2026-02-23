import json
import os
from sklearn.utils import shuffle
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
import matplotlib  
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from parsers.PrimitiveParsers import scriptFunctionParser
from mpi4py import MPI
from parsers.PrimitiveParsers import CSVFileParser, ObsAndParamDataParser
import csv
from tqdm import tqdm  # make sure tqdm is installed
import npeet_plus as npeet
  
class mutualInfo: 

    def __init__(self, model_path, model_out_names, solver_info, SA_info, dt, sa_output_dir,     
                param_id_path=None, params_for_id_path=None, use_MPI=False, verbose=False,    
                sim_time=2.0, pre_time=20.0):  
          
        self.model_path = model_path  
        self.output_dir = None  
        self.verbose = verbose  
        self.set_output_dir(sa_output_dir)  
  
        self.solver_info = solver_info  
        self.SA_info = SA_info  
        # Remove sample_type dependency - MI uses random sampling  
        self.num_params = None  
        self.protocol_info = None  
        self.dt = dt  
          
        # set up observables functions  
        from parsers.PrimitiveParsers import scriptFunctionParser  
        self.sfp = scriptFunctionParser()  
        self.operation_funcs_dict = self.sfp.get_operation_funcs_dict()  
          
        self.obs_and_param_parser = None  
        self.gt_df = None  
        self.obs_info = None  
              
  
        if param_id_path is not None:  
            from parsers.PrimitiveParsers import ObsAndParamDataParser  
            self.obs_and_param_parser = ObsAndParamDataParser()  
            parsed_data = self.obs_and_param_parser.parse_obs_data_json(  
                param_id_obs_path=param_id_path,  
                pre_time=pre_time,  
                sim_time=sim_time  
            )  
            self.gt_df = parsed_data["gt_df"]  
            self.protocol_info = parsed_data["protocol_info"]  
            self.prediction_info = parsed_data["prediction_info"]  
  
            self.obs_info = self.obs_and_param_parser.process_obs_info(gt_df=self.gt_df, output_dir=self.output_dir, dt=self.dt)  
            self.protocol_info = self.obs_and_param_parser.process_protocol_and_weights(  
                gt_df=self.gt_df,  
                protocol_info=self.protocol_info,  
                dt=self.dt  
            )  
  
        if self.protocol_info is None:  
            self.protocol_info = {  
                "pre_times": [pre_time],  
                "sim_times": [[sim_time]],  
                "params_to_change": [[None]]  
            }  
  
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
  
        from mpi4py import MPI  
        self.comm = MPI.COMM_WORLD  
        self.rank = self.comm.Get_rank()  
        self.num_procs = self.comm.Get_size()  
        self.use_mpi = use_MPI  
  
        self.params_for_id_path = params_for_id_path  
        self.param_id_info = None  
        if self.params_for_id_path:  
            self.param_id_info = self.obs_and_param_parser.get_param_id_info(self.params_for_id_path)  
            self.obs_and_param_parser.save_param_names(self.param_id_info, self.output_dir)  
  
        # Adjust SA_info creation for MI (no sample_type needed)  
        if self.param_id_info is not None:  
            self.SA_info = self.create_MI_info(self.SA_info["num_samples"])
    
    def create_MI_info(self, num_samples):  
        """  
        Create MI-specific configuration dictionary.  
        Unlike Sobol, MI doesn't need sample_type or Saltelli-specific fields.  
        """  
        if not hasattr(self, "param_id_info") or not self.param_id_info:  
            raise ValueError("param_id_info is not set")  
        
        MI_info = {  
            "num_samples": num_samples,  
            "param_names": [name[0] if isinstance(name, list) else name for name in self.param_id_info["param_names"]],  
            "param_mins": list(self.param_id_info["param_mins"]),  
            "param_maxs": list(self.param_id_info["param_maxs"])  
        }  
        
        self.num_params = len(MI_info["param_names"])  
        return MI_info

    def _is_rank0(self):  
        try:  
            return MPI.COMM_WORLD.Get_rank() == 0  
        except Exception:  
            return True  
  
    def _rank0_print(self, *args, **kwargs):  
        if self._is_rank0():  
            print(*args, **kwargs)

    def initialise_sim_helper(self):
        if opencor_available:
            return OpenCORSimulationHelper(self.model_path, self.dt, self.sim_time,
                                solver_info=self.solver_info, pre_time=self.pre_time)
        else:
            return PythonSimulationHelper(self.model_path, self.dt, self.sim_time,
                                solver_info=self.solver_info, pre_time=self.pre_time)
    
    def _setup_obs_and_param_data(self, param_id_path, params_for_id_path, pre_time, sim_time):  
        """  
        Initialize observation and parameter data using the same logic as sobolSA.  
        This method populates:  
            - self.obs_and_param_parser  
            - self.gt_df  
            - self.protocol_info  
            - self.prediction_info  
            - self.obs_info  
            - self.param_id_info  
        """  
        from parsers.PrimitiveParsers import ObsAndParamDataParser  
    
        # Initialize parser  
        self.obs_and_param_parser = ObsAndParamDataParser()  
    
        # Parse observation data  
        parsed_data = self.obs_and_param_parser.parse_obs_data_json(  
            param_id_obs_path=param_id_path,  
            pre_time=pre_time,  
            sim_time=sim_time  
        )  
        self.gt_df = parsed_data["gt_df"]  
        self.protocol_info = parsed_data["protocol_info"]  
        self.prediction_info = parsed_data["prediction_info"]  
    
        # Process observables  
        self.obs_info = self.obs_and_param_parser.process_obs_info(  
            gt_df=self.gt_df, output_dir=self.output_dir, dt=self.dt  
        )  
    
        # Process protocol and weights  
        self.protocol_info = self.obs_and_param_parser.process_protocol_and_weights(  
            gt_df=self.gt_df,  
            protocol_info=self.protocol_info,  
            dt=self.dt  
        )  
    
        # Parse parameter info if provided  
        if params_for_id_path:  
            self.param_id_info = self.obs_and_param_parser.get_param_id_info(params_for_id_path)  
            self.obs_and_param_parser.save_param_names(self.param_id_info, self.output_dir)
            
    def set_output_dir(self, path):
        
        self.output_dir = path
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_samples(self):  
        """Generate random samples (different from Sobol's Saltelli)"""  
        if not hasattr(self, 'param_id_info') or not self.param_id_info:  
            raise ValueError("param_id_info is not set")  
          
        n_samples = self.SA_info["num_samples"]  
        n_params = len(self.param_id_info["param_names"])  
          
        # Simple random sampling (you can customize this)  
        samples = np.random.uniform(  
            low=self.param_id_info["param_mins"],  
            high=self.param_id_info["param_maxs"],  
            size=(n_samples, n_params)  
        )  
          
        self.num_samples = n_samples  
        return samples  
    
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

        self._rank0_print(f"[MPI Rank {self.rank}] Starting samples {start}:{end} (total {len(local_samples)})")

        local_outputs = []

        # Create a single progress bar for rank 0 only to avoid noisy output from all ranks
        with tqdm(total=len(local_samples), desc=f"Rank {self.rank}", position=self.rank, leave=True, disable=self.rank != 0) as pbar:
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

                        self.sim_helper.reset_and_clear()
                    else:
                        print(f"[MPI Rank {self.rank}] Simulation failed for params: {param_vals}, after {retry_count} retries")
                        # Set a flag in operands_outputs_dict to indicate failure
                        operands_outputs_dict[(0, 0)] = {"failed": True}

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
                                self._rank0_print(f"[MPI Rank {self.rank}] Simulation failed for params: {param_vals}, subexp={subexp_count} after {retry_count} retries")
                                # Set a flag in operands_outputs_dict to indicate failure
                                operands_outputs_dict[(exp_idx, this_sub_idx)] = {"failed": True}

                                # reset at the end of each experiment
                                if this_sub_idx == self.protocol_info["num_sub_per_exp"][exp_idx] - 1:
                                    self.sim_helper.reset_and_clear()

                features = []
                for j in range(len(self.obs_info["operations"])):
                    func = self.operation_funcs_dict[self.obs_info["operations"][j]]
                    exp_idx = self.obs_info["experiment_idxs"][j]
                    subexp_idx = self.obs_info["subexperiment_idxs"][j]
                    operands_outputs = operands_outputs_dict.get((exp_idx, subexp_idx), None)
                    if operands_outputs is not None and not (isinstance(operands_outputs, dict) and operands_outputs == {"failed": True}):
                        feature = func(*operands_outputs[j], **self.obs_info["operation_kwargs"][j])
                        if feature is None or (isinstance(feature, (float, int)) and np.isnan(feature)):
                            feature = np.nanmean(features) if not np.all(np.isnan(features)) else 0.0

                        features.append(feature)
                    else:
                        # WARNING: using mean biases variance estimates (shrinks variance), underestimates sensitivity
                        # TODO: come up with a better way to impute missing features
                        # Append the mean of the current features (ignoring None) -> reduces variance and bias induces toward zero
                        features.append(np.mean(features))

                # print(f"[MPI Rank {self.rank}] sample {param_vals}, features: {features}")
                local_outputs.append(features)
                pbar.update(1)

        self._rank0_print(f"[MPI Rank {self.rank}] Finished processing samples {start}:{end}")

        # Gather results at rank 0
        all_outputs = self.comm.gather(local_outputs, root=0)

        if self.rank == 0:
            outputs = [item for sublist in all_outputs for item in sublist]
            outputs = np.array(outputs)

            # # Flatten the list of lists
            # flattened = [item for sublist in all_outputs for item in sublist]
            
            # # Sort by the global index to restore original order
            # flattened.sort(key=lambda x: x[0])
            
            # # Extract just the features
            # # outputs = np.array([features for idx, features in flattened])
            # outputs = np.array(flattened)

            self._rank0_print(f"[MPI Rank 0] Gathered and flattened all outputs. Total outputs: {outputs.shape}")
            return outputs
        else:
            return None

    def calculate_cmi_matrices_per_feature(self, p_samples, outputs):
        """
        Generates a list of matrices. 
        Each matrix corresponds to an output feature y_k.
        Cell (i, j) in the matrix is I(p_i; y_k | p_j).
        """
        n_samples, n_params = p_samples.shape
        
        # 1. Generate Y samples (n_samples, n_features)
        y_raw = np.array(outputs)
        if y_raw.ndim == 1: y_raw = y_raw.reshape(-1, 1)
        n_features = y_raw.shape[1]

        # Storage for the matrices: shape (n_features, n_params, n_params)
        all_matrices = np.zeros((n_features, n_params, n_params))

        for k in range(n_features):
            yk = y_raw[:, k].reshape(-1, 1)
            
            for i in range(n_params):
                pi = p_samples[:, i].reshape(-1, 1)
                
                for j in range(n_params):
                    if i == j:
                        # Diagonal: I(p_i; y_k) - standard Mutual Information
                        all_matrices[k, i, j] = npeet.mi(pi, yk) * np.log(2)
                    else:
                        # Off-diagonal: I(p_i; y_k | p_j)
                        pj = p_samples[:, j].reshape(-1, 1)
                        all_matrices[k, i, j] = npeet.mi(pi, yk, z=pj) * np.log(2)

        return all_matrices

    def calculate_normalised_MI_CMI(self, p, y, z=None, n_permutations = 100):
        """
        Calculates Corrected Normalised Mutual Information using Permutation.
        If z is provided, calculates CMI.
        """

        # 1. Calculate Raw MI/CMI and Entropy
        h_p = npeet.entropy(p) * np.log(2)

        if z is None:
            raw_mi = npeet.mi(p, y) * np.log(2)
        else:
            raw_mi = npeet.mi(p, y, z=z) * np.log(2)

        # 2. Permutation Loop (Null Hypothesis)
        null_values = []
        for _ in range(n_permutations):
            y_shuffled = shuffle(y) # Destroy p -> y relationship
            if z is None:
                val = npeet.mi(p, y_shuffled) * np.log(2)
            else:
                # Note: We shuffle y but keep the (p, z) relationship intact
                val = npeet.mi(p, y_shuffled, z=z) * np.log(2)
            null_values.append(val)

        # 3. Correct the Bias
        null_mean = np.mean(null_values)
        corrected_mi = max(0, raw_mi - null_mean)

        # 4. Normalise
        nmi_corrected = corrected_mi / h_p

        # Calculate Z-score for significance (optional but helpful)
        null_std = np.std(null_values) if np.std(null_values) > 0 else 1e-9
        z_score = (raw_mi - null_mean) / null_std
        
        return nmi_corrected, z_score

    def calculate_normalized_MI_matrices(self, p_samples, outputs, n_shuffles=100):
        n_samples, n_params = p_samples.shape
        y_raw = np.atleast_2d(outputs).T if np.ndim(outputs) == 1 else np.array(outputs)
        y_raw = y_raw + np.random.normal(0, 1e-1, y_raw.shape)
        n_features = y_raw.shape[1]

        all_matrices = np.zeros((n_features, n_params, n_params))

        for k in range(n_features):
            yk = y_raw[:, k].reshape(-1, 1)

            for i in range(n_params):
                pi = p_samples[:, i].reshape(-1, 1)
                
                for j in range(n_params):
                    pj = p_samples[:, j].reshape(-1, 1)
                    
                    if i == j:
                        # Diagonal: Use the helper without a 'z' (Standard MI)
                        val, _ = self.calculate_normalised_MI_CMI(pi, yk, n_permutations=n_shuffles)
                    else:
                        # Off-diagonal: Use the helper with pj as 'z' (Conditional MI)
                        val, _ = self.calculate_normalised_MI_CMI(pi, yk, z=pj, n_permutations=n_shuffles)
                    
                    all_matrices[k, i, j] = val
                        
        return all_matrices

    def plot_cmi_feature_matrices(self, cmi_matrices):
        """
        Saves a 2D heatmap for each model output feature.
        
        Parameters:
            cmi_matrices (np.ndarray): Shape (n_features, n_params, n_params)
                                    where [k, i, j] is I(pi; yk | pj)
        """
        if self.rank != 0:
            return

        print("\nGenerating CMI Feature Heatmaps...")

        # 1. Define Axis Labels
        # Use your existing logic for output/param labels
        output_labels = self.get_output_labels(cmi_matrices.shape[0])
        param_labels = [rf"{name}" for name in self.param_id_info["param_names_for_plotting"]]

        # Total samples used for NPEET estimation
        title_prefix = f"CMI Dependency (N={self.num_samples})"

        # 2. Iterate through each feature k
        for k in range(cmi_matrices.shape[0]):
            feature_data = cmi_matrices[k]
            feature_name = output_labels[k]
            
            # Convert to DataFrame for easier plotting
            df_data = pd.DataFrame(feature_data, index=param_labels, columns=param_labels)
            
            # Scaling figure size based on parameter count
            fig_dim = max(8, len(param_labels) * 0.8)
            plt.figure(figsize=(fig_dim, fig_dim * 0.8))
            
            sns.heatmap(
                df_data,
                annot=True,               
                fmt=".3f",                # 3 decimals for MI is usually better
                cmap="rocket",            # Different colormap to distinguish from Sobol
                linewidths=0.5,           
                linecolor='lightgray',
                cbar_kws={'label': 'CMI (Nats)'}
            )

            plt.title(f'{title_prefix}\nFeature: {feature_name}', fontsize=14)
            plt.xlabel('Conditioning Parameter ($p_j$)', fontsize=12)
            plt.ylabel('Target Parameter ($p_i$)', fontsize=12)
            
            plt.xticks(rotation=45, ha='right', fontsize=9) 
            plt.yticks(rotation=0, fontsize=9) 
            
            plt.tight_layout()
            
            # 3. Save following your specific naming convention
            clean_feature_name = str(feature_name).replace(' ', '_').replace('/', '_')
            file_name = f"CMI_Matrix_{clean_feature_name}.png"
            save_path = os.path.join(self.output_dir, file_name)
            
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved CMI matrix for {feature_name} to {save_path}")

    def get_output_labels(self, num_labels):
        """
        Generates a list of output labels for Sobol sensitivity analysis plots.

        Labels are generated based on whether plotting information exists in self.obs_info
        
        Args:
            self (object): The instance containing the obs_info dictionary.
            sobol_indices (np.ndarray): Array used for determining the number of labels.
            S1_all (np.ndarray): Array used for determining the number of labels (often has same shape as sobol_indices).

        Returns:
            list: A list of formatted label strings.
        """
        
        end_range = num_labels

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
            
        return output_labels
    
    def interpret_mi_results(self, normalized_mi_matrices, save_to_file=True):  
        """  
        Interpret normalized MI/CMI matrices per feature and generate identifiability reports.  
        
        Args:  
            normalized_mi_matrices (np.ndarray): Shape (n_features, n_params, n_params)  
            save_to_file (bool): Whether to save the reports to files  
        """  
        if self.rank != 0:  
            return  
    
        n_features, n_params, _ = normalized_mi_matrices.shape  
        param_names = self.param_id_info["param_names_for_plotting"]  
        feature_labels = self.get_output_labels(n_features)  # reuse sobolSA helper [1](#8-0)   
    
        all_results = []  
    
        for k in range(n_features):  
            feature_matrix = normalized_mi_matrices[k]  
            feature_name = feature_labels[k]  
    
            # Marginal MI = diagonal; Conditional MI = mean off-diagonal per param  
            marginal_mi = np.diagonal(feature_matrix)  
            conditional_mi = np.array([  
                np.mean(np.delete(feature_matrix[i, :], i)) for i in range(n_params)  
            ])  
    
            print(f"\n{'='*90}")  
            print(f"Feature: {feature_name}")  
            print(f"{'Parameter':<20} | {'NMI (Marginal)':<15} | {'NCMI (Cond.)':<15} | {'Status'}")  
            print("-" * 90)  
    
            for i, name in enumerate(param_names):  
                nmi = marginal_mi[i]  
                ncmi = conditional_mi[i]  
    
                if ncmi < 0.05:  
                    status = "🔴 Non-Identifiable"  
                elif ncmi > nmi * 1.5:  
                    status = "🟡 Interaction/Conditional"  
                elif nmi > ncmi * 1.5:  
                    status = "🟠 Redundant/Correlated"  
                else:  
                    status = "🟢 Strongly Identifiable"  
    
                print(f"{name:<20} | {nmi:<15.4f} | {ncmi:<15.4f} | {status}")  
                all_results.append({  
                    'feature': feature_name,  
                    'parameter': name,  
                    'nmi': nmi,  
                    'ncmi': ncmi,  
                    'status': status  
                })  
            print("="*90)  
    
        if save_to_file:  
            df_all = pd.DataFrame(all_results)  
            output_path = os.path.join(self.output_dir, "mi_interpretation_per_feature.csv")  
            df_all.to_csv(output_path, index=False)  
            print(f"Per-feature interpretation report saved to: {output_path}")
            
    def run(self):  
        """Main execution method"""  
        
        if self.rank == 0:
            samples = self.generate_samples()
        else:
            samples = None
        
        samples = self.comm.bcast(samples, root=0)

        if self.use_mpi:  
            outputs = self.generate_outputs_mpi(samples)  
            if self.rank == 0:  
                normalized_cmi_matrices = self.calculate_normalized_MI_matrices(samples, outputs)  
                self.plot_cmi_feature_matrices(normalized_cmi_matrices) 
                self.interpret_mi_results(normalized_cmi_matrices) 
                return normalized_cmi_matrices  
            else:  
                return None  
        else:  
            outputs = self.generate_outputs(samples)  
            normalized_cmi_matrices = self.calculate_normalized_MI_matrices(samples, outputs)  
            self.plot_cmi_feature_matrices(normalized_cmi_matrices)  
            return normalized_cmi_matrices
