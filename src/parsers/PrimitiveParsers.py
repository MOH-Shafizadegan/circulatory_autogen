'''
Created on 29/10/2021

@author: Finbar Argus, Gonzalo D. Maso Talou
'''

import pandas as pd
import numpy as np
import os
import sys
import csv
import json
import copy
import yaml
import re

root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))
user_inputs_dir = os.path.join(root_dir, 'user_run_files')
src_dir = os.path.join(os.path.dirname(__file__), '..')
param_id_dir = os.path.join(src_dir, 'param_id')
base_dir = os.path.join(src_dir, '..')
operation_funcs_user_dir = os.path.join(base_dir, 'funcs_user')

class scriptFunctionParser(object):
    '''
    Parses scripts with functions into objects (dicts) which holds the functions
    '''

    def __init__(self):
        sys.path.append(param_id_dir)
        sys.path.append(operation_funcs_user_dir)
        '''
        Constructor
        '''
    
    def get_operation_funcs_dict(self):
        import operation_funcs
        import operation_funcs_user
        operation_funcs_dict = {}
        funcs = [item for item in dir(operation_funcs) if callable(getattr(operation_funcs, item))]
        funcs_user = [item for item in dir(operation_funcs_user) if callable(getattr(operation_funcs_user, item))]

        # create dict with keys of string of function names
        for func in funcs:
            operation_funcs_dict[func] = getattr(operation_funcs, func)
        for func in funcs_user:
            operation_funcs_dict[func] = getattr(operation_funcs_user, func)
        
        return operation_funcs_dict

    def get_cost_funcs_dict(self):
        # import cost_funcs # currently all costs are in cost_funcs_user
        import cost_funcs_user
        cost_funcs_dict = {}
        # funcs = [item for item in dir(cost_funcs) if callable(getattr(cost_funcs, item))]
        funcs_user = [item for item in dir(cost_funcs_user) if callable(getattr(cost_funcs_user, item))]

        # create dict with keys of string of function names
        # for func in funcs:
        #     cost_funcs_dict[func] = getattr(cost_funcs, func)
        for func in funcs_user:
            cost_funcs_dict[func] = getattr(cost_funcs_user, func)
        
        return cost_funcs_dict

class YamlFileParser(object):
    '''
    Parses Yaml files 
    '''
    def __init__(self):
        '''
        Constructor
        '''
    
    def parse_user_inputs_file(self, inp_data_dict, obs_path_needed=False, do_generation_with_fit_parameters=False):
        
        if inp_data_dict is None:
            with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
                inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
            if "user_inputs_path_override" in inp_data_dict.keys() and inp_data_dict["user_inputs_path_override"]:
                if os.path.exists(inp_data_dict["user_inputs_path_override"]):
                    user_files_dir = os.path.dirname(inp_data_dict["user_inputs_path_override"])
                    with open(inp_data_dict["user_inputs_path_override"], 'r') as file:
                        inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)
                else:
                    print(f"User inputs file not found at {inp_data_dict['user_inputs_path_override']}")
                    print("Check the user_inputs_path_override key in user_inputs.yaml and set it to False if "
                            "you want to use the default user_inputs.yaml location")
                    exit()
            else:
                user_files_dir = ''
        else:
            user_files_dir = ''
    
        file_prefix = inp_data_dict['file_prefix']

        # overwrite dir paths if set in user_inputs.yaml
        if "resources_dir" in inp_data_dict.keys():
            inp_data_dict['resources_dir'] = os.path.join(user_files_dir, inp_data_dict['resources_dir'])
        else:
            inp_data_dict['resources_dir'] = os.path.join(root_dir, 'resources')
        if "param_id_output_dir" in inp_data_dict.keys():
            inp_data_dict['param_id_output_dir'] = os.path.join(user_files_dir, inp_data_dict['param_id_output_dir'])
        else:
            inp_data_dict['param_id_output_dir'] = os.path.join(root_dir, 'param_id_output')
        if "generated_models_dir" in inp_data_dict.keys():
            inp_data_dict['generated_models_dir'] = os.path.join(user_files_dir, inp_data_dict['generated_models_dir'])
        else:
            inp_data_dict['generated_models_dir'] = os.path.join(root_dir, 'generated_models')
        
        if obs_path_needed:
            if 'param_id_obs_path' in inp_data_dict.keys():
                inp_data_dict['param_id_obs_path'] = os.path.join(user_files_dir, inp_data_dict['param_id_obs_path'])
                if not os.path.exists(inp_data_dict['param_id_obs_path']):
                    print(f'param_id_obs_path={inp_data_dict["param_id_obs_path"]} does not exist')
                    exit()
            else:
                print(f'param_id_obs_path needs to be defined in user_inputs.yaml')
                exit()

            if 'params_for_id_file' in inp_data_dict.keys():
                inp_data_dict['params_for_id_path'] = os.path.join(inp_data_dict['resources_dir'], inp_data_dict['params_for_id_file'])
            else:
                inp_data_dict['params_for_id_path'] = os.path.join(inp_data_dict['resources_dir'], f'{file_prefix}_params_for_id.csv')

            if not os.path.exists(inp_data_dict['params_for_id_path']):
                print(f'params_for_id path of {inp_data_dict["params_for_id_path"]} doesn\'t exist, user must create this file')
                exit()

        if do_generation_with_fit_parameters:
            data_str_addon = re.sub('.json', '', os.path.split(inp_data_dict['param_id_obs_path'])[1])
            inp_data_dict['param_id_output_dir_abs_path'] = os.path.join(inp_data_dict['param_id_output_dir'], 
                                                                         inp_data_dict['param_id_method'] + f'_{file_prefix}_{data_str_addon}')
            inp_data_dict['generated_models_subdir'] = os.path.join(inp_data_dict['generated_models_dir'], 
                                                                    file_prefix + '_' + data_str_addon)
        else:
            inp_data_dict['generated_models_subdir'] = os.path.join(inp_data_dict['generated_models_dir'], file_prefix)
        
        if not os.path.exists(inp_data_dict['generated_models_dir']):
            os.mkdir(inp_data_dict['generated_models_dir'])

        if not os.path.exists(inp_data_dict['generated_models_subdir']):
            os.mkdir(inp_data_dict['generated_models_subdir'])
            
        inp_data_dict['model_path'] = os.path.join(inp_data_dict['generated_models_subdir'], f'{file_prefix}.cellml')

        if do_generation_with_fit_parameters:
            inp_data_dict['uncalibrated_model_path'] = os.path.join(inp_data_dict["generated_models_dir"], file_prefix, 
                                               file_prefix + '.cellml')
        else:
            inp_data_dict['uncalibrated_model_path'] = inp_data_dict['model_path']


        if 'pre_time' in inp_data_dict.keys():
            inp_data_dict['pre_time'] = inp_data_dict['pre_time']
        else:
            inp_data_dict['pre_time'] = None
        if 'sim_time' in inp_data_dict.keys():
            inp_data_dict['sim_time'] = inp_data_dict['sim_time']
        else:
            inp_data_dict['sim_time'] = None

        if inp_data_dict['solver_info'] is None:
            print('solver_info must be defined in user_inputs.yaml',
                'MaximumStep is now an entry of solver_info in the user_inputs.yaml file')
            exit()

        if 'DEBUG' in inp_data_dict.keys(): 
            if inp_data_dict['DEBUG']:
                inp_data_dict['ga_options'] = inp_data_dict['debug_ga_options']
                inp_data_dict['mcmc_options'] = inp_data_dict['debug_mcmc_options']
            else:
                pass
        else:
            inp_data_dict['DEBUG'] = False

        if not 'external_modules_dir' in inp_data_dict.keys():
            inp_data_dict['external_modules_dir'] = None
        
        # for sensitivity analysis and parameter identification
        if not 'sa_options' in inp_data_dict.keys():
            inp_data_dict['sa_options'] = None

        if inp_data_dict['sa_options'] is None:
            inp_data_dict['sa_options'] = {
                'method': 'sobol',
                'num_samples': 32,
                'sample_type': 'saltelli',
                'output_dir': os.path.join(root_dir, 'sensitivity_outputs', file_prefix + '_SA_results')
            }
        else:
            if 'output_dir' not in inp_data_dict['sa_options'].keys():
                inp_data_dict['sa_options']['output_dir'] = os.path.join(root_dir, 'sensitivity_outputs', file_prefix + '_SA_results')  
            else:
                if not os.path.isabs(inp_data_dict['sa_options']['output_dir']):
                    inp_data_dict['sa_options']['output_dir'] = os.path.join(root_dir, 'sensitivity_outputs', inp_data_dict['sa_options']['output_dir']) 
            
            if not os.path.exists(inp_data_dict['sa_options']['output_dir']):
                os.makedirs(inp_data_dict['sa_options']['output_dir'], exist_ok=True)
            
            if 'method' not in inp_data_dict['sa_options'].keys():
                print('No method specified for sensitivity analysis, setting to sobol by default')
                inp_data_dict['sa_options']['method'] = 'sobol'
            if 'num_samples' not in inp_data_dict['sa_options'].keys():
                print('No num_samples specified for sensitivity analysis, setting to 32 by default')
                inp_data_dict['sa_options']['num_samples'] = 32
            if 'sample_type' not in inp_data_dict['sa_options'].keys():
                print('No sample_type specified for sensitivity analysis, setting to saltelli by default')
                inp_data_dict['sa_options']['sample_type'] = 'saltelli'
            
        if 'do_ia' not in inp_data_dict.keys():
            inp_data_dict['do_ia'] = False
        
        if 'ia_options' not in inp_data_dict.keys():
            inp_data_dict['ia_options'] = {
                'method': 'Laplace'
            }
        else:
            if 'method' not in inp_data_dict['ia_options'].keys():
                print('No method specified for identifiability analysis, setting to Laplace by default')
                inp_data_dict['ia_options']['method'] = 'Laplace'

        # for generation only
    
        inp_data_dict['vessels_csv_abs_path'] = os.path.join(inp_data_dict['resources_dir'], file_prefix + '_vessel_array.csv')
        inp_data_dict['parameters_csv_abs_path'] = os.path.join(inp_data_dict['resources_dir'], inp_data_dict['input_param_file'])

        return inp_data_dict

class CSVFileParser(object):
    '''
    Parses CSV files
    '''

    def __init__(self):
        '''
        Constructor
        '''
        
    def get_data_as_dataframe_multistrings(self, filename, has_header=True):
        '''
        Returns the data in the CSV file as a Pandas dataframe where entries in the data array that have two
        entries are put in a list in the entry for the dataframe
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''
        if( has_header ):
            csv_dataframe = pd.read_csv(filename, dtype=str, na_filter=False)
        else:
            csv_dataframe = pd.read_csv(filename, dtype=str, header=None, na_filter=False)

        csv_dataframe = csv_dataframe.rename(columns=lambda x: x.strip())
        for II in range(csv_dataframe.shape[0]):
            for column_name in csv_dataframe.columns:
                entry = csv_dataframe[column_name][II]
                if type(entry) is not str:
                    sub_entries = []
                else:
                    sub_entries = entry.split()

                if column_name in ['vessel_name', 'inp_vessels', 'out_vessels']:
                    if sub_entries == []:
                        new_entry = []
                        pass
                    else:
                        new_entry = [sub_entry.strip() for sub_entry in sub_entries]
                else:
                    if sub_entries == []:
                        new_entry = []
                    else:
                        new_entry = sub_entries[0].strip()

                csv_dataframe.loc[II, column_name] = new_entry

        # for column_name in csv_dataframe.columns:
        #     if column_name == 'vessel_name':
        #         continue
        #     csv_dataframe[column_name] = csv_dataframe[column_name].str.strip()
    
        return csv_dataframe

    def get_data_as_dataframe(self, filename, has_header=True):
        '''
        Returns the data in the CSV file as a Pandas dataframe
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''
        if (has_header):
            csv_dataframe = pd.read_csv(filename, dtype=str)
        else:
            csv_dataframe = pd.read_csv(filename, dtype=str, header=None)

        for column_name in csv_dataframe.columns:
            csv_dataframe[column_name] = csv_dataframe[column_name].str.strip()

        return csv_dataframe

    def get_data_as_nparray(self,filename,has_header=True):
        '''
        Returns the data in the CSV file as a numpy array
        :param filename: filename of CSV file
        :param has_header: If CSV file has a header
        '''

        csv_dataframe = self.get_data_as_dataframe(filename, has_header)
    
        csv_np_array = csv_dataframe.to_numpy()
        dtypes = []
        for column in list(csv_dataframe.columns):
            dtypes.append((column,'<U80'))
            
        csv_np_array = np.array(list(zip(*csv_np_array.T)), dtype=dtypes)
    
        return csv_np_array

    def get_data_as_dictionary(self,filename):
        '''
        Returns the data in the CSV file as a Python dictionary
        :param filename: filename of CSV file
        '''

        csv_dataframe = self.get_data_as_dataframe(filename).T
        csv_dictionary = csv_dataframe.to_dict()
        
        return list(csv_dictionary.values())


    def get_param_id_params_as_lists_of_tuples(self, param_id_dir):

        param_names = []

        with open(os.path.join(os.path.join(param_id_dir, 'param_names_for_gen.csv')), 'r') as f:
            rd = csv.reader(f)
            for row in rd:
                param_names.append(row)


        # get date identifier of the parameter id
        date_id = np.load(os.path.join(os.path.join(param_id_dir, 'date.npy'))).item()

        param_vals = np.load(os.path.join(param_id_dir, 'best_param_vals.npy'))
        param_name_and_val = []
        
        for name_or_list, val in zip(param_names, param_vals):
            if isinstance(name_or_list, list):
                for name in name_or_list:
                    param_name_and_val.append((name, val))
            else:
                param_name_and_val.append((name, val))

        return param_name_and_val, date_id

    def get_param_id_info(self, params_for_id_path, idxs_to_ignore= None):
    
        if not params_for_id_path:
            print(f'params_for_id_path cannot be None, exiting')
            return None

        csv_parser = CSVFileParser()
        input_params = csv_parser.get_data_as_dataframe_multistrings(params_for_id_path)

        # --- 1. Filter the DataFrame first ---
        # Create a mask for indices to KEEP (not ignore)
        if idxs_to_ignore is not None:
            all_indices = set(range(input_params.shape[0]))
            valid_indices = sorted(list(all_indices - set(idxs_to_ignore)))
            # Filter the DataFrame based on valid indices
            # .copy() is used to avoid SettingWithCopyWarning, though reset_index usually handles this
            filtered_params = input_params.iloc[valid_indices].reset_index(drop=True)
        else:
            filtered_params = input_params.copy()
            
        N_params = filtered_params.shape[0]

        param_id_info = {}
        param_names_for_gen = []
        param_id_info["param_names"] = [] # The list of names to be stored

        # --- 2. Iterate ONLY over the filtered data ---
        for II in range(N_params):
            # Current row data from the filtered DataFrame
            row = filtered_params.iloc[II]

            # A. Build the full, complex names (e.g., 'vessel_name/param_name')
            param_full_names = [
                row["vessel_name"][JJ] + '/' + row["param_name"] 
                for JJ in range(len(row["vessel_name"]))
            ]
            param_id_info["param_names"].append(param_full_names)

            # B. Build the simplified names for generator/code
            if row["vessel_name"][0] == 'global':
                param_names_for_gen.append([row["param_name"]])
            else:
                param_gen_names = [
                    row["param_name"] + '_' + row["vessel_name"][JJ] 
                    for JJ in range(len(row["vessel_name"]))
                ]
                param_names_for_gen.append(param_gen_names)
        
        # --- 3. Set Arrays using the filtered DataFrame (Simple Array Creation) ---

        param_id_info["param_mins"] = filtered_params["min"].to_numpy(dtype=float)
        param_id_info["param_maxs"] = filtered_params["max"].to_numpy(dtype=float)
        
        # Plotting Names
        if "name_for_plotting" in filtered_params.columns:
            param_id_info["param_names_for_plotting"] = filtered_params["name_for_plotting"].to_numpy()
        else:
            # Use the first element of the complex name list generated above
            param_id_info["param_names_for_plotting"] = np.array([p_names[0] 
                                                                    for p_names in param_id_info["param_names"]])
        
        # Priors
        if "prior" in filtered_params.columns:
            param_id_info["param_prior_types"] = filtered_params["prior"].to_numpy()
        else:
            param_id_info["param_prior_types"] = np.array(["uniform"] * N_params)

        param_id_info["param_names_for_gen"] = param_names_for_gen
        
        return param_id_info

    def save_param_names(self, param_id_info, output_dir, rank=0):
        """
        Saves the generated parameter names and generator names to CSV files.
        Requires the dictionary returned by _process_param_info.
        """
        if rank == 0:
            # 1. Save param_names (vessel_name/param_name format)
            param_names_path = os.path.join(output_dir, 'param_names.csv')
            with open(param_names_path, 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerows(param_id_info["param_names"])
            
            # 2. Save param_names_for_gen (simplified format)
            param_gen_path = os.path.join(output_dir, 'param_names_for_gen.csv')
            with open(param_gen_path, 'w', newline='') as f:
                wr = csv.writer(f)
                wr.writerows(param_id_info["param_names_for_gen"])
        return

class JSONFileParser(object):
    '''
    Parses json files
    '''

    def __init__(self):
        '''
        Constructor
        '''

    def json_to_dataframe(self, json_path):
        with open(json_path, encoding='utf-8-sig') as rf:
            json_obj = json.load(rf)
        df = pd.DataFrame(json_obj)
        return df

    def json_to_dataframe_with_user_dir(self, json_dir, json_user_dir, external_modules_dir):
        dfs = [self.json_to_dataframe(os.path.join(json_dir, file)) \
                for file in os.listdir(json_dir) if file.endswith('.json')]
        user_module_dfs = [self.json_to_dataframe(os.path.join(json_user_dir, file)) \
                for file in os.listdir(json_user_dir) if file.endswith('.json')]
        if external_modules_dir is not None:
            external_module_dfs = [self.json_to_dataframe(os.path.join(external_modules_dir, file)) \
                    for file in os.listdir(external_modules_dir) if file.endswith('.json')]
        else:
            external_module_dfs = []
            
        df = None
        for json_df in dfs:
            if df is None:
                df = json_df
            else:
                # concatenate dataframes, ignore index to reset the index
                # so that it is not duplicated
                df = pd.concat([df, json_df], ignore_index=True)

        for user_module_df in user_module_dfs:
            df = pd.concat([df, user_module_df], ignore_index=True)
        for external_module_df in external_module_dfs:
            df = pd.concat([df, external_module_df], ignore_index=True)
        return df

    def append_module_config_info_to_vessel_df(self, vessel_df, module_df):
        # add columns to vessel_df

        add_on_lists = {column:[] for column in module_df.columns[2:]}
        for vessel_tup in vessel_df.itertuples():
            vessel_type = vessel_tup.vessel_type
            BC_type = vessel_tup.BC_type
            if len(BC_type) <1 or len(vessel_type) <1:
                print('You have an empty entry in your vessel array, exiting')
                exit()
            this_vessel_module_df = module_df.loc[((module_df["vessel_type"] == vessel_type)
                                                   & (module_df["BC_type"] == BC_type))].squeeze()
            if this_vessel_module_df.empty:
                print(f'combination of vessel_type = {vessel_type} and BC_type = {BC_type} doesn\'t exist, check module_config.json',
                        'for this combination')
                exit()
            for column in add_on_lists:
                # deepcopy to make sure that the lists for different vessel same module are not linked
                try:
                    if np.isnan(this_vessel_module_df[column]):
                        add_on_lists[column].append("None")
                    else:
                        add_on_lists[column].append(copy.deepcopy(this_vessel_module_df[column]))
                except:
                    add_on_lists[column].append(copy.deepcopy(this_vessel_module_df[column]))

        for column in add_on_lists:
            vessel_df[column] = add_on_lists[column]

    def _parse_json_data(self, param_id_obs_path, pre_time=None, sim_time=None):
        """
        Loads the ground truth observation data from the JSON file and returns 
        the core data structures: gt_df, protocol_info, and prediction_info.
        """
        
        try:
            with open(param_id_obs_path, encoding='utf-8-sig') as rf:
                json_obj = json.load(rf)
        except FileNotFoundError:
            print(f"Error: File not found at {param_id_obs_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in file at {param_id_obs_path}")
            return None

        gt_df, protocol_info, prediction_info = None, None, None

        # --- Case 1: Simple list of data items ---
        if type(json_obj) == list:
            gt_df = pd.DataFrame(json_obj)
            protocol_info = {"pre_times": [pre_time], 
                             "sim_times": [[sim_time]],
                             "params_to_change": [[None]]}
            prediction_info = {'names': [], 'units': [], 'names_for_plotting': [], 'experiment_idxs': []}
            

        # --- Case 2: Dictionary structure ---
        elif type(json_obj) == dict:
            # Load Data Items (gt_df)
            if 'data_items' in json_obj.keys():
                gt_df = pd.DataFrame(json_obj['data_items'])
            elif 'data_item' in json_obj.keys():
                gt_df = pd.DataFrame(json_obj['data_item']) 
            else:
                print("data_items not found in json object. ",
                      "Please check that data_items is the key for the list of data items")

            # Load Protocol Info
            if 'protocol_info' in json_obj.keys():
                protocol_info = json_obj['protocol_info']
                if "sim_times" not in protocol_info: protocol_info["sim_times"] = [[sim_time]]
                if "pre_times" not in protocol_info: protocol_info["pre_times"] = [pre_time]
            else:
                if pre_time is None or sim_time is None:
                    print("protocol_info not found in json object. ",
                          "If this is the case sim_time and pre_time must be set",
                          "in the user_inputs.yaml file")
                    exit()
                protocol_info = {"pre_times": [pre_time], "sim_times": [[sim_time]], "params_to_change": [[None]]}

            # Load Prediction Info
            if 'prediction_items' in json_obj.keys():
                prediction_info = {'names': [], 'units': [], 'names_for_plotting': [], 'experiment_idxs': []}
                for entry in json_obj['prediction_items']:
                    if 'variable' not in entry: print('"variable" missing, exiting'); exit()
                    if 'unit' not in entry: print('"unit" missing, exiting'); exit()
                    
                    prediction_info['names'].append(entry['variable'])
                    prediction_info['units'].append(entry['unit'])
                    prediction_info['names_for_plotting'].append(entry.get('name_for_plotting', entry['variable']))
                    prediction_info['experiment_idxs'].append(entry.get('experiment_idx', 0))
            else:
                prediction_info = None
            
        else:
            print(f"Error: unknown data type for imported json object of {type(json_obj)}")
            return None
        
        return {
            "gt_df": gt_df, 
            "protocol_info": protocol_info, 
            "prediction_info": prediction_info
        }

    def _process_obs_info(self, gt_df):
        """
        Generates the detailed obs_info dictionary, including names, units, 
        plotting defaults, operations, and kwargs from the ground truth dataframe.
        """
        obs_info = {}
        
        # --- Simple Array Generation ---
        N = gt_df.shape[0]
        
        obs_info["obs_names"] = gt_df["variable"].tolist()
        obs_info["data_types"] = gt_df["data_type"].tolist()
        obs_info["units"] = gt_df["unit"].tolist()
        obs_info["experiment_idxs"] = [gt_df.iloc[II].get("experiment_idx", 0) for II in range(N)]
        obs_info["subexperiment_idxs"] = [gt_df.iloc[II].get("subexperiment_idx", 0) for II in range(N)]

        # --- Plotting Colors ---
        possible_colors = ['b', 'g', 'c', 'm', 'y', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:orange']
        obs_info["plot_colors"] = [gt_df.iloc[II].get("plot_color", possible_colors[II % len(possible_colors)]) 
                                        for II in range(N)]
        
        # --- Plotting Type Defaults (Logic preserved) ---
        obs_info["plot_type"] = []
        warning_printed = False
        for II in range(N):
            if "plot_type" not in gt_df.iloc[II].keys():
                if gt_df.iloc[II]["data_type"] == "constant":
                    if not warning_printed:
                        print('constant data types plot type defaults to horizontal lines',
                            'change "plot_type" in obs_data.json to change this')
                        warning_printed = True
                    obs_info["plot_type"].append("horizontal")
                elif gt_df.iloc[II]["data_type"] == "prob_dist":
                    if not warning_printed:
                        print('prob_dist data types plot type defaults to horizontal lines',
                            'change "plot_type" in obs_data.json to change this')
                        warning_printed = True
                    obs_info["plot_type"].append("horizontal")
                elif gt_df.iloc[II]["data_type"] == "series":
                    obs_info["plot_type"].append("series")
                elif gt_df.iloc[II]["data_type"] == "frequency":
                    obs_info["plot_type"].append("frequency")
                elif gt_df.iloc[II]["data_type"] == "plot_dist":
                    obs_info["plot_type"].append("horizontal")
                else:
                    print(f'data type {gt_df.iloc[II]["data_type"]} not recognised')
            else:
                obs_info["plot_type"].append(gt_df.iloc[II]["plot_type"])
                if obs_info["plot_type"][II] in ["None", "null", "Null", "none", "NONE"]:
                    obs_info["plot_type"][II] = None

        # --- Operations (Mapping obs_type to operation) ---
        obs_info["operations"] = []
        obs_info["operands"] = []
        obs_info["operation_kwargs"] = [gt_df.iloc[II].get("operation_kwargs", {}) for II in range(N)]
        obs_info["freqs"] = [gt_df.iloc[II].get("frequencies") for II in range(N)]
        obs_info["names_for_plotting"] = [gt_df.iloc[II].get("name_for_plotting", obs_info["obs_names"][II]) for II in range(N)]

        for II in range(N):
            op = gt_df.iloc[II].get("operation")
            obs_type = gt_df.iloc[II].get("obs_type")
            operands = gt_df.iloc[II].get("operands")

            if op in ["Null", "None", "null", "none", "", "nan", np.nan, None]:
                if obs_type in ["series", "frequency"]:
                    obs_info["operations"].append(None)
                    obs_info["operands"].append(operands)
                elif obs_type in ["min", "max", "mean"]: 
                    obs_info["operations"].append(obs_type)
                    obs_info["operands"].append([gt_df.iloc[II]["variable"]])
                else:
                    obs_info["operations"].append(None)
                    obs_info["operands"].append(operands)
            else:
                obs_info["operations"].append(op)
                obs_info["operands"].append(operands)

        # --- Weights and Cost Types ---
        weights = gt_df["weight"].to_numpy()
        data_types = np.array(obs_info["data_types"])
        
        obs_info["num_obs"] = N
        obs_info["weight_const_vec"] = weights[data_types == "constant"]
        obs_info["weight_series_vec"] = weights[data_types == "series"]
        obs_info["weight_amp_vec"] = weights[data_types == "frequency"]
        obs_info["weight_prob_dist_vec"] = weights[data_types == "prob_dist"]

        phase_weights = gt_df.get("phase_weight", pd.Series([1] * N))
        obs_info["weight_phase_vec"] = phase_weights[data_types == "frequency"].to_numpy()

        obs_info["cost_type"] = [gt_df.iloc[II].get("cost_type", "MSE") for II in range(N)]
        
        return obs_info

    def _process_protocol_and_weights(self, gt_df, protocol_info, dt):
        """
        Calculates time totals, validates protocol labels/colors, and generates 
        the scaled weight maps for experiment/subexperiment cost calculation.
        """
        protocol = protocol_info
        df = gt_df
        
        # --- Protocol Info Preprocessing ---
        protocol['num_experiments'] = len(protocol["sim_times"])
        protocol['num_sub_per_exp'] = [len(protocol["sim_times"][exp_idx]) for exp_idx in range(protocol["num_experiments"])]
        protocol['num_sub_total'] = sum(protocol['num_sub_per_exp'])
        
        protocol["total_sim_times_per_exp"] = []
        protocol["tSims_per_exp"] = []
        protocol["num_steps_total_per_exp"] = []

        for exp_idx in range(protocol['num_experiments']):
            total_sim_time = np.sum(protocol["sim_times"][exp_idx])
            num_steps_total = int(total_sim_time / dt)
            tSim_per_exp = np.linspace(0.0, total_sim_time, num_steps_total + 1)
            
            protocol["total_sim_times_per_exp"].append(total_sim_time)
            protocol["tSims_per_exp"].append(tSim_per_exp)
            protocol["num_steps_total_per_exp"].append(num_steps_total)
            
        # --- Protocol Info Validation ---
        N_exp = protocol['num_experiments']
        
        if "experiment_colors" not in protocol:
            protocol["experiment_colors"] = ['r'] * N_exp
        elif len(protocol["experiment_colors"]) != N_exp:
            print('Error: experiment_colors length does not match num_experiments, exiting')
            exit()
            
        if "experiment_labels" not in protocol:
            protocol["experiment_labels"] = [None] * N_exp
        elif len(protocol["experiment_labels"]) != N_exp:
            print('Error: experiment_labels length does not match num_experiments, exiting')
            exit()

        # --- Weight Mapping Initialization ---
        
        # Ensure experiment_idx and subexperiment_idx exist in the DataFrame
        # IMPORTANT: These columns must be added safely if they don't exist
        df["experiment_idx"] = df.apply(lambda row: row.get("experiment_idx", 0), axis=1)
        df["subexperiment_idx"] = df.apply(lambda row: row.get("subexperiment_idx", 0), axis=1)

        # Initialize nested lists for weight maps (one list per data type)
        const_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]
        series_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]
        amp_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]
        phase_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]
        prob_dist_map = [[[] for _ in range(protocol['num_sub_per_exp'][exp_idx])] for exp_idx in range(N_exp)]

        
        # --- Calculate Scaled Weight Maps ---
        
        for exp_idx in range(N_exp):
            for this_sub_idx in range(protocol['num_sub_per_exp'][exp_idx]):
                
                # Mask to find observations belonging to the current experiment/subexperiment
                mask = (df["experiment_idx"] == exp_idx) & (df["subexperiment_idx"] == this_sub_idx)
                
                # Iterate over all possible data types
                for data_type, weight_map in [
                    ("constant", const_map), ("series", series_map), ("frequency", amp_map), ("prob_dist", prob_dist_map)
                ]:
                    # Create the full weight vector (Weight if matched, 0.0 otherwise)
                    full_weights = np.where(mask & (df["data_type"] == data_type), df["weight"], 0.0)
                    weight_map[exp_idx][this_sub_idx] = full_weights
                
                # Handle phase map separately
                freq_mask = mask & (df["data_type"] == "frequency")
                # Use "phase_weight" if present, otherwise use "weight", or 0.0
                phase_weights = np.where(freq_mask, df.apply(lambda row: row.get("phase_weight", row["weight"]), axis=1), 0.0)
                phase_map[exp_idx][this_sub_idx] = phase_weights

        # --- Store Final Maps in protocol_info ---
        protocol["scaled_weight_const_from_exp_sub"] = const_map
        protocol["scaled_weight_series_from_exp_sub"] = series_map
        protocol["scaled_weight_amp_from_exp_sub"] = amp_map
        protocol["scaled_weight_phase_from_exp_sub"] = phase_map
        protocol["scaled_weight_prob_dist_from_exp_sub"] = prob_dist_map
        
        return protocol




