import os
import sys
import yaml
import traceback
from mpi4py import MPI

root_dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir_path, 'src'))

user_inputs_dir = os.path.join(root_dir_path, 'user_run_files')
from scripts.sensitivity_analysis_run_script import run_SA

if __name__ == '__main__':
    try:
        with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
            inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_procs = comm.Get_size()
        if rank == 0:
            print('_________Running all sensitivity analysis tests_____________')
            print('')


        if 'user_inputs_path_override' in inp_data_dict.keys():
            del inp_data_dict['user_inputs_path_override']
        if 'resources_dir' in inp_data_dict.keys():
            # remove that entry so it doesnt get passed to the param_id script
            # so the default dirs are used
            del inp_data_dict['resources_dir']
        if 'generated_models_dir' in inp_data_dict.keys():
            del inp_data_dict['generated_models_dir']
        if 'param_id_output_dir' in inp_data_dict.keys():
            del inp_data_dict['param_id_output_dir']
        
        if rank == 0:
            print('')
            print('running 3compartment SA test')
        inp_data_dict['file_prefix'] = '3compartment'
        inp_data_dict['input_param_file'] = '3compartment_parameters.csv'
        inp_data_dict['param_id_method'] = 'genetic_algorithm'
        inp_data_dict['solver'] = 'CVODE'
        inp_data_dict['pre_time'] = 20
        inp_data_dict['sim_time'] = 2
        inp_data_dict['solver_info'] = {}
        inp_data_dict['solver_info']['MaximumStep'] = 0.001
        inp_data_dict['solver_info']['MaximumNumberOfSteps'] = 5000
        inp_data_dict['dt'] = 0.01
        inp_data_dict['DEBUG'] = True
        inp_data_dict['param_id_obs_path'] = os.path.join(root_dir_path,'resources/3compartment_obs_data.json')
        inp_data_dict['do_mcmc'] = True
        inp_data_dict['debug_ga_options']['num_calls_to_function'] = 60
        inp_data_dict['plot_predictions'] = True
        inp_data_dict['model_out_names'] = ['heart/u_lv']
        inp_data_dict['sa_options'] = {
            'method': 'sobol',
            'num_samples': 2,
            'sample_type': 'saltelli',
            'output_dir': os.path.join(root_dir_path, 'sensitivity_outputs/3compartment_SA_results')
        }

        run_SA(inp_data_dict)

    except:
        print(traceback.format_exc())
        exit()
