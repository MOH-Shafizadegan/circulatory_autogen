import os
import sys
import yaml
import traceback
import numpy as np
from mpi4py import MPI

root_dir_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir_path, 'src'))

user_inputs_dir = os.path.join(root_dir_path, 'user_run_files')
from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from scripts.param_id_run_script import run_param_id
from scripts.plot_param_id_script import plot_param_id
from scripts.example_format_obs_data_json_file import example_format_obs_data_json_file

if __name__ == '__main__':
    try:
        with open(os.path.join(user_inputs_dir, 'user_inputs.yaml'), 'r') as file:
            inp_data_dict = yaml.load(file, Loader=yaml.FullLoader)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        num_procs = comm.Get_size()
        if rank == 0:
            print('_________Running Benchmark tests_____________')
            print('')

        # generate_with_new_architecture(False, inp_data_dict)
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
            print('running Simple ODE Benchmark parameter id test')
        inp_data_dict['file_prefix'] = 'Simple_ODE_Benchmark'
        inp_data_dict['input_param_file'] = 'Simple_ODE_Benchmark_parameters.csv'
        inp_data_dict['param_id_method'] = 'genetic_algorithm'
        inp_data_dict['solver'] = 'CVODE'
        inp_data_dict['pre_time'] = 0
        inp_data_dict['sim_time'] = 8
        inp_data_dict['solver_info'] = {}
        inp_data_dict['solver_info']['MaximumStep'] = 0.001
        inp_data_dict['solver_info']['MaximumNumberOfSteps'] = 5000
        inp_data_dict['dt'] = 0.01
        inp_data_dict['DEBUG'] = False
        inp_data_dict['param_id_obs_path'] = os.path.join(root_dir_path,'resources/Simple_ODE_Benchmark_obs_data.json')
        inp_data_dict["ga_options"] = {
            "cost_convergence": 0.0001,
        }
        inp_data_dict['do_mcmc'] = True
        inp_data_dict["mcmc_options"] = {
            "num_steps": 100,
            "num_walkers": 64
        }
        inp_data_dict['plot_predictions'] = True
        inp_data_dict['do_ia'] = True
        inp_data_dict['ia_options'] = {
            'method': 'Laplace'
            }
        if rank == 0:
            print('running Simple ODE Benchmark param id')
        run_param_id(inp_data_dict)

        if rank == 0:
            # also test running autogeneration with the fit parameters
            print('running autogeneration with fit parameters for Simple ODE Benchmark model')
            generate_with_new_architecture(True, inp_data_dict)
            # also test plotting
            print('running plotting for Simple ODE Benchmark model')


        plot_param_id(inp_data_dict, generate=False)
        comm.Barrier()
        
        
        print('param ID tests complete. TODO add more param id tests to test',
              'all functionality')
        MPI.Finalize()

    except:
        print(traceback.format_exc())
        exit()
