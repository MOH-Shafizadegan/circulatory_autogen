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
            print('_________Running all param_id tests_____________')
            print('')

        # print('running 3compartment autogeneration test')
        # inp_data_dict['file_prefix'] = '3compartment'
        # inp_data_dict['input_param_file'] = '3compartment_parameters.csv'


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
            print('running test for obs_data file creation with NKE pump model')
        inp_data_dict['file_prefix'] = 'NKE_pump'
        inp_data_dict['input_param_file'] = 'NKE_pump_parameters.csv'
        inp_data_dict['model_type'] = 'cellml_only'
        inp_data_dict['solver'] = 'CVODE'
        inp_data_dict['DEBUG'] = True
        inp_data_dict['pre_time'] = 1
        inp_data_dict['sim_time'] = 100
        inp_data_dict['solver_info'] = {}
        inp_data_dict['solver_info']['MaximumStep'] = 0.001
        inp_data_dict['solver_info']['MaximumNumberOfSteps'] = 5000
        inp_data_dict['do_mcmc'] = False
        inp_data_dict['dt'] = 0.01
        inp_data_dict['param_id_method'] = 'genetic_algorithm'
        inp_data_dict['plot_predictions'] = True
        
        # delete the obs_data file if it exists
        obs_data_path = os.path.join(root_dir_path, 'resources/NKE_pump_obs_data.json')
        inp_data_dict['param_id_obs_path'] = os.path.join(root_dir_path, 'resources/NKE_pump_obs_data.json')

        # generate the obs_data file
        if rank == 0:
            if os.path.exists(obs_data_path):
                os.remove(obs_data_path)
                print('removed existing obs_data file at', obs_data_path)
            # generate obs file
            example_format_obs_data_json_file()
            # generate the model
            generate_with_new_architecture(False, inp_data_dict)
        comm.Barrier()
        # now test the param id for the NKE pump model and the generated
        # obs_data file
        run_param_id(inp_data_dict)
        
        if rank == 0:
            print('')
            print('running 3compartment parameter id test')
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
        inp_data_dict['do_ia'] = True
        inp_data_dict['ia_options'] = {
            'method': 'Laplace'
            }
        if rank == 0:
            print('running 3compartment param id')
        run_param_id(inp_data_dict)

        if rank == 0:
            # also test running autogeneration with the fit parameters
            print('running autogeneration with fit parameters for 3compartment model')
            generate_with_new_architecture(True, inp_data_dict)
            # also test plotting
            print('running plotting for 3compartment model')


        plot_param_id(inp_data_dict, generate=False)
        comm.Barrier()
        
        if rank == 0:
            print('')
            print('running simple_physiological parameter id test')
        else:
            print(f'rank {rank} going into simple_phys run')
        

        inp_data_dict['file_prefix'] = 'simple_physiological'
        inp_data_dict['input_param_file'] = 'simple_physiological_parameters.csv'
        inp_data_dict['param_id_method'] = 'genetic_algorithm'
        inp_data_dict['solver'] = 'CVODE'
        inp_data_dict['pre_time'] = 20
        inp_data_dict['sim_time'] = 2
        inp_data_dict['solver_info'] = {}
        inp_data_dict['solver_info']['MaximumStep'] = 0.001
        inp_data_dict['solver_info']['MaximumNumberOfSteps'] = 5000
        inp_data_dict['dt'] = 0.01
        inp_data_dict['DEBUG'] = True
        inp_data_dict['param_id_obs_path'] = os.path.join(root_dir_path,'resources/simple_physiological_obs_data.json')
        inp_data_dict['do_mcmc'] = True
        inp_data_dict['debug_ga_options']['num_calls_to_function'] = 60
        inp_data_dict['plot_predictions'] = True
        inp_data_dict['do_ia'] = False
        inp_data_dict['ia_options'] = {
            'method': 'Laplace'
            }
        run_param_id(inp_data_dict)

        if rank == 0:
            # also test running autogeneration with the fit parameters
            generate_with_new_architecture(True, inp_data_dict)

        # also test plotting. generate=False because running autogeneration above to test it.
        plot_param_id(inp_data_dict, generate=False)
            
        if rank == 0:
            print('')
            print('running test_fft parameter id test')
        
        comm.Barrier()
        
        inp_data_dict['file_prefix'] = 'test_fft'
        inp_data_dict['input_param_file'] = 'test_fft_parameters.csv'
        inp_data_dict['param_id_method'] = 'genetic_algorithm'
        inp_data_dict['solver'] = 'CVODE'
        inp_data_dict['pre_time'] = 1
        inp_data_dict['sim_time'] = 1
        inp_data_dict['solver_info'] = {}
        inp_data_dict['solver_info']['MaximumStep'] = 0.001
        inp_data_dict['solver_info']['MaximumNumberOfSteps'] = 5000
        inp_data_dict['dt'] = 0.01
        inp_data_dict['DEBUG'] = True
        inp_data_dict['param_id_obs_path'] = os.path.join(root_dir_path,'resources/test_fft_obs_data.json')
        inp_data_dict['param_id_output_dir'] = os.path.join(root_dir_path, 'param_id_output/')   
        inp_data_dict['do_mcmc'] = False
        inp_data_dict['debug_ga_options']['num_calls_to_function'] = 60
        inp_data_dict['plot_predictions'] = False
        run_param_id(inp_data_dict)

        # also test plotting
        plot_param_id(inp_data_dict, generate=True)

        if rank == 0:

            # check that the cost is zero for the test_fft
            fft_cost = np.load(os.path.join(inp_data_dict['param_id_output_dir'], 'genetic_algorithm_test_fft_test_fft_obs_data', 'best_cost.npy'))
            if fft_cost < 1e-10:
                print('fft cost is zero as expected. Success!')
            else:
                print('fft cost is not zero. Failure in the Frequency parameter identitication!')
                raise ValueError('fft cost is not zero. Failure!')

            print('')
            print('running SN_simple parameter id test')
        
        comm.Barrier()
        
        inp_data_dict['file_prefix'] = 'SN_simple'
        inp_data_dict['input_param_file'] = 'SN_simple_parameters.csv'
        inp_data_dict['param_id_method'] = 'genetic_algorithm'
        inp_data_dict['solver'] = 'CVODE'
        inp_data_dict['pre_time'] = 999 # this gets overwritten by the obs_data.json file
        inp_data_dict['sim_time'] = 999 # this gets overwritten by the obs_data.json file
        inp_data_dict['solver_info'] = {}
        inp_data_dict['solver_info']['MaximumStep'] = 0.001
        inp_data_dict['solver_info']['MaximumNumberOfSteps'] = 5000
        inp_data_dict['dt'] = 0.0001
        inp_data_dict['DEBUG'] = True
        inp_data_dict['param_id_obs_path'] = os.path.join(root_dir_path,'resources/SN_simple_obs_data.json')
        inp_data_dict['do_mcmc'] = True
        inp_data_dict['debug_ga_options']['num_calls_to_function'] = 30
        inp_data_dict['plot_predictions'] = True
        run_param_id(inp_data_dict)

        # also test plotting
        plot_param_id(inp_data_dict, generate=True)
        # TODO More tests here. 
        # Add the lung_ROM to test the frequency domain fitting

        print('param ID tests complete. TODO add more param id tests to test',
              'all functionality')
        MPI.Finalize()

    except:
        print(traceback.format_exc())
        exit()
