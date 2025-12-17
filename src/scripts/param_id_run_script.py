'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
from mpi4py import MPI
root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))
from param_id.paramID import CVS0DParamID
import traceback
import yaml
import numpy as np
from parsers.PrimitiveParsers import YamlFileParser
from identifiabilty_analysis.identifiabilityAnalysis import IdentifiabilityAnalysis
from utilities.utility_funcs import gradient_fd_4th, gradient_descent

def run_param_id(inp_data_dict=None):

    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict, obs_path_needed=True, do_generation_with_fit_parameters=False)

    DEBUG = inp_data_dict['DEBUG']
    model_path = inp_data_dict['model_path']
    model_type = inp_data_dict['model_type']
    param_id_method = inp_data_dict['param_id_method']
    file_prefix = inp_data_dict['file_prefix']
    params_for_id_path = inp_data_dict['params_for_id_path']
    param_id_obs_path = inp_data_dict['param_id_obs_path']
    sim_time = inp_data_dict['sim_time']
    pre_time = inp_data_dict['pre_time']
    solver_info = inp_data_dict['solver_info']
    dt = inp_data_dict['dt']
    ga_options = inp_data_dict['ga_options']
    resources_dir = inp_data_dict['resources_dir']
    param_id_output_dir = inp_data_dict['param_id_output_dir']
    

    if DEBUG:
        print('WARNING: DEBUG IS ON, TURN THIS OFF IF YOU WANT TO DO ANYTHING QUICKLY')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    print(f'starting script for rank = {rank}')

    param_id = CVS0DParamID(model_path, model_type, param_id_method, False, file_prefix,
                            params_for_id_path=params_for_id_path,
                            param_id_obs_path=param_id_obs_path,
                            sim_time=sim_time, pre_time=pre_time,
                            solver_info=solver_info, dt=dt, ga_options=ga_options, DEBUG=DEBUG,
                            param_id_output_dir=param_id_output_dir, resources_dir=resources_dir)

    if rank == 0:
        if os.path.exists(os.path.join(param_id.output_dir, 'param_names_to_remove.csv')):
            os.remove(os.path.join(param_id.output_dir, 'param_names_to_remove.csv'))


    if param_id_method == 'bayesian':
        acq_func = 'PI'  # 'gp_hedge'
        n_initial_points = 5
        random_seed = 1234
        acq_func_kwargs = {'xi': 0.01, 'kappa': 0.1} # these parameters favour exploitation if they are low
                                                            # and exploration if high, see scikit-optimize docs.
                                                            # xi is used when acq_func is “EI” or “PI”,
                                                            # kappa is used when acq_func is "LCB"
                                                            # gp_hedge, chooses the best from "EI", "PI", and "LCB
                                                            # so it needs both xi and kappa
        # TODO this needs to be defined better if we want to keep bayesian optimiser functionality
        if DEBUG:
            num_calls_to_function = inp_data_dict['debug_ga_options']['num_calls_to_function']
        else:
            num_calls_to_function = inp_data_dict['ga_options']['num_calls_to_function']
        param_id.set_bayesian_parameters(num_calls_to_function, n_initial_points, acq_func,  random_seed,
                                            acq_func_kwargs=acq_func_kwargs)
    # param_id.run()
    # param_id.param_id.set_best_param_vals(np.asarray([5.0,         0.2503606,  8.13095765, 0.42525994]))
    # param_id.param_id.set_best_param_vals(np.asarray([1.9159433, 0.0959415, 4.69310993, 0.22715065]))
    # param_id.param_id.set_best_param_vals(np.asarray([1.9376382, 0.09732969, 4.70212316, 0.22340672]))
    # param_id.param_id.set_best_param_vals(np.asarray([1.83649539, 0.09269065, 5.18408127, 0.24448176]))
    # param_id.param_id.set_best_param_vals(np.asarray([3.98464999, 20.95608989, 4.06212272]))
    # param_id.param_id.set_best_param_vals(np.asarray([10.76528131,  2.13884358,  1.15140459,  2.41859019,  3.67832026]))
    # param_id.param_id.set_best_param_vals(param_id.param_id.param_norm_obj.unnormalise(np.asarray([0.38416302, 0.09465154])))
    # param_id.param_id.set_best_param_vals(param_id.param_id.param_norm_obj.unnormalise(np.asarray([0.50756665, 0.49541924])))
    # param_id.param_id.set_best_param_vals(np.asarray([0.30009163, 0.09994633]))
    # param_id.param_id.set_best_param_vals(np.asarray([2.00278165, 4.06986398]))
    param_id.param_id.set_best_param_vals(np.asarray([1.97068532, 0.09984034, 5.47444089, 0.24829631]))
    # param_id.param_id.set_best_param_vals(np.asarray([2.14223108, 0.87047693, 3.70784294, 1.34989044]))
    # param_id.param_id.set_best_param_vals(np.asarray([2.15278257, 0.85762446, 3.72747759, 1.35065014]))
    best_param_vals = param_id.get_best_param_vals()

    from scipy.optimize import minimize

    # gradient_at_best = gradient_fd_4th(param_id.param_id.get_lnlikelihood_lnprior_from_params, best_param_vals, param_norm_obj = param_id.param_id.param_norm_obj)
    # print(f'Gradient at best parameters: {gradient_at_best}')
    if True: # gradient_at_best.any()>1e-3:
        print('Warning: best parameters do not appear to be optimal, gradient is not close to zero.')
        check = "y"
        if check.lower() == 'y':
            print('running convex optimization...')

            # bounds = [(min_val, max_val) for min_val, max_val in zip(param_id.param_id_info["param_mins"], param_id.param_id_info["param_maxs"])]
            # bounds = [(p * 0.9, p * 1.1) for p in param_id.param_id.param_norm_obj.normalise(best_param_vals)]
            # results_convex = minimize(lambda p: -1 * param_id.param_id.get_lnlikelihood_lnprior_from_params(param_id.param_id.param_norm_obj.unnormalise(p)), param_id.param_id.param_norm_obj.normalise(best_param_vals), method='BFGS', 
            #                           options={'disp': True, 'gtol': 1e-5})
            # results_convex = minimize(lambda p: -1 * param_id.param_id.get_lnlikelihood_lnprior_from_params(param_id.param_id.param_norm_obj.unnormalise(p)), 
            #                           param_id.param_id.param_norm_obj.normalise(best_param_vals), method='L-BFGS-B', bounds=bounds,
            #                           options={'disp': True, 'gtol': 1e-9, 'ftol': 1e-9})
            
            # breakpoint()
            fun = lambda p: -1 * param_id.param_id.get_lnlikelihood_lnprior_from_params(param_id.param_id.param_norm_obj.unnormalise(p))
            x, convex_opt_hist = gradient_descent(fun, param_id.param_id.param_norm_obj.normalise(best_param_vals), lambda x, eps=1e-3, param_norm_obj=param_id.param_id.param_norm_obj: 
                                                  gradient_fd_4th(fun, param_id.param_id.param_norm_obj.unnormalise(x), eps=eps, param_norm_obj=param_norm_obj), backtracking=True)
            
            best_param_vals = param_id.param_id.param_norm_obj.unnormalise(x)
            print(f'new best parameters: {best_param_vals}')
            gradient_at_best = gradient_fd_4th(param_id.param_id.get_lnlikelihood_lnprior_from_params, best_param_vals, param_norm_obj = param_id.param_id.param_norm_obj)
            print(f'Gradient at new best parameters: {gradient_at_best}')
        else:
            print('continuing with current best parameters...')
    
    # param_id.close_simulation() comment for identifiability analysis run otherwise the model will be closed before analysis
    do_mcmc = inp_data_dict['do_mcmc']

    if DEBUG:
        mcmc_options = inp_data_dict['debug_mcmc_options']
    else:
        mcmc_options = inp_data_dict['mcmc_options']

    if do_mcmc:
        mcmc = CVS0DParamID(model_path, model_type, param_id_method, True, file_prefix,
                                params_for_id_path=params_for_id_path,
                                param_id_obs_path=param_id_obs_path,
                                sim_time=sim_time, pre_time=pre_time,
                                solver_info=solver_info, dt=dt, mcmc_options=mcmc_options, DEBUG=DEBUG,
                                param_id_output_dir=param_id_output_dir, resources_dir=resources_dir)
        mcmc.set_best_param_vals(best_param_vals)
        # mcmc.set_mcmc_parameters() TODO
        mcmc.run_mcmc()

    if inp_data_dict['do_ia']:
        # id_analysis = IdentifiabilityAnalysis(model_path, model_type, param_id_method, False, file_prefix,
        #                                      params_for_id_path=params_for_id_path,
        #                                      param_id_obs_path=param_id_obs_path,
        #                                      sim_time=sim_time, pre_time=pre_time,
        #                                      solver_info=solver_info, dt=dt, DEBUG=DEBUG,
        #                                      param_id_output_dir=param_id_output_dir, resources_dir=resources_dir,
        #                                      param_id=param_id.param_id) # pass in param_id object so we can use its cost functions
        print('running identifiability analysis')
        id_analysis = IdentifiabilityAnalysis(model_path, model_type, file_prefix, param_id_output_dir=param_id_output_dir,
                                            resources_dir=resources_dir, param_id=param_id.param_id)  # pass in param_id object so we can use its cost functions

        id_analysis.set_best_param_vals(best_param_vals)    
        print('Running identifiability analysis with method:', inp_data_dict['ia_options']['method'])
        #id_analysis.run_identifiability_analysis(inp_data_dict['identifiability_analysis_options'])
        id_analysis.set_fd_step(convex_opt_hist["step"][-1])
        # id_analysis.set_fd_step(1e-3)
        id_analysis.run(inp_data_dict['ia_options'])
    
    if rank == 0:
        print('param id complete')
        

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    try:
        run_param_id()
        MPI.Finalize()
    except:
        print(traceback.format_exc())
        comm.Abort()
        exit()
