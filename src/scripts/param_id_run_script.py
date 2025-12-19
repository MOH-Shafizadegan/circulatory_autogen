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
from utilities.utility_funcs import gradient_fd_4th, gradient_descent, extract_hessian_from_samples, gradient_fd_6th, gradient_richardson_4pt, gradient_fd_8th, nesterov_optimizer, nesterov_accelerated_gradient, torch_adam_optimizer, gradient_richardson
from petsc4py import PETSc
import numdifftools as nd

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
    # param_id.param_id.set_best_param_vals(np.asarray([1.97068532, 0.09984034, 5.47444089, 0.24829631]))
    # param_id.param_id.set_best_param_vals(param_id.param_id.param_norm_obj.unnormalise(np.asarray([0.3440791, 0.03996025, 0.51448196, 0.1190943])))
    param_id.param_id.set_best_param_vals(param_id.param_id.param_norm_obj.unnormalise(np.asarray([0.36090739, 0.04185665, 0.43036572, 0.10262255])))
    # param_id.param_id.set_best_param_vals(np.asarray([2.14223108, 0.87047693, 3.70784294, 1.34989044]))
    # param_id.param_id.set_best_param_vals(np.asarray([2.15278257, 0.85762446, 3.72747759, 1.35065014]))
    # param_id.param_id.set_best_param_vals(np.asarray([2.01101225, 0.10141726, 3.38774938, 0.1744839]))
    best_param_vals = param_id.get_best_param_vals()

    # plot_cost_vs_params(param_id.param_id, num_points=50, param_range_factor=0.004)

    from scipy.optimize import minimize

    # fun = lambda p: -1 * param_id.param_id.get_lnlikelihood_lnprior_from_params(param_id.param_id.param_norm_obj.unnormalise(p))
    # gradient_at_best = nd.Gradient(param_id.param_id.get_lnlikelihood_lnprior_from_params)(best_param_vals)
    # print(f'nd Gradient at best parameters: {gradient_at_best}')
    # gradient_at_best = gradient_fd_4th(param_id.param_id.get_lnlikelihood_lnprior_from_params, best_param_vals, param_norm_obj = param_id.param_id.param_norm_obj, eps=1e-4)
    # print(f'fd 4 point Gradient at best parameters: {gradient_at_best}')
    # gradient_at_best = gradient_fd_4th(fun, param_id.param_id.param_norm_obj.normalise(best_param_vals), eps=1e-4, param_norm_obj = param_id.param_id.param_norm_obj, is_normalised=True)
    # print(f'normalised fd 4 point Gradient at best parameters: {gradient_at_best}')
    # gradient_at_best = gradient_fd_8th(param_id.param_id.get_lnlikelihood_lnprior_from_params, best_param_vals, param_norm_obj = param_id.param_id.param_norm_obj, eps=1e-4)
    # print(f'fd 8 point Gradient at best parameters: {gradient_at_best}')
    # gradient_at_best = gradient_richardson_4pt(param_id.param_id.get_lnlikelihood_lnprior_from_params, best_param_vals, param_norm_obj = param_id.param_id.param_norm_obj, eps=1e-4)
    # print(f'fd 4 richardson point Gradient at best parameters: {gradient_at_best}')
    
    if True: # gradient_at_best.any()>1e-3:
        print('Warning: best parameters do not appear to be optimal, gradient is not close to zero.')
        check = "y"
        if check.lower() == 'y':
            print('running convex optimization...')

            # bounds = [(min_val, max_val) for min_val, max_val in zip(param_id.param_id_info["param_mins"], param_id.param_id_info["param_maxs"])]
            # bounds = [(p * 0.9, p * 1.1) for p in param_id.param_id.param_norm_obj.normalise(best_param_vals)]
            # results_convex = minimize(lambda p: -1 * param_id.param_id.get_lnlikelihood_lnprior_from_params(param_id.param_id.param_norm_obj.unnormalise(p)), param_id.param_id.param_norm_obj.normalise(best_param_vals), 
            #                           method='BFGS', options={'disp': True, 'gtol': 1e-5})
            # results_convex = minimize(lambda p: -1 * param_id.param_id.get_lnlikelihood_lnprior_from_params(param_id.param_id.param_norm_obj.unnormalise(p)), 
            #                           param_id.param_id.param_norm_obj.normalise(best_param_vals), method='L-BFGS-B', bounds=bounds,
            #                           options={'disp': True, 'gtol': 1e-9, 'ftol': 1e-9})
            
            # breakpoint()
            fun = lambda p: -1 * param_id.param_id.get_lnlikelihood_lnprior_from_params(param_id.param_id.param_norm_obj.unnormalise(p))
            # x, convex_opt_hist = gradient_descent(fun, param_id.param_id.param_norm_obj.normalise(best_param_vals), lambda x, eps=1e-3, param_norm_obj=param_id.param_id.param_norm_obj: 
            #                                       gradient_fd_6th(fun, param_id.param_id.param_norm_obj.unnormalise(x), eps=eps, param_norm_obj=param_norm_obj), backtracking=True)
            # [1e-2, 1e-4, 2e-2, 1e-2]
            # x, convex_opt_hist = torch_adam_optimizer(fun, param_id.param_id.param_norm_obj.normalise(best_param_vals), lambda x, eps=1e-4, param_norm_obj=param_id.param_id.param_norm_obj: 
            #                                       gradient_richardson_4pt(fun, param_id.param_id.param_norm_obj.unnormalise(x), eps=eps, param_norm_obj=param_norm_obj))
            # x, convex_opt_hist = torch_adam_optimizer(fun, param_id.param_id.param_norm_obj.normalise(best_param_vals), nd.Gradient)
            
            # x = nesterov_optimizer(fun, param_id.param_id.param_norm_obj.normalise(best_param_vals), lr=1e-5)
            # x_norm_opt, _ = petsc_optimize(fun, best_param_vals, param_id.param_id.param_norm_obj)
            # best_param_vals = param_id.param_id.param_norm_obj.unnormalise(x_norm_opt)

            # if rank == 0:
            #     param_id.param_id.set_best_param_vals(best_param_vals)



            # best_param_vals = param_id.param_id.param_norm_obj.unnormalise(results_convex.x)
            # best_param_vals = param_id.param_id.param_norm_obj.unnormalise(x)
            # param_id.param_id.set_best_param_vals(best_param_vals)
            # print(f'new best parameters: {best_param_vals}')
            samples, losses = latin_hypercube_sample_and_evaluate(param_id.param_id.get_lnlikelihood_lnprior_from_params, best_param_vals, radius=0.001, n_samples=30)
            extract_hessian_from_samples(samples, losses)
            # plot_cost_vs_params(param_id.param_id, num_points=50, param_range_factor=0.0001)
            # gradient_at_best = gradient_fd_4th(param_id.param_id.get_lnlikelihood_lnprior_from_params, best_param_vals, param_norm_obj = param_id.param_id.param_norm_obj)
            # print(f'Gradient at new best parameters: {gradient_at_best}')
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
        # id_analysis.set_fd_step(convex_opt_hist["step"][-1])
        # id_analysis.set_fd_step(1e-3)
        id_analysis.run(inp_data_dict['ia_options'])
    
    if rank == 0:
        print('param id complete')
        
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import qmc

def plot_cost_vs_params(param_id, num_points=5, param_range_factor=0.2):
        """
        Plots the cost function with respect to each parameter, holding others at best-fit values.
        Each subplot shows cost vs. one parameter.
        Args:
            num_points (int): Number of points to evaluate per parameter.
            param_range_factor (float): Fraction of parameter range to explore around best-fit.
        """
        if param_id.best_param_vals is None:
            print("Best parameter values not set. Run parameter identification first.")
            return

        n_params = len(param_id.best_param_vals)
        fig, axs = plt.subplots(int(np.ceil(n_params/3)), 3, figsize=(15, 4 * int(np.ceil(n_params/3))))
        axs = axs.flatten()

        for i in range(n_params):
            best_vals = param_id.best_param_vals.copy()
            param_min = param_id.param_id_info["param_mins"][i]
            param_max = param_id.param_id_info["param_maxs"][i]
            best_val = best_vals[i]
            # Explore a window around the best value
            # window = param_range_factor * (param_max - param_min)
            # scan_min = max(param_min, best_val - window)
            # scan_max = min(param_max, best_val + window)
            scan_min = best_val - param_range_factor * best_val
            scan_max = best_val + param_range_factor * best_val
            param_values = np.linspace(scan_min, scan_max, num_points)
            costs = []
            for val in param_values:
                test_vals = best_vals.copy()
                test_vals[i] = val
                cost = param_id.get_lnlikelihood_lnprior_from_params(test_vals)
                costs.append(cost)
                print(f"Evaluating cost for {param_id.param_id_info['param_names_for_plotting'][i]} = {val}  -> cost = {cost}")
            axs[i].plot(param_values, costs, marker='o')
            axs[i].set_xlabel(param_id.param_id_info["param_names_for_plotting"][i])
            axs[i].set_ylabel("Cost")
            axs[i].set_title(f"Cost vs {param_id.param_id_info['param_names_for_plotting'][i]}")
            axs[i].grid(True)

            # Plot the best value as a star
            axs[i].plot(best_val, param_id.get_lnlikelihood_lnprior_from_params(best_vals), marker='*', color='red', markersize=15, label='Best Value')
            axs[i].legend()

        # Hide unused subplots
        for j in range(n_params, len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.savefig(os.path.join(param_id.output_dir, "cost_vs_params.png"))
        plt.close(fig)
        print(f"Saved cost vs parameter plots to {param_id.output_dir}/cost_vs_params.png")    

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    try:
        run_param_id()
        MPI.Finalize()
    except:
        print(traceback.format_exc())
        comm.Abort()
        exit()

def petsc_optimize(fun, initial_vals, param_norm_obj):
    # 2. Initialize Parallel Vector (X)
    x0_norm = param_norm_obj.normalise(initial_vals)
    n_global = len(x0_norm)
    
    comm = MPI.COMM_WORLD
    x = PETSc.Vec().createMPI(size=n_global, comm=comm)
    # Set the initial guess locally (only on appropriate ranks)
    low, high = x.getOwnershipRange()
    x.setArray(x0_norm[low:high])

    # 3. Define the Objective + Gradient Callback
    def form_function_gradient(tao, x_vec, g_vec):
        """
        PETSc callback: Updates gradient g_vec and returns objective f.
        """
        # Convert PETSc Vec back to NumPy for your unnormalise/fun calls
        # (This is local to each rank)
        scatter, x_seq = PETSc.Scatter().toAll(x_vec)
        scatter.begin(x_vec, x_seq, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)
        scatter.end(x_vec, x_seq, PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)

        with x_seq as x_full:
            # This will now work because shapes match (4,) and (4,)
            f_val = fun(x_full)
            
            # Calculate gradient (this returns a full (4,) array)
            grad_full = gradient_fd_6th(fun, param_norm_obj.unnormalise(x_full), 
                                        eps=1e-3, param_norm_obj=param_norm_obj)

        # x_local = x_vec.getArray(readonly=True)

        # Parallel Evaluation of f and g
        # Note: If 'fun' is not internally parallel, you may need a 
        # wrapper that handles MPI gather/scatter.
        # f = fun(x_local)

        # Compute gradient (using your existing 6th order finite difference)
        # Gradient must be filled into the PETSc g_vec
        # grad_val = gradient_fd_6th(fun, x_local, 
        #                           eps=1e-3, param_norm_obj=param_norm_obj)
        # g_vec.setArray(grad_val)

        low, high = g_vec.getOwnershipRange()
        g_vec.setArray(grad_full[low:high])
        
        return f_val

    # 4. Create TAO Solver (The Parallel Optimizer)
    tao = PETSc.TAO().create(comm=comm)
    
    # xl = PETSc.Vec().createMPI(size=n_global, comm=comm)
    # xu = PETSc.Vec().createMPI(size=n_global, comm=comm)

    # # 2. Set the bound values (use your param_norm_obj to normalise them first)
    # # Example: all parameters bounded between 0 and 1
    # xl.set(0.0)
    # xu.set(1.0)

    # # 3. Inform TAO of the bounds
    # tao.setVariableBounds(xl, xu)

    # 4. Use a solver that supports bounds
    # Standard 'cg' and 'lmvm' work with bounds in modern PETSc
    # tao.setType('bncg') # Bound-constrained Nonlinear Conjugate Gradient
    
    # 'cg' (Conjugate Gradient) is the high-perf parallel version of momentum
    # 'lmvm' is a more advanced quasi-Newton version
    tao.setType('cg') 
    
    # Set the callback and the solution vector
    tao.setObjectiveGradient(form_function_gradient, None)
    tao.setSolution(x)

    # 5. Set Solver Options (Equivalent to Learning Rate/Momentum)
    # These can also be passed via command line, e.g., -tao_monitor
    tao.setTolerances(gatol=1e-5, grtol=1e-5)
    tao.setFromOptions()

    tao.setMonitor(my_cluster_monitor)
    
    # 6. Solve
    tao.solve(x)
    
    # Get optimized results
    x_opt_norm = x.getArray()
    
    # Cleanup
    tao.destroy()
    x.destroy()
    
    return x_opt_norm, [] # history can be retrieved via tao.getMonitor if needed

def my_cluster_monitor(tao):
    # 1. Extract basic status from the tao context
    # getSolutionStatus() returns: (iterations, f, gnorm, cnorm, xdiff, reason)
    its, f_val, g_norm, c_norm, x_diff, reason = tao.getSolutionStatus()
    
    # 2. Safely extract the Step Direction (Velocity)
    # Direction is managed by the internal LineSearch object
    ls = tao.getLineSearch()
    v_norm = 0.0
    if ls:
        try:
            # Retrieve the vector currently used as the step direction
            v_vec = ls.getStepDirection()
            if v_vec:
                v_norm = v_vec.norm()
        except Exception:
            # If the direction is not yet allocated (Iter 0), default to 0.0
            pass

    # 3. Print only on Rank 0 for parallel efficiency
    if PETSc.COMM_WORLD.getRank() == 0:
        print(f"Iter: {its:4d} | Cost: {f_val:.6e} | "
              f"Grad: {g_norm:.6e} | Step: {v_norm:.6e}")

            
def latin_hypercube_sample_and_evaluate(fun, center, radius, n_samples, param_norm_obj=None):
    """
    Generate Latin Hypercube samples around a center point, run the function, and return samples and results.
    The range for each parameter is set as:
        scan_min = center[i] - radius * center[i]
        scan_max = center[i] + radius * center[i]
    Args:
        fun: Callable that takes a parameter vector and returns a scalar result.
        center (np.ndarray): Center point for sampling.
        radius (float): Fractional range for each parameter (param_range_factor).
        n_samples (int): Number of samples.
        param_norm_obj (optional): If provided, used to clip samples to parameter bounds.
    Returns:
        samples (np.ndarray), results (np.ndarray)
    """
    center = np.asarray(center)
    n_params = center.shape[0]
    scan_mins = center - radius * center
    scan_maxs = center + radius * center

    sampler = qmc.LatinHypercube(d=n_params)
    lhs_unit = sampler.random(n=n_samples)
    samples = scan_mins + lhs_unit * (scan_maxs - scan_mins)

    # Optionally clip to parameter bounds if param_norm_obj is present
    if param_norm_obj is not None:
        param_mins = np.asarray(param_norm_obj.unnormalise(np.zeros(n_params)))
        param_maxs = np.asarray(param_norm_obj.unnormalise(np.ones(n_params)))
        samples = np.clip(samples, param_mins, param_maxs)
    results = []
    for i, p in enumerate(samples):
        res = fun(p)
        print(f"Sample {i+1}/{n_samples}: params={p}, result={res}")
        results.append(res)
    results = np.array(results)
    return samples, results