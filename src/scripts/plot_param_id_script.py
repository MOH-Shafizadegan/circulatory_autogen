'''
Created on 29/10/2021

@author: Finbar J. Argus
'''

import sys
import os
from mpi4py import MPI
from distutils import util
import re

root_dir = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(root_dir, 'src'))

user_inputs_dir = os.path.join(root_dir, 'user_run_files')

from param_id.paramID import CVS0DParamID, MCMC_plotter
from identifiabilty_analysis.identifiabilityAnalysis import IdentifiabilityAnalysis
from scripts.script_generate_with_new_architecture import generate_with_new_architecture
from utilities.utility_funcs import obj_to_string, change_parameter_values_and_save
import traceback
from distutils import util
import yaml
from parsers.PrimitiveParsers import YamlFileParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import corner
from matplotlib.ticker import ScalarFormatter


def plot_param_id(inp_data_dict=None, generate=True):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    
    if rank != 0:
        return

    yaml_parser = YamlFileParser()
    inp_data_dict = yaml_parser.parse_user_inputs_file(inp_data_dict, obs_path_needed=True, do_generation_with_fit_parameters=True)

    DEBUG = inp_data_dict['DEBUG']
    model_path = inp_data_dict['model_path']
    uncalibrated_model_path = inp_data_dict['uncalibrated_model_path']
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
    mcmc_options = inp_data_dict['mcmc_options']
    resources_dir = inp_data_dict['resources_dir']
    param_id_output_dir = inp_data_dict['param_id_output_dir']
    param_id_output_dir_abs_path = inp_data_dict['param_id_output_dir_abs_path']
    plot_predictions = inp_data_dict['plot_predictions']
    do_sensitivity = inp_data_dict['do_sensitivity']
    do_mcmc = inp_data_dict['do_mcmc']
    generated_models_subdir = inp_data_dict['generated_models_subdir']
    do_identify = inp_data_dict['do_ia']
    ia_option = inp_data_dict['ia_options']
    
    # run the generation script with new param values
    if generate:
        generate_with_new_architecture(True, inp_data_dict=inp_data_dict)
    else:
        # if we arent doing the autogeneration then we need to create a model with the best fit parameters anyway
        # load the best parameter values from the param_id_output_dir
       
        best_param_vals = np.load(os.path.join(param_id_output_dir_abs_path, 'best_param_vals.npy'))
        param_names = np.loadtxt(os.path.join(param_id_output_dir_abs_path, 'param_names.csv'), dtype=str)
        
        # change the parameter values in the model file
        change_parameter_values_and_save(uncalibrated_model_path, param_names, best_param_vals, 
                                         model_path)
        # set the model path to the new model file
        print('Best fit model saved to: ' + model_path)
        
        # set the model_path to the uncalibrated model path for now so we can run it. 
        # this will be changed when we can run the new cellml2.0 model from python.
        model_path = uncalibrated_model_path


    
    param_id = CVS0DParamID(model_path, model_type, param_id_method, False, file_prefix,
                            params_for_id_path=params_for_id_path,
                            param_id_obs_path=param_id_obs_path,
                            sim_time=sim_time, pre_time=pre_time,
                            solver_info=solver_info, ga_options=ga_options, dt=dt,
                            param_id_output_dir=param_id_output_dir, resources_dir=resources_dir, one_rank=True)

    if os.path.exists(os.path.join(param_id.output_dir, 'param_names_to_remove.csv')):
        with open(os.path.join(param_id.output_dir, 'param_names_to_remove.csv'), 'r') as r:
            param_names_to_remove = []
            for row in r:
                name_list = row.split(',')
                name_list = [name.strip() for name in name_list]
                param_names_to_remove.append(name_list)
        param_id.remove_params_by_name(param_names_to_remove)
        

    # simulate with best values first to check cost
    param_id.simulate_with_best_param_vals()
    param_id.plot_outputs()
    if do_mcmc:
        if os.path.exists(os.path.join(param_id.output_dir, 'mcmc_chain.npy')):
            if not plot_predictions:
                param_id.plot_mcmc()
        else:
            print('no mcmc chain has been created at ' + 
                    os.path.join(param_id.output_dir, 'mcmc_chain.npy'))
    param_id.save_prediction_data()
    if do_sensitivity:
        param_id.run_sensitivity(None)
    # param_id.close_simulation() # remove this for identifiability analysis

    print(f'doing ia for rank {rank}')
    
    if do_identify:
        id_analysis = IdentifiabilityAnalysis(model_path, model_type, file_prefix, param_id_output_dir=param_id_output_dir,
                                              resources_dir=resources_dir, param_id=param_id.param_id)  # pass in param_id object so we can use its cost functions
        id_analysis.set_best_param_vals(param_id.get_best_param_vals())       
        # id_analysis.run_identifiability_analysis(ia_option) # this should already be done
        label_list =[f'${param_id.param_id_info["param_names_for_plotting"][II]}$' for II in range(len(param_id.param_id_info["param_names_for_plotting"]))]
        print(label_list)
        id_analysis.plot_laplace_results_analytical(label_list, param_id.plot_dir)
        print('Identifiability analysis plotting complete')
    
    param_id.close_simulation()


    if plot_predictions:
        mcmc_plotter = MCMC_plotter(model_path, model_type, param_id_method, file_prefix,
                                            params_for_id_path=params_for_id_path,
                                            param_id_obs_path=param_id_obs_path,
                                            num_calls_to_function=1,
                                            sim_time=sim_time, pre_time=pre_time,
                                            solver_info=solver_info, dt=dt, ga_options=ga_options, 
                                            mcmc_options=mcmc_options, DEBUG=DEBUG,
                                            param_id_output_dir=param_id_output_dir, resources_dir=resources_dir)
    
        if do_mcmc:
            mcmc_plotter.plot_mcmc_and_predictions()
    
        if do_mcmc and do_identify:
            plot_mcmc_and_laplace(param_id, id_analysis)

    return

def plot_mcmc_and_laplace(param_id, id_analysis):
    """
    Same layout as original MCMC corner plot, but overlays
    ANALYTIC Laplace approximation:
        - 1D Gaussian PDFs + 95% CI lines
        - 2D Gaussian contours
        - 95% confidence ellipse
    """

    from matplotlib.patches import Ellipse
    from scipy.stats import norm, multivariate_normal
    import matplotlib

    # -------------------------------
    # 1. Load MCMC samples
    # -------------------------------
    flat_samples, samples, num_params = param_id.get_mcmc_samples()

    label_list = [
        f'${param_id.param_id_info["param_names_for_plotting"][II]}$'
        for II in range(num_params)
    ]
    overwrite_params_to_plot_idxs = list(range(num_params))

    # Truths
    if param_id.mcmc_instead:
        mcmc_object = param_id.mcmc_object
        if mcmc_object.best_param_vals is None:
            best_param_vals = np.load(os.path.join(param_id.output_dir, 'best_param_vals.npy'))
            mcmc_object.set_best_param_vals(best_param_vals)
        truths = mcmc_object.best_param_vals[overwrite_params_to_plot_idxs]

    else:
        if param_id.param_id.best_param_vals is None:
            best_param_vals = np.load(os.path.join(param_id.output_dir, 'best_param_vals.npy'))
            param_id.param_id.set_best_param_vals(best_param_vals)
        truths = param_id.param_id.best_param_vals[overwrite_params_to_plot_idxs]

    # -------------------------------
    # 2. Load Laplace results
    # -------------------------------
    if id_analysis.mean_Lapalace is None:
        parent = os.path.dirname(id_analysis.param_id_output_dir)
        id_analysis.mean_Lapalace = np.load(os.path.join(parent, id_analysis.file_name_prefix + "_laplace_mean.npy"))
        id_analysis.covariance_matrix_Laplace = np.load(os.path.join(parent, id_analysis.file_name_prefix + "_laplace_covariance.npy"))

    mean = id_analysis.mean_Lapalace[overwrite_params_to_plot_idxs]
    cov  = id_analysis.covariance_matrix_Laplace[np.ix_(overwrite_params_to_plot_idxs, overwrite_params_to_plot_idxs)]
    std = np.sqrt(np.diag(cov))

    # Axis limits
    plot_limits = [(mean[i] - 5*std[i], mean[i] + 5*std[i]) for i in range(num_params)]

    # -------------------------------
    # 3. Create standard MCMC corner plot
    # -------------------------------
    fig = corner.corner(
        flat_samples[:, overwrite_params_to_plot_idxs],
        bins=20,
        hist_bin_factor=2,
        smooth=0.5,
        quantiles=(0.05, 0.5, 0.95),
        labels=label_list,
        truths=truths,
        range=plot_limits,
        plot_density=True,
        plot_contours=True,
        data_kwargs={"color": "darkblue", "alpha": 0.10},
        label_kwargs={"fontsize": 18},
        hist_kwargs={"density": True},
        fontsize=20
    )

    axes = np.array(fig.axes).reshape(num_params, num_params)

    # -------------------------------
    # Helper for ellipse
    # -------------------------------
    def ellipse_params(cov2):
        vals, vecs = np.linalg.eigh(cov2)
        # vals[vals < 1e-14] = 1e-14
        chi2 = 5.991 # 95% CI for 2D
        width  = 2*np.sqrt(vals[0]*chi2)
        height = 2*np.sqrt(vals[1]*chi2)
        vx = vecs[:, 0]
        angle = np.degrees(np.arctan2(vx[1], vx[0]))
        print(width, height, angle)
        return width, height, angle

    # -------------------------------
    # 4. Overlay analytic Laplace
    # -------------------------------
    for i in range(num_params):
        for j in range(num_params):
            ax = axes[i, j]

            # -------------------------------------
            # 1D diagonal: analytic Gaussian PDF
            # -------------------------------------
            if i == j:
                mu = mean[i]
                s  = std[i]

                x = np.linspace(mu - 4*s, mu + 4*s, 300)
                y = norm.pdf(x, mu, s)

                # Blue analytic PDF
                ax.plot(x, y, color="blue", linewidth=2)
                ax.fill_between(x, y, color="blue", alpha=0.25)

                # 95% CI
                ci_low  = mu - 1.96*s
                ci_high = mu + 1.96*s
                ax.axvline(ci_low,  color="black", linestyle="--", linewidth=1.2)
                ax.axvline(ci_high, color="black", linestyle="--", linewidth=1.2)

            # -------------------------------------
            # 2D off–diagonal: analytic contours
            # -------------------------------------
            elif i>j:
                mu = [mean[j], mean[j]]
                cov2 = cov[np.ix_([j, i], [j, i])]

                # grid
                xs = np.linspace(mu[0] - 3*np.sqrt(cov2[0,0]),
                                 mu[0] + 3*np.sqrt(cov2[0,0]), 200)
                ys = np.linspace(mu[1] - 3*np.sqrt(cov2[1,1]),
                                 mu[1] + 3*np.sqrt(cov2[1,1]), 200)
                X, Y = np.meshgrid(xs, ys)
                Z = multivariate_normal(mu, cov2).pdf(np.dstack((X, Y)))

                # analytic contours
                ax.contour(X, Y, Z, levels=10, cmap="Blues")

                # 95% ellipse
                w, h, ang = ellipse_params(cov2)
                ell = Ellipse(mu, w, h, angle=ang,
                              facecolor="blue", alpha=0.15, edgecolor="black")
                ax.add_patch(ell)

    # -------------------------------
    # 5. Formatting & Saving
    # -------------------------------
    axes_all = fig.get_axes()
    for idx, ax in enumerate(axes_all):
        if idx >= num_params*(num_params-1):
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        else:
            ax.tick_params(axis='x', labelbottom=False)

        if idx % num_params == 0:
            ax.tick_params(axis='y', labelsize=10)
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        else:
            ax.tick_params(axis='y', labelleft=False)

    plt.subplots_adjust(hspace=0.15, wspace=0.15)

    save_path = os.path.join(
        param_id.plot_dir,
        f"mcmc_laplace_overlay_{param_id.file_name_prefix}_{param_id.param_id_obs_file_prefix}.pdf"
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Analytic Laplace overlay saved: {save_path}")

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    # get argument of whether to run the autogeneration, defaults to True
    if len(sys.argv) == 2:
        generate = util.strtobool(sys.argv[1])
    else:
        generate = True
    try:
        plot_param_id(generate=generate)
    except:
        print(traceback.format_exc())
        comm.Abort()
        exit()