'''
@author: Finbar J. Argus
'''

import numpy as np
import os
import sys
from sys import exit
from matplotlib.ticker import FuncFormatter
import corner
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities'))
import math as math
import opencor as oc
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import paperPlotSetup
import diagnostics
import utility_funcs
from utility_funcs import calculate_hessian
import traceback
from utility_funcs import Normalise_class
paperPlotSetup.Setup_Plot(3)
from opencor_helper import SimulationHelper
from parsers.PrimitiveParsers import scriptFunctionParser
from mpi4py import MPI
import re
from numpy import genfromtxt
from importlib import import_module
import csv
from datetime import date
# from skopt import gp_minimize, Optimizer
from parsers.PrimitiveParsers import CSVFileParser
import pandas as pd
import json
import math
# from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib/..*" )
from matplotlib.patches import Ellipse
from scipy.stats import norm, multivariate_normal
from matplotlib.ticker import ScalarFormatter

class IdentifiabilityAnalysis():
    """
    Class for doing identifiability analysis on a 0D model
    """
    def __init__(self, model_path, model_type, file_name_prefix, DEBUG=False,
                 param_id_output_dir=None, resources_dir=None, param_id=None):

        self.model_path = model_path
        self.model_type = model_type
        self.file_name_prefix = file_name_prefix
        self.DEBUG = DEBUG
        self.param_id_output_dir = param_id_output_dir
        self.resources_dir = resources_dir
        self.best_param_vals = None
        self.covariance_matrix_Laplace = None
        self.mean_Lapalace = None
        self.param_id = param_id
        self.fd_step = 1e-6
        if self.param_id is None:
            # TODO intialise the param_id_object here
            raise ValueError("param_id object must be provided to IdentifiabilityAnalysis")
        

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

    def set_best_param_vals(self, best_param_vals):
        self.param_id.set_best_param_vals(best_param_vals)
        self.best_param_vals = best_param_vals

    def run(self, ia_options):
        """
        Run the identifiability analysis based on the chosen method
        """
        if ia_options['method'] == 'profile_likelihood':
            self.run_profile_likelihood(ia_options)
        elif ia_options['method'] == 'Laplace':
            if self.rank == 0:
                # currently Laplace is not parallelised, so only run on rank 0
                self.run_laplace_approximation(ia_options)
        return

    def run_profile_likelihood(self, ia_options):
        # TODO
        print("Profile Likelihood method not yet implemented")
        exit()
        pass

    def run_laplace_approximation(self, ia_options):

        # TODO fix hessian calculation now that it uses lnlikelihood + lnprior
        Hessian = calculate_hessian(self.param_id, epsilon=self.fd_step)
        covariance_matrix = np.linalg.inv(-1*Hessian)
        mean = self.best_param_vals
        print("Laplace Approximation Results:")
        print("Mean (Best Parameter Values):", mean)
        print("Covariance Matrix:\n", covariance_matrix)
        self.covariance_matrix_Laplace = covariance_matrix
        self.mean_Lapalace = mean
        parent_dir = os.path.dirname(self.param_id_output_dir)
        np.save(os.path.join(parent_dir, self.file_name_prefix + '_laplace_mean.npy'), self.mean_Lapalace)
        np.save(os.path.join(parent_dir, self.file_name_prefix + '_laplace_covariance.npy'), self.covariance_matrix_Laplace)

    def set_fd_step(self, step):
        self.fd_step = step

    def plot_laplace_results(self, parameter_names, output_dir):
        """
        Plot the results of the Laplace approximation as corner plots.

        Args:
          parameter_names: List of parameter names corresponding to the best_param_vals.
          output_dir: Directory to save the plots.
        """
          

        if self.covariance_matrix_Laplace is None or self.mean_Lapalace is None:
            try:
                parent_dir = os.path.dirname(self.param_id_output_dir)
                self.mean_Lapalace = np.load(os.path.join(parent_dir, self.file_name_prefix + '_laplace_mean.npy'))
                self.covariance_matrix_Laplace = np.load(os.path.join(parent_dir, self.file_name_prefix + '_laplace_covariance.npy'))
                print("Loaded Laplace approximation results from files.")
            except Exception as e:
                print("Error loading Laplace approximation results:", e)
                print("Please run the Laplace approximation before plotting.")
                return

        samples = np.random.multivariate_normal(self.mean_Lapalace, self.covariance_matrix_Laplace, size=100000)
        print(f'samples shape: {samples.shape}')
        figure = corner.corner(samples, labels=parameter_names, truths=self.mean_Lapalace, bins=20, hist_bin_factor=2, smooth=0.5, quantiles=(0.05, 0.5, 0.95))
        plot_path = os.path.join(output_dir, f"{self.file_name_prefix}_laplace_corner_plot.pdf")
        
        axes = figure.get_axes()
        num_params = len(parameter_names)
        # for idx, ax in enumerate(axes):
        #     if idx >= num_params*(num_params - 1):

        #         ax.tick_params(axis='both', rotation=0)
        #         formatterx = matplotlib.ticker.ScalarFormatter()
        #         ax.xaxis.set_major_formatter(formatterx)
        #         ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        #     if idx%num_params == 0:

        #         ax.tick_params(axis='both', rotation=0)
        #         formattery = matplotlib.ticker.ScalarFormatter()
        #         ax.yaxis.set_major_formatter(formattery)
        #         ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    
        # from matplotlib.ticker import FuncFormatter

        # sci_formatter = FuncFormatter(lambda x, _: f"{x:.2e}")

        # for idx, ax in enumerate(axes):
        #     if idx >= num_params * (num_params - 1):
        #         ax.xaxis.set_major_formatter(sci_formatter)
        #     if idx % num_params == 0:
        #         ax.yaxis.set_major_formatter(sci_formatter)

        


        def make_sci_label_formatter(exponent):
            def formatter(val, pos):
                if val == 0:
                    return "0"
                else:
                    # Divide by 10**exponent and format
                    return f"{val / (10 ** exponent):.2f}"
            return FuncFormatter(formatter)

        # We'll store the exponent per axis
        for idx, ax in enumerate(axes):
            if idx >= num_params * (num_params - 1):  # Bottom row → x-axis
                x_min, x_max = ax.get_xlim()
                if x_max == x_min:
                    continue
                # Use log10 of max abs value to determine exponent
                exponent = int(np.floor(np.log10(np.max(np.abs([x_min, x_max])))))
                
                # Apply formatter that divides by 10^exp
                ax.xaxis.set_major_formatter(make_sci_label_formatter(exponent))
                
                # Add ×10^exp label just outside the plot
                ax.text(1.0, 0, fr'$\times 10^{{{exponent}}}$', 
                        transform=ax.transAxes,
                        va='bottom', ha='right',
                        fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            if idx % num_params == 0:  # Left column → y-axis
                y_min, y_max = ax.get_ylim()
                if y_max == y_min:
                    continue
                exponent = int(np.floor(np.log10(np.max(np.abs([y_min, y_max])))))
                
                ax.yaxis.set_major_formatter(make_sci_label_formatter(exponent))
                
                # Add ×10^exp label above the y-axis
                ax.text(0, 1.0, fr'$\times 10^{{{exponent}}}$', 
                        transform=ax.transAxes,
                        va='top', ha='left',
                        rotation=0,
                        fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


        plt.subplots_adjust(hspace=0.12, wspace=0.1)
        figure.savefig(plot_path)
        print(f"Laplace approximation corner plot saved to {plot_path}")

    def plot_laplace_results_analytical(self, parameter_names, output_dir):
        """
        ANALYTIC Laplace approximation:
            - 1D Gaussian PDFs + 95% CI lines
            - 2D Gaussian contours
            - 95% confidence ellipse
        """

        num_params = len(parameter_names)

        # -------------------------------
        # 2. Load Laplace results
        # -------------------------------
        if self.mean_Lapalace is None:
            parent = os.path.dirname(self.param_id_output_dir)
            self.mean_Lapalace = np.load(os.path.join(parent, self.file_name_prefix + "_laplace_mean.npy"))
            self.covariance_matrix_Laplace = np.load(os.path.join(parent, self.file_name_prefix + "_laplace_covariance.npy"))

        mean = self.mean_Lapalace
        cov  = self.covariance_matrix_Laplace
        w, V = np.linalg.eigh(cov)
        w_clipped = np.clip(w, 1e-8, None)
        cov = V @ np.diag(w_clipped) @ V.T
        std = np.sqrt(np.diag(cov))

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
        fig, axes = plt.subplots(num_params, num_params, figsize=(3*num_params, 3*num_params))

        for i in range(num_params):
            for j in range(num_params):
                ax = axes[i, j]

                # -------------------------------------
                # 1D diagonal: analytic Gaussian PDF
                # -------------------------------------
                if i < j:
                    ax.set_axis_off()
                    continue  # Skip all subsequent plotting logic for this subplot

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

                    std_str = f"$\sigma$ = {s:.2g}"
                    # Place the text in the top-left corner of the subplot
                    ax.text(0.05, 0.95, std_str, 
                            transform=ax.transAxes, 
                            fontsize=10, 
                            verticalalignment='top', 
                            horizontalalignment='left',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"))
                    
                    # Ensure PDF plots are not cropped by ticks
                    ax.margins(x=0)

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

        plot_path = os.path.join(output_dir, f"{self.file_name_prefix}_laplace_corner_plot.pdf")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Analytic Laplace results saved: {plot_path}")
