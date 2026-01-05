import numpy as np
import math
import os,sys
import libcellml
import utilities.libcellml_helper_funcs as cellml
import utilities.libcellml_utilities as libcellml_utils
import numdifftools as nd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import approx_fprime
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats.qmc as qmc

class Normalise_class:
    def __init__(self, param_mins, param_maxs, mod_first_variables=0, modVal = 1.0):
        self.param_mins = param_mins
        self.param_maxs = param_maxs
        self.mod_first_variables=mod_first_variables
        self.modVal = modVal
        self.range = self.param_maxs - self.param_mins

    def normalise(self, x):
        xDim = len(x.shape)
        if xDim == 1:
            y = (x - self.param_mins)/(self.param_maxs - self.param_mins)
        elif xDim == 2:
            y = (x - self.param_mins.reshape(-1, 1))/(self.param_maxs.reshape(-1, 1) - self.param_mins.reshape(-1, 1))
        elif xDim == 3:
            y = ((x.reshape(x.shape[0], -1) - self.param_mins.reshape(-1, 1)) /
                 (self.param_maxs.reshape(-1, 1) - self.param_mins.reshape(-1, 1))).reshape(x.shape[0], x.shape[1],
                                                                                            x.shape[2])
        else:
            print('normalising not set up for xDim = {}, exiting'.format(xDim))
            exit()

        return y

    def unnormalise(self, x):
        xDim = len(x.shape)
        if xDim == 1:
            y = x * (self.param_maxs - self.param_mins) + self.param_mins
        elif xDim == 2:
            y = x * (self.param_maxs.reshape(-1, 1) - self.param_mins.reshape(-1, 1)) + self.param_mins.reshape(-1, 1)
        elif xDim == 3:
            y = (x.reshape(x.shape[0], -1)*(self.param_maxs.reshape(-1, 1) - self.param_mins.reshape(-1, 1)) +
                 self.param_mins.reshape(-1, 1)).reshape(x.shape[0], x.shape[1], x.shape[2])
        else:
            print('normalising not set up for xDim = {}, exiting'.format(xDim))
            exit()
        return y
    
    def get_jacobian(self):
        """
        Returns the Jacobian matrix J at point p, where J[i, j] = d(theta_i) / d(p_j).
        For Min-Max (linear) scaling, J is a diagonal matrix where J[i, i] = 1/Range_i.
        
        Args:
            p (np.ndarray): The unnormalized parameter vector (not strictly needed here, 
                            but required for general non-linear transformations).
        
        Returns:
            np.ndarray: The n x n Jacobian matrix J.
        """
        n = len(self.param_mins)
        J = np.zeros((n, n))
        
        # Diagonal elements are 1 / Range_i
        np.fill_diagonal(J, 1.0 / self.range)
        
        return J

def obj_to_string(obj, extra='    '):
    return str(obj.__class__) + '\n' + '\n'.join(
        (extra + (str(item) + ' = ' +
                  (obj_to_string(obj.__dict__[item], extra + '    ') if hasattr(obj.__dict__[item], '__dict__') else str(
                      obj.__dict__[item])))
         for item in sorted(obj.__dict__)))

def bin_resample(data, freq_1, freq_ds):

    new_len = len(freq_ds)
    new_data = np.zeros((new_len))
    new_count = 0 
    this_count = 0 
    addup = 0 
    for II in range(0, len(freq_1)):
        
        dist_behind = np.abs(freq_1[II] - freq_ds[new_count])
        dist_infront = np.abs(freq_1[II] - freq_ds[new_count+1])
        if dist_behind < dist_infront:
            addup += data[II]
            this_count += 1
        else:
            if new_count == 0:
                # overwrite with 0th entry of data
                # this ignores some data points directly after 0 frequency
                new_data[0] = data[0]
            else:
                new_data[new_count] = addup / this_count
            addup = data[II]
            this_count = 1 
            new_count += 1

        if new_count == len(freq_ds) - 1:
            # add all remaining data points to this new datapoint and average
            new_data[new_count] = np.sum(data[II+1:]) / len(data[II+1:])
            break

    return new_data

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def change_parameter_values_and_save(cellml_file, parameter_names, parameter_values, output_file):
    """
    Load a CellML model, change initial values of specified variables,
    then serialize and save the updated model.

    Args:
      cellml_file: Path to the .cellml file to modify.
      parameter_names: List of variable names to change.
      parameter_values: Corresponding list of new initial values.
      output_file: Optional; where to write the new model. Overwrites original if None.
    """

    if len(parameter_names) != len(parameter_values):
        raise ValueError("Names and values lists must have equal length.")

    # Parse the model
    # parse the model in non-strict mode to allow non CellML 2.0 models
    model = cellml.parse_model(os.path.join(cellml_file), False)
    # resolve imports, in non-strict mode
    importer = cellml.resolve_imports(model, os.path.dirname(cellml_file), False)
    # need a flattened model for analysing
    flat_model = cellml.flatten_model(model, importer)
    model_string = cellml.print_model(flat_model)

    # Update variables
    for name, new_val in zip(parameter_names, parameter_values):
        module, name = os.path.split(name)
        found = False
        for comp_index in range(flat_model.componentCount()):
            comp = flat_model.component(comp_index)
            if comp.name() == 'parameters':
                name_mod = name + '_' + module
            elif comp.name() == 'parameters_global':
                name_mod = name
                pass
            elif comp.name() == module:
                name_mod = name
                pass
            else:
                continue

            if comp.hasVariable(name_mod):
                var = comp.variable(name_mod)
                if var.initialValue() == '':
                    print(f"Variable '{name_mod}' does not have an initial value in this module, probably defined in another module, such as parameters.")
                else:
                    var.setInitialValue(str(new_val))
                    found = True
            # print(comp.variableCount())
            # print([comp.variable(i).name() for i in range(comp.variableCount())])
        if not found:
            print(f"Parameter '{name}' not found in any component.")

    # Serialize updated model
    printer = libcellml.Printer()
    new_content = printer.printModel(flat_model)

    # Save to file
    target = output_file 
    with open(target, 'w', encoding='utf-8') as f:
        f.write(new_content)

def calculate_hessian(param_id, AD=False, epsilon=1e-7):    
    """
    Calculate the Hessian matrix of the cost function at the best parameter values.

    Args:
      param_id: An instance of the parameter identification class with a get_cost_from_params method and best_param_vals attribute.

    Returns:
      Hessian matrix as a 2D numpy array.
    """
    if param_id.best_param_vals is None:
        raise ValueError("Best parameter values must be set in param_id before calculating Hessian.")
    
    # TODO The below is not correct yet. Finbar to fix.

    best_params = param_id.best_param_vals
    n_params = len(best_params)
    hessian = np.zeros((n_params, n_params))

    if AD:
        # If using automatic differentiation, implement accordingly
        raise NotImplementedError("Automatic differentiation not implemented yet.")

    else:
        # calculate hessian of the lnlikelihood by fitting a quadratic surface
        samples, losses = latin_hypercube_sample_and_evaluate(param_id.get_lnlikelihood_lnprior_from_params, 
                                                              best_params, radius=0.001, n_samples=30)
        hessian = extract_hessian_from_samples(samples, losses, param_id.output_dir)

    return hessian

def hessian_fd(f, theta, eps=2.5e-3, param_norm_obj=None):
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    H = np.zeros((n, n))

    if param_norm_obj:
        theta_norm = param_norm_obj.normalise(theta)
        h = eps * np.minimum(np.abs(theta_norm), 1.0)
    else:
        h = eps * np.minimum(np.abs(theta), 1.0)    
    
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ej = np.zeros(n)
            ei[i] = h[i]; ej[j] = h[j]
            
            if param_norm_obj:
                fpp = f(param_norm_obj.unnormalise(theta_norm + ei + ej))
                fpm = f(param_norm_obj.unnormalise(theta_norm + ei - ej))
                fmp = f(param_norm_obj.unnormalise(theta_norm - ei + ej))
                fmm = f(param_norm_obj.unnormalise(theta_norm - ei - ej))
            else:
                fpp = f(theta + ei + ej)
                fpm = f(theta + ei - ej)
                fmp = f(theta - ei + ej)
                fmm = f(theta - ei - ej)
            
            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
            H[j, i] = H[i, j]

    if param_norm_obj:
        J = param_norm_obj.get_jacobian()          
        # Apply the transformation: H_p[i, j] = H_theta[i, j] * J_ii * J_jj
        H_p = J.T @ H @ J

        return H_p
    
    return H

def hessian_gauss_newton(residual, theta, eps=1e-3):
    """
    Calculate the Gauss-Newton approximation of the Hessian matrix.

    Args:
      residuals: Function that returns residuals given parameters.
      theta: Parameter values at which to compute the Hessian.
      eps: Small perturbation for finite difference.
    Returns:
      Hessian matrix as a 2D numpy array.
    """
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    m = len(residual(theta))
    J = np.zeros((m, n))
    
    # Relative step sizes
    h = eps * np.maximum(np.abs(theta), 1.0)
    
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = h[j]
        
        r_plus = residual(theta + ej)
        r_minus = residual(theta - ej)
        
        J[:, j] = (r_plus - r_minus) / (2 * h[j])
    
    H_gn = J.T @ J
    return H_gn

def nesterov_optimizer(fun, x0, lr=1e-3, eps=1e-4, momentum=0.9, iterations=200, verbose=False):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)

    for i in range(iterations):
        # 1. Look Ahead: Move slightly in the direction of previous velocity
        x_look_ahead = x + (momentum * v)
        
        # 2. Calculate Gradient at the "Look-Ahead" position
        grad = approx_fprime(x_look_ahead, fun, eps)
        g_norm = np.linalg.norm(grad)
        fval = fun(x_look_ahead)

        # 3. Update Velocity: v = (momentum * v) - (lr * grad)
        v = (momentum * v) - (lr * grad)
        v_norm = np.linalg.norm(v)

        if i==0:
            init_v_norm = np.linalg.norm(v)

        # 4. Update Position
        x = x + v
        
        if verbose and i % 10 == 0:
            print(
                f"[NAG] iter={i:4d}  f={fval:.6e}  "
                f"||g||={g_norm:.3e}  ||v|| ={v_norm:.3e}  "
                f"x_new={x}  g = {grad}"
            )

    return x

def extract_hessian_from_samples(samples, losses, plot_dir=None):
    """
    Fits a 2nd degree polynomial to (samples, losses) 
    and returns the reconstructed Hessian matrix.
    """

    # Save samples and losses to CSV
    df = pd.DataFrame(samples, columns=[f"x{i}" for i in range(samples.shape[1])])
    df["loss"] = losses
    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        csv_path = os.path.join(plot_dir, "samples_and_losses.csv")
    else:
        csv_path = "samples_and_losses.csv"
    df.to_csv(csv_path, index=False)

    # 1. Prepare Polynomial Features (degree=2)
    # This generates [1, x1, x2, x1^2, x1*x2, x2^2]
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(samples)
    
    # 2. Fit the linear regression: Loss = Coefs * X_poly
    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, losses)
    coeffs = model.coef_
    
    # 3. Map coefficients back to a Hessian Matrix
    n_params = samples.shape[1]
    hessian = np.zeros((n_params, n_params))
    
    # Get the feature names to map coefficients correctly
    feature_names = poly.get_feature_names_out()
    
    # Map the coefficients back to the Hessian matrix
    for val, name in zip(coeffs, feature_names):
        # Quadratic terms (e.g., 'x0^2')
        if '^2' in name:
            idx = int(name.split('x')[1].split('^')[0])
            hessian[idx, idx] = val * 2  # Factor of 2 from Taylor Expansion
        
        # Interaction terms (e.g., 'x0 x1')
        elif ' ' in name:
            parts = name.split(' ')
            idx1 = int(parts[0].replace('x', ''))
            idx2 = int(parts[1].replace('x', ''))
            hessian[idx1, idx2] = val
            hessian[idx2, idx1] = val # Maintain symmetry
            
    return hessian

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
