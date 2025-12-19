import numpy as np
import math
import os,sys
import libcellml
import utilities.libcellml_helper_funcs as cellml
import utilities.libcellml_utilities as libcellml_utils
import numdifftools as nd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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
        # calculate hessian of the lnlikelihood with finite differences
        breakpoint()
        # hessian = hessian_fd_high_order(param_id.get_lnlikelihood_lnprior_from_params, best_params, eps=epsilon, param_norm_obj=param_id.param_norm_obj)
        hessian = nd.Hessian(param_id.get_lnlikelihood_lnprior_from_params)(best_params)

        # print(hessian)
        # print(hessian_nd)
        
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

def hessian_fd_high_order(f, theta, eps=1e-3, param_norm_obj=None):
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    H = np.zeros((n, n))

    if param_norm_obj:
        theta_norm = param_norm_obj.normalise(theta)
        h = eps * np.maximum(np.abs(theta_norm), 1.0)
        x = theta_norm
        def feval(z):
            return f(param_norm_obj.unnormalise(z))
    else:
        h = eps * np.maximum(np.abs(theta), 1.0)
        x = theta
        feval = f

    fx = feval(x)

    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]

        # ---- Diagonal: 4th-order stencil ----
        fpp = feval(x + 2*ei)
        fp  = feval(x + ei)
        fm  = feval(x - ei)
        fmm = feval(x - 2*ei)

        H[i, i] = (-fpp + 16*fp - 30*fx + 16*fm - fmm) / (12*h[i]**2)

        # ---- Off-diagonal terms ----
        for j in range(i+1, n):
            ej = np.zeros(n)
            ej[j] = h[j]

            # Standard mixed stencil
            f1 = feval(x + ei + ej)
            f2 = feval(x + ei - ej)
            f3 = feval(x - ei + ej)
            f4 = feval(x - ei - ej)

            H_ij_1 = (f1 - f2 - f3 + f4) / (4*h[i]*h[j])

            # Wider stencil
            f1 = feval(x + 2*ei + 2*ej)
            f2 = feval(x + 2*ei - 2*ej)
            f3 = feval(x - 2*ei + 2*ej)
            f4 = feval(x - 2*ei - 2*ej)

            H_ij_2 = (f1 - f2 - f3 + f4) / (16*h[i]*h[j])

            # Average for robustness
            H[i, j] = H[j, i] = 0.5 * (H_ij_1 + H_ij_2)

    # ---- Transform back if parameters were normalised ----
    if param_norm_obj:
        J = param_norm_obj.get_jacobian()
        H = J.T @ H @ J

    return H

def gradient_fd(f, theta, eps=1e-6, param_norm_obj=None):
    
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    g = np.zeros(n)

    # --- Setup ---
    if param_norm_obj:
        theta_norm = param_norm_obj.normalise(theta)
        h = eps * np.minimum(np.abs(theta_norm), 1.0)
        print(theta)
        print(theta_norm)
        print(h)
    else:
        h = eps * np.minimum(np.abs(theta), 1.0)

    # --- Loop ---
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]

        if param_norm_obj:
            f_plus  = f(param_norm_obj.unnormalise(theta_norm + ei))
            f_minus = f(param_norm_obj.unnormalise(theta_norm - ei))
        else:
            f_plus  = f(theta + ei)
            f_minus = f(theta - ei)

        g[i] = (f_plus - f_minus) / (2.0 * h[i])
        
    # --- Chain rule ---
    if param_norm_obj:
        J = param_norm_obj.get_jacobian()
        return J.T @ g

    return g

def gradient_fd_4th(f, theta, eps=1e-3, param_norm_obj=None, is_normalised=False):
    # print(eps)
    # print(theta)
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    g = np.zeros(n)

    # --- Setup ---
    if not is_normalised:
        # print('using param norm obj')
        theta_norm = param_norm_obj.normalise(theta)
        h = eps * np.maximum(np.abs(theta_norm), 1.0)
        unnorm = param_norm_obj.unnormalise
        # unnorm = lambda x: x
    else:
        print("here")
        theta_norm = theta
        h = eps * np.maximum(np.abs(theta), 1.0)
        unnorm = lambda x: x

    # --- Loop ---
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]

        f_p2 = f(unnorm(theta_norm + 2*ei))
        f_p1 = f(unnorm(theta_norm + ei))
        f_m1 = f(unnorm(theta_norm - ei))
        f_m2 = f(unnorm(theta_norm - 2*ei))

        # print(f'Theta: [{unnorm(theta_norm + 2*ei)}], [{unnorm(theta_norm + ei)}], [{unnorm(theta_norm - ei)}], [{unnorm(theta_norm - 2*ei)}]')
        # print(f'Theta norm: [{theta_norm + 2*ei}], [{theta_norm + ei}], [{theta_norm - ei}], [{theta_norm - 2*ei}]')
        # print(f'Gradient calc at index {i}: step = {ei[i]}, f_p2={f_p2}, f_p1={f_p1}, f_m1={f_m1}, f_m2={f_m2}')

        g[i] = (-f_p2 + 8*f_p1 - 8*f_m1 + f_m2) / (12*h[i])

    # --- Chain rule ---
    if param_norm_obj or is_normalised:
        J = param_norm_obj.get_jacobian()
        return J.T @ g

    return g

def gradient_fd_6th(f, theta, eps=1e-3, param_norm_obj=None):
    # print(eps)
    # print(theta)
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    g = np.zeros(n)

    # --- Setup ---
    if param_norm_obj:
        # print('using param norm obj')
        theta_norm = param_norm_obj.normalise(theta)
        h = eps * np.maximum(np.abs(theta_norm), 1.0)
        # unnorm = param_norm_obj.unnormalise
        unnorm = lambda x: x
    else:
        theta_norm = theta
        h = eps * np.maximum(np.abs(theta), 1.0)
        unnorm = lambda x: x

    # --- Loop ---
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]

        f_p3 = f(unnorm(theta_norm + 3*ei))
        f_p2 = f(unnorm(theta_norm + 2*ei))
        f_p1 = f(unnorm(theta_norm + ei))
        f_m1 = f(unnorm(theta_norm - ei))
        f_m2 = f(unnorm(theta_norm - 2*ei))
        f_m3 = f(unnorm(theta_norm - 3*ei))

        g[i] = (-f_m3 + 9*f_m2 - 45*f_m1 + 45*f_p1 - 9*f_p2 + f_p3) / (60*h[i])

    # --- Chain rule ---
    if param_norm_obj:
        J = param_norm_obj.get_jacobian()
        return J.T @ g

    return g

def gradient_fd_8th(f, theta, eps=1e-3, param_norm_obj=None, is_normalised=False):
    # print(eps)
    # print(theta)
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    g = np.zeros(n)

    # --- Setup ---
    if not is_normalised:
        # print('using param norm obj')
        theta_norm = param_norm_obj.normalise(theta)
        h = eps * np.maximum(np.abs(theta_norm), 1.0)
        unnorm = param_norm_obj.unnormalise
        # unnorm = lambda x: x
    else:
        theta_norm = theta
        h = eps * np.maximum(np.abs(theta), 1.0)
        unnorm = lambda x: x

    # --- Loop ---
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]

        f_p4 = f(unnorm(theta_norm + 4*ei))
        f_p3 = f(unnorm(theta_norm + 3*ei))
        f_p2 = f(unnorm(theta_norm + 2*ei))
        f_p1 = f(unnorm(theta_norm + ei))
        f_m1 = f(unnorm(theta_norm - ei))
        f_m2 = f(unnorm(theta_norm - 2*ei))
        f_m3 = f(unnorm(theta_norm - 3*ei))
        f_m4 = f(unnorm(theta_norm - 4*ei))

        g[i] = (3*f_p4 - 32*f_m3 + 168*f_m2 - 672*f_m1 + 672*f_p1 - 168*f_p2 + 32*f_p3 + 3*f_p4) / (840*h[i])

    # --- Chain rule ---
    if param_norm_obj or is_normalised:
        J = param_norm_obj.get_jacobian()
        return J.T @ g

    return g

def gradient_richardson(f, theta, eps=1e-3, param_norm_obj=None, max_order=3):
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    
    # 1. Setup normalization
    if param_norm_obj:
        theta_norm = param_norm_obj.normalise(theta)
        h_base = eps * np.maximum(np.abs(theta_norm), 1.0)
        # unnorm = param_norm_obj.unnormalise
        unnorm = lambda x: x
    else:
        theta_norm = theta
        h_base = eps * np.maximum(np.abs(theta), 1.0)
        unnorm = lambda x: x

    g_final = np.zeros(n)

    # 2. Loop over each dimension
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        
        # Table to store approximations: R[order][step_idx]
        # Column 0 stores base central differences at h, h/2, h/4...
        R = np.zeros((max_order, max_order))
        
        for j in range(max_order):
            h = h_base[i] / (2**j)
            # Base 2nd-order central difference
            f_p = f(unnorm(theta_norm + h * ei))
            f_m = f(unnorm(theta_norm - h * ei))
            R[j, 0] = (f_p - f_m) / (2 * h)
            
            # Fill the row with Richardson extrapolates
            for k in range(1, j + 1):
                # Central difference error terms are even (h^2, h^4, etc.)
                # Factor p = 2 * k
                p = 2 * k 
                R[j, k] = (4**k * R[j, k-1] - R[j-1, k-1]) / (4**k - 1)
        
        # The last diagonal element is the highest-order approximation
        g_final[i] = R[max_order-1, max_order-1]

    # 3. Chain rule for normalization
    if param_norm_obj:
        J = param_norm_obj.get_jacobian()
        return J.T @ g_final

    return g_final

def gradient_richardson_4pt(f, theta, eps=1e-3, param_norm_obj=None):
    
    theta = np.asarray(theta, dtype=float)
    n = len(theta)
    
    # --- Normalization Setup ---
    if param_norm_obj:
        theta_norm = param_norm_obj.normalise(theta)
        h_base = eps * np.maximum(np.abs(theta_norm), 1.0)
        unnorm = param_norm_obj.unnormalise
        # unnorm = lambda x: x
    else:
        theta_norm = theta
        h_base = eps * np.maximum(np.abs(theta), 1.0)
        unnorm = lambda x: x

    g_final = np.zeros(n)

    def get_4pt_stencil(curr_theta, h_vec, dim_idx):
        """Standard 4th-order central difference stencil"""
        ei = np.zeros(n)
        ei[dim_idx] = h_vec[dim_idx]
        
        # 4 points: +/- 1h and +/- 2h
        f_p2 = f(unnorm(curr_theta + 2*ei))
        f_p1 = f(unnorm(curr_theta + 1*ei))
        f_m1 = f(unnorm(curr_theta - 1*ei))
        f_m2 = f(unnorm(curr_theta - 2*ei))
        
        # 4th-order coefficients: (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / 12h
        return (-f_p2 + 8*f_p1 - 8*f_m1 + f_m2) / (12 * h_vec[dim_idx])

    for i in range(n):
        # 1. Calculate 4th-order gradient at step h
        G_h = get_4pt_stencil(theta_norm, h_base, i)
        
        # 2. Calculate 4th-order gradient at step h/2
        h_half = h_base / 2.0
        G_h_half = get_4pt_stencil(theta_norm, h_half, i)
        
        # 3. Richardson Extrapolation
        # Since base is O(h^4), the error is eliminated by (2^4 * G_fine - G_coarse) / (2^4 - 1)
        g_final[i] = (16 * G_h_half - G_h) / 15

    # --- Chain rule / Jacobian ---
    if param_norm_obj:
        J = param_norm_obj.get_jacobian()
        return J.T @ g_final

    return g_final

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

def gradient_descent(fun, x0, grad_fun, step0=1e-2, max_iter=500, tol=1e-4, backtracking=True, verbose=True):
    
    x = np.asarray(x0, dtype=float)
    fval = fun(x)

    history = {
        "x": [x.copy()],
        "f": [fval],
        "grad_norm": [],
        "step": []
    }

    for k in range(max_iter):

        # Step size
        alpha = step0

        g = grad_fun(x)
        g_norm = np.linalg.norm(g)

        history["grad_norm"].append(g_norm)

        if g_norm < tol:
            if verbose:
                print(f"[GD] Converged at iter {k}, ||g||={g_norm:.3e}")
            break

        # Descent direction
        p = -g

        if backtracking:
            c = 1e-4
            rho = 0.5

            # Armijo condition
            while True:
                x_new = x + alpha * p
                f_new = fun(x_new)

                if f_new <= fval + c * alpha * np.dot(g, p):
                    break

                alpha *= rho

                if alpha < 1e-12:
                    if verbose:
                        print("[GD] Step size collapsed")
                    
                    # alpha = 1e-6  # take a small but safe step
                    
                    print("[GD] Checking function values near current point:")
                    for delta in [-2, -1, 0, 1, 2]:
                                test_x = x + delta * alpha * p
                                test_f = fun(test_x)
                                print(f"  f(x + {delta} * alpha * p) = f ({test_x}) = {test_f:.10e}")
                            
                    breakpoint()

                    # g_safe = grad_fun(x, eps=1e-4)  # could use finite difference with safe_step here
                    # g = g_safe
                    # p = -g
                    # x_new = x + alpha * p
                    # f_new = fun(x_new) 
                    # break
                    return x, history

        else:

            # Print function values in the vicinity of the current point to check for local minimum
            print("[GD] Checking function values near current point:")
            for delta in [-2, -1, 0, 1, 2]:
                        test_x = x + delta * alpha * p
                        test_f = fun(test_x)
                        print(f"  f(x + {delta} * alpha * p) = f ({test_x}) = {test_f:.6e}")
                    
            breakpoint()

            x_new = x + alpha * p
            f_new = fun(x_new)

        # Update
        x = x_new
        fval = f_new

        history["x"].append(x.copy())
        history["f"].append(fval)
        history["step"].append(alpha)

        if verbose:
            print(
                f"[GD] iter={k:4d}  f={fval:.6e}  "
                f"||g||={g_norm:.3e}  alpha={alpha:.3e}  "
                f"x_new={x_new}"
            )

    return x, history

# def gradient_descent(fun, x0, grad_fun, step0=1e-5, max_iter=500, tol=1e-4, momentum_rate=0.9, verbose=True):
    
#     x = np.asarray(x0, dtype=float)
#     fval = fun(x)
#     # Initialize velocity vector (v) to zero
#     v = np.zeros_like(x)

#     history = {
#         "x": [x.copy()],
#         "f": [fval],
#         "grad_norm": [],
#         "step": []
#     }

#     for k in range(max_iter):

#         # Step size (fixed learning rate is standard for momentum)
#         alpha = step0

#         g = grad_fun(x)
#         g_norm = np.linalg.norm(g)

#         history["grad_norm"].append(g_norm)

#         if g_norm < tol:
#             if verbose:
#                 print(f"[GD] Converged at iter {k}, ||g||={g_norm:.3e}")
#             break

#         # Momentum update rule for velocity
#         v = momentum_rate * v - alpha * g
#         # v = momentum_rate * v + (1-momentum_rate) * g
        
#         # The step direction 'p' is now the new velocity vector
#         p = v

#         # Update position using the velocity
#         x_new = x + v
#         f_new = fun(x_new)

#         # Update state variables
#         x = x_new
#         fval = f_new

#         history["x"].append(x.copy())
#         history["f"].append(fval)
#         history["step"].append(alpha)

#         if verbose:
#             print(
#                 f"[GD] iter={k:4d}  f={fval:.6e}  "
#                 f"||g||={g_norm:.3e}  step={p}  "
#                 f"||v|| ={np.linalg.norm(v):.3e}"
#             )

#     return x, history

def nesterov_accelerated_gradient(fun, x0, grad_fun, step0=1e-5, max_iter=500, tol=1e-4, momentum_rate=0.9, verbose=True):
    x = np.asarray(x0, dtype=float)
    fval = fun(x)
    v = np.zeros_like(x)

    history = {
        "x": [x.copy()],
        "f": [fval],
        "grad_norm": [],
        "step": []
    }

    for k in range(max_iter):
        alpha = step0

        # --- NAG CORE CHANGE START ---
        # 1. Lookahead: Project position using current momentum
        x_lookahead = x + momentum_rate * v
        
        # 2. Evaluate gradient at the lookahead position
        g = grad_fun(x_lookahead)
        # --- NAG CORE CHANGE END ---

        g_norm = np.linalg.norm(g)
        history["grad_norm"].append(g_norm)

        if g_norm < tol:
            if verbose:
                print(f"[NAG] Converged at iter {k}, ||g||={g_norm:.3e}")
            break

        # 3. Update velocity using the lookahead gradient
        v = momentum_rate * v - alpha * g
        
        # 4. Update actual position
        x_new = x + v
        f_new = fun(x_new)

        x = x_new
        fval = f_new

        history["x"].append(x.copy())
        history["f"].append(fval)
        history["step"].append(alpha)

        if verbose:
            print(
                f"[NAG] iter={k:4d}  f={fval:.6e}  "
                f"||g||={g_norm:.3e}  ||v|| ={np.linalg.norm(v):.3e}  "
                f"x_new={x_new}  g = {g}"
            )

    return x, history

def adam_optimizer(fun, x0, grad_fun, step0=1e-5, max_iter=500, tol=1e-4, 
                   beta1=0.9, beta2=0.999, epsilon=1e-8, verbose=True):
    """
    ADAM Optimizer
    :param beta1: Exponential decay rate for the first moment (momentum)
    :param beta2: Exponential decay rate for the second moment (variance)
    :param epsilon: Small constant to prevent division by zero
    """
    x = np.asarray(x0, dtype=float)
    fval = fun(x)
    
    # Initialize first (m) and second (v) moments to zero
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    history = {
        "x": [x.copy()],
        "f": [fval],
        "grad_norm": [],
        "step": []
    }

    for k in range(1, max_iter + 1):
        # 1. Calculate gradient at current position
        g = grad_fun(x)
        g_norm = np.linalg.norm(g)
        history["grad_norm"].append(g_norm)

        if g_norm < tol:
            if verbose:
                print(f"[ADAM] Converged at iter {k}, ||g||={g_norm:.3e}")
            break

        # 2. Update biased first moment estimate
        m = beta1 * m + (1 - beta1) * g
        
        # 3. Update biased second raw moment estimate
        v = beta2 * v + (1 - beta2) * (g**2)

        # 4. Compute bias-corrected first and second moment estimates
        # (Needed because moments are initialized at zero, causing bias toward zero)
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)

        # 5. Update position
        # Each parameter gets its own step size: alpha / (sqrt(v_hat) + epsilon)
        x_new = x - step0 * m_hat / (np.sqrt(v_hat) + epsilon)
        f_new = fun(x_new)

        x = x_new
        fval = f_new

        history["x"].append(x.copy())
        history["f"].append(fval)
        history["step"].append(step0)

        if verbose:
            print(
                f"[ADAM] iter={k:4d}  f={fval:.6e}  "
                f"||g||={g_norm:.3e}  m_norm={np.linalg.norm(m):.3e}  "
                f"x_new={x_new}"
            )

    return x, history

import torch

def torch_adam_optimizer(fun, x0, grad_fun, lr=1e-5, max_iter=500, tol=1e-4, patience=10, min_delta=1e-3, verbose=True):
    # 1. Initialize parameters as a PyTorch tensor with gradient tracking
    x = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    
    # 2. Setup the built-in Adam optimizer
    # Parameters: lr (learning rate), betas (beta1, beta2), eps (epsilon)
    optimizer = torch.optim.Adam([x], lr=lr, betas=(0.9, 0.999), eps=1e-8)

    best_loss = float('inf')
    stagnant_count = 0  # Counter for oscillations/stagnation

    history = {"x": [], "f": []}

    for k in range(max_iter):
        # Clear previous gradients
        optimizer.zero_grad()
        
        # Compute the objective function (loss)
        x_numpy = x.detach().numpy()
        
        # Compute gradients via backpropagation
        f_val = fun(x_numpy)
        g_val = grad_fun(x_numpy) # Use your Finite Difference/PETSc grad here
        
        # 4. Manually put the gradient back into the PyTorch tensor
        x.grad = torch.from_numpy(g_val).float()
        
        # Check for convergence using the gradient norm
        grad_norm = x.grad.norm().item()

        if abs(best_loss - f_val) > min_delta:
            best_loss = f_val
            stagnant_count = 0  # Reset if we make genuine progress
        else:
            stagnant_count += 1 # Increment if oscillating or stuck
            
        if stagnant_count >= patience:
            if verbose:
                print(f"\n[TERMINATE] Loss stopped reducing significantly.")
                print(f"Iter {k}: Oscillating for {patience} steps. Best Loss: {best_loss:.6e}")
            break

        if grad_norm < tol:
            if verbose:
                print(f"[Torch-Adam] Converged at iter {k}, ||g||={grad_norm:.3e}")
            break
        
        # Perform the Adam update step
        optimizer.step()
        
        # Record history
        history["x"].append(x.detach().numpy().copy())
        history["f"].append(f_val)

        if verbose and k % 1 == 0:
            print(f"Iter {k:4d}: f = {f_val:.6e}, ||g|| = {grad_norm:.3e}, x = {x.detach().numpy()}, g = {g_val}, stagnant_count = {stagnant_count}")

    return x.detach().numpy(), history

from scipy.optimize import approx_fprime
import pandas as pd

def nesterov_optimizer(fun, x0, lr=1e-3, momentum=0.9, iterations=200):
    x = np.array(x0, dtype=float)
    v = np.zeros_like(x)
    eps = 1e-4

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
        
        if i % 1 == 0:
            print(
                f"[NAG] iter={i:4d}  f={fval:.6e}  "
                f"||g||={g_norm:.3e}  ||v|| ={v_norm:.3e}  "
                f"x_new={x}  g = {grad}"
            )

        # if v_norm < 0.1*init_v_norm or i >= 0.5*iterations:
        #     samples.append(np.copy(x))
        #     losses.append(fval)

    return x

def extract_hessian_from_samples(samples, losses):
    """
    Fits a 2nd degree polynomial to (samples, losses) 
    and returns the reconstructed Hessian matrix.
    """

    # Save samples and losses to CSV
    df = pd.DataFrame(samples, columns=[f"x{i}" for i in range(samples.shape[1])])
    df["loss"] = losses
    df.to_csv("samples_and_losses.csv", index=False)

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

    breakpoint()
            
    return hessian