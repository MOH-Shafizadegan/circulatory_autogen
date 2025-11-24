# Sensitivity Analysis

Sensitivity Analysis (SA) is a recommended first step in the calibration pipeline because it performs crucial **parameter screening**. Calibration is the process of tuning numerous model parameters (inputs) until the model's output matches experimental data. Without SA, a modeler might waste weeks tuning parameters that have virtually no impact on the final result, or unknowingly tune parameters that are highly correlated, leading to non-unique solutions. SA efficiently identifies:

* **Influential parameters:** Which parameters contribute most significantly to the model's output variance.

* **Non-influential parameters:** Which parameters can be fixed or ignored, drastically simplifying the calibration space.

SA ensures that the calibration effort is focused, efficient, and physically meaningful.

## The Sobol Method

The Sobol method is a powerful, **global, variance-based** sensitivity analysis technique. Unlike local methods that only test parameter changes one at a time, the Sobol method explores the entire input space simultaneously.

* **Key Feature:** It quantifies the contribution of each individual input parameter (First-Order Index, S_i) and the contribution of parameter interactions (Second-Order, Total-Order Index, S_{Ti}) to the overall output variance.

This comprehensive approach allows modelers to understand not only which parameters are important on their own, but also how complex, synergistic interactions between two or more parameters drive the final simulation results.

## SA in Circulatory_Autogen
Since Sensitivity Analysis (SA) is intertwined with parameter identification, you will need the same input files as required for parameter identification. This includes both the **`parameter_for_id.csv`** and the **`obs_data.json`** files. However, the exact values of the data terms in the observation file are not critical for SA itself, as you are simply exploring parameter space and variance, not matching the simulation output to observed data.

Crucially, each data item defined in your `obs_data.json` file is treated as a feature for SA, meaning you will receive a separate set of plots (one for first and total order indices, and one for second order indices) for **each** data item.

### Configuration for `user_inputs.yaml`

To run the Sobol analysis, you need to add a specific `sa_options` block to your `user_inputs.yaml` configuration file:
```
sa_options: 
    method: 'sobol' 
    num_samples: 1024 
    sample_type: saltelli
    output_dir: <SA_outputs_path>
```
Currently, the available options for the `method` are **`'naive'`** and **`'sobol'`**. Available sample type are [**'saltelli'**]. What we call `num_samples` here is actually the `num_samples` in

`actual_num_samples = num_samples (2M+2)`

where M is the number of parameters. This means the `num_samples` that you set doesn't need to be dependent on M.

An indicator that the **sample size may be too low** is the observation of **relatively large negative values for the Sobol indices** in the results; if this occurs, you should increase the sample size and re-run the analysis.

## How to run SA script

First, ensure you have the required sensitivity analysis packages specified in see [getting started](getting-started.md) 

To run the script, use the following command (which utilizes **MPI for parallelized computation** on CPU):

```
./run_sensitivity_analysis <NUM_CORES>
```

After successful execution, you will find the SA plots—including the first, second, and total order indices—in the directory specified by `output_dir`.
