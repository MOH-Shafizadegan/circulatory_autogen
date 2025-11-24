# Identifiability Analysis

Identifiability Analysis (IA) ensures that identified parameters can be trusted, i.e. have a small uncertainty 

## The Laplace Approximation

The Laplace approximation makes an approximation of your parameter posterior distribution, assuming it is gaussian. This uses the Hessian of the log-likelihood with respect to the parameters. 

## IA in Circulatory_Autogen

IA is run following a parameter identification run.

### Configuration for `user_inputs.yaml`

To run IA, you need to set 
```
do_ia: True
```

and add a specific `ia_options` block to your `user_inputs.yaml` configuration file:
```
ia_options: 
    method: 'Laplace' 
```
Currently, the available options for the `method` are **`'Laplace'`**. 

