'''
Generating a workflow model -- a library module on its own, or a module array -- with some of
its parameters overridden.
'''

import csv
import json
import os
import shutil

from libcuflynx.calibration_workflow.naming import STEP_VESSEL, host_model_name
from libcuflynx.utilities.config_schemas import copy_supermodule_instance_parameters

PARAMETER_HEADER = ('variable_name', 'units', 'value', 'data_reference')
_MODEL_EXTENSIONS = {'cellml': '.cellml', 'python': '.py', 'casadi_python': '.py',
                     'aadc_python': '.py', 'cpp': '.cpp'}


def write_parameter_rows(path, rows):
    '''Write ``rows`` (dicts with PARAMETER_HEADER keys) as a parameters CSV.'''
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(PARAMETER_HEADER), extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in PARAMETER_HEADER})


def read_parameter_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return [dict(row) for row in csv.DictReader(f)]


def generate_parameterisation(target, work_dir, file_prefix, overrides=(), library_inputs=None,
                              model_type='cellml', solver=None, vessel=STEP_VESSEL,
                              instance_parameters=None):
    '''Generate the module parameterisation ``target`` (a spec.Target) alone, as a single
    module-array record named ``vessel``, under ``work_dir``. A supermodule's record gets
    ``instance_parameters`` (a supermodule parameters file) as its instance parameters file, or a
    new instance's, from its version.

    ``overrides`` are parameter rows (``variable_name`` as a host parameters file names it,
    ``mod__A/p``, ``units``, ``value``, ``data_reference``) written as the host parameters file, which wins
    over every parameterisation and supermodule default. ``library_inputs`` are the generation
    keys that locate the modules (``module_library_dirs`` etc.).

    Returns ``{'model_path', 'flat_model_path', 'parameters_path', 'resources_dir',
    'generated_models_dir', 'inp_data_dict'}`` (``flat_model_path``: the CellML with its
    imports resolved, None for other model types); raises RuntimeError when generation
    fails. Not MPI-aware: call it on one rank.
    '''
    work_dir = os.path.abspath(work_dir)
    resources_dir = os.path.join(work_dir, 'resources')
    os.makedirs(resources_dir, exist_ok=True)
    record = {'name': vessel, 'module_type': target.module_type,
              'module_subtype': target.version, 'parameterisation': target.parameterisation,
              'inp_instances': [], 'out_instances': []}
    array = os.path.join(resources_dir, f'{file_prefix}_module_array.json')
    with open(array, 'w') as f:
        json.dump([record], f, indent=1)
    _write_new_instance(array, vessel, library_inputs, instance_parameters)
    write_parameter_rows(os.path.join(resources_dir, f'{file_prefix}_parameters.csv'),
                         list(overrides))
    return _generate(work_dir, resources_dir, file_prefix, model_type, solver, library_inputs,
                     target.label())


def _write_new_instance(array, vessel, library_inputs, instance_parameters=None):
    '''Give the record ``vessel`` of the one-record module array ``array`` its supermodule
    instance parameters file, when it is a supermodule instance: ``instance_parameters`` if given,
    else a new instance's, from its version (``supermodules.write_instance_parameters``).'''
    from libcuflynx.utilities.config_schemas import (supermodule_instance_parameters_path,
                                                     write_supermodule_parameters_file)
    if instance_parameters is not None:
        write_supermodule_parameters_file(supermodule_instance_parameters_path(array, vessel),
                                          instance_parameters)
        return
    from libcuflynx.utilities.module_library import ModuleSources
    from libcuflynx.utilities.supermodules import write_instance_parameters
    write_instance_parameters(array, ModuleSources(dict(library_inputs or {})).supermodule_registry(),
                              replace=True)


def generate_module_array(model, work_dir, file_prefix, overrides=(), library_inputs=None,
                          model_type='cellml', solver=None):
    '''Generate the module array ``model`` (a spec.ModuleArray) under ``work_dir``, from a copy
    of its module array, its supermodule instances' parameters files and its parameters file.
    ``overrides`` (rows named as a host parameters
    file names them) replace the rows of the parameters file that set the same parameter. Returns as :func:`generate_parameterisation`.'''
    work_dir = os.path.abspath(work_dir)
    resources_dir = os.path.join(work_dir, 'resources')
    os.makedirs(resources_dir, exist_ok=True)
    extension = os.path.splitext(model.module_array)[1].lower() or '.csv'
    array = os.path.join(resources_dir, f'{file_prefix}_module_array{extension}')
    shutil.copyfile(model.module_array, array)
    copy_supermodule_instance_parameters(model.module_array, array)
    overridden = {host_model_name(row['variable_name']) for row in overrides}
    rows = [r for r in read_parameter_rows(model.parameters)
            if host_model_name(r.get('variable_name') or '') not in overridden]
    write_parameter_rows(os.path.join(resources_dir, f'{file_prefix}_parameters.csv'),
                         rows + list(overrides))
    return _generate(work_dir, resources_dir, file_prefix, model_type, solver, library_inputs,
                     model.label())


def generate_model(model, work_dir, file_prefix, overrides=(), library_inputs=None,
                   model_type='cellml', solver=None):
    '''Generate a workflow model of either kind: :func:`generate_module_array` for a
    spec.ModuleArray, :func:`generate_parameterisation` for a library module.'''
    if model.kind == 'array':
        return generate_module_array(model, work_dir, file_prefix, overrides, library_inputs,
                                     model_type, solver)
    return generate_parameterisation(model, work_dir, file_prefix, overrides, library_inputs,
                                     model_type, solver)


def _generate(work_dir, resources_dir, file_prefix, model_type, solver, library_inputs, label):
    # imported here: generation pulls in libCellML, which nothing else in the workflow needs
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    generated_models_dir = os.path.join(work_dir, 'generated_models')
    config = {'file_prefix': file_prefix, 'input_param_file': f'{file_prefix}_parameters.csv',
              'model_type': model_type, 'resources_dir': resources_dir,
              'generated_models_dir': generated_models_dir, 'DEBUG': False}
    if solver:
        config['solver'] = solver
    config.update(library_inputs or {})
    if not generate_with_new_architecture(False, config):
        raise RuntimeError(f'generating {label} failed; the generator\'s messages above say '
                           f'why.')
    model_dir = os.path.join(generated_models_dir, file_prefix)
    flat = os.path.join(model_dir, f'{file_prefix}_flat.cellml')
    return {
        'model_path': os.path.join(model_dir,
                                   file_prefix + _MODEL_EXTENSIONS.get(model_type, '.cellml')),
        # the CellML model with its imports resolved into one file (cellml only)
        'flat_model_path': flat if os.path.isfile(flat) else None,
        'parameters_path': os.path.join(model_dir, f'{file_prefix}_parameters.csv'),
        'resources_dir': resources_dir,
        'generated_models_dir': generated_models_dir,
        'inp_data_dict': config,
    }
