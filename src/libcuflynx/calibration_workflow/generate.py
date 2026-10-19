'''
Generating a workflow model -- a library module on its own, or a module array -- with some of
its parameters overridden.
'''

import csv
import json
import os
import shutil

from libcuflynx.calibration_workflow.naming import STEP_VESSEL, host_model_name

PARAMETER_HEADER = ('variable_name', 'units', 'value', 'data_reference')
# the value generation writes for a parameter nothing gave one
UNSET_VALUE = 'EMPTY_MUST_BE_FILLED'


class UnsetModelParameters(RuntimeError):
    '''Generation left parameters without a value: ``model_names`` (``x_C``).'''

    def __init__(self, label, model_names):
        self.model_names = list(model_names)
        super().__init__(f'generating {label}: no value for {", ".join(self.model_names)}.')
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
                              protocol=None):
    '''Generate the module parameterisation ``target`` (a spec.Target) alone, as a single
    module-array record named ``vessel``, under ``work_dir``.

    ``overrides`` are parameter rows (``variable_name`` as a host parameters file names it,
    ``mod__A/p``, ``units``, ``value``, ``data_reference``) written as the host parameters file, which wins
    over every parameterisation and supermodule default. ``library_inputs`` are the generation
    keys that locate the modules (``module_library_dirs`` etc.).

    Returns ``{'model_path', 'flat_model_path', 'parameters_path', 'resources_dir',
    'generated_models_dir', 'inp_data_dict'}`` (``flat_model_path``: the CellML with its
    imports resolved, None for other model types); raises RuntimeError when generation
    fails. Not MPI-aware: call it on one rank.

    ``protocol`` (``{qname: value}``, as :func:`select.protocol_values` reads an obs_data)
    gives the parameters generation leaves without a value: their rows are added and the model
    generated again. Parameters still without one raise :class:`UnsetModelParameters`.
    '''
    work_dir = os.path.abspath(work_dir)
    resources_dir = os.path.join(work_dir, 'resources')
    os.makedirs(resources_dir, exist_ok=True)
    record = {'name': vessel, 'module_type': target.module_type,
              'module_subtype': target.version, 'parameterisation': target.parameterisation,
              'inp_instances': [], 'out_instances': []}
    with open(os.path.join(resources_dir, f'{file_prefix}_module_array.json'), 'w') as f:
        json.dump([record], f, indent=1)
    write_parameter_rows(os.path.join(resources_dir, f'{file_prefix}_parameters.csv'),
                         list(overrides))
    return _generate(work_dir, resources_dir, file_prefix, model_type, solver, library_inputs,
                     target.label(), protocol)


def generate_module_array(model, work_dir, file_prefix, overrides=(), library_inputs=None,
                          model_type='cellml', solver=None, protocol=None):
    '''Generate the module array ``model`` (a spec.ModuleArray) under ``work_dir``, from a copy
    of its module array and parameters file. ``overrides`` (rows named as a host parameters
    file names them) replace the rows of the parameters file that set the same parameter.
    A select.SelectedArray is written from its records, with ``overrides`` as its parameters
    file. Returns as :func:`generate_parameterisation`.'''
    work_dir = os.path.abspath(work_dir)
    resources_dir = os.path.join(work_dir, 'resources')
    os.makedirs(resources_dir, exist_ok=True)
    records = getattr(model, 'records', None)
    if records is not None:
        with open(os.path.join(resources_dir, f'{file_prefix}_module_array.json'), 'w') as f:
            json.dump(records, f, indent=1)
        rows = []
    else:
        extension = os.path.splitext(model.module_array)[1].lower() or '.csv'
        shutil.copyfile(model.module_array,
                        os.path.join(resources_dir, f'{file_prefix}_module_array{extension}'))
        overridden = {host_model_name(row['variable_name']) for row in overrides}
        rows = [r for r in read_parameter_rows(model.parameters)
                if host_model_name(r.get('variable_name') or '') not in overridden]
    write_parameter_rows(os.path.join(resources_dir, f'{file_prefix}_parameters.csv'),
                         rows + list(overrides))
    return _generate(work_dir, resources_dir, file_prefix, model_type, solver, library_inputs,
                     model.label(), protocol)


def generate_model(model, work_dir, file_prefix, overrides=(), library_inputs=None,
                   model_type='cellml', solver=None, protocol=None):
    '''Generate a workflow model of either kind: :func:`generate_module_array` for a
    spec.ModuleArray, :func:`generate_parameterisation` for a library module.'''
    if model.kind == 'array':
        return generate_module_array(model, work_dir, file_prefix, overrides, library_inputs,
                                     model_type, solver, protocol=protocol)
    return generate_parameterisation(model, work_dir, file_prefix, overrides, library_inputs,
                                     model_type, solver, protocol=protocol)


def _unset(resources_dir, file_prefix):
    '''``[(model name, units)]`` the failed generation left without a value.'''
    path = os.path.join(resources_dir, f'{file_prefix}_parameters_unfinished.csv')
    if not os.path.isfile(path):
        return []
    return [(r['variable_name'], r.get('units', '')) for r in read_parameter_rows(path)
            if (r.get('value') or '').strip() == UNSET_VALUE]


def _generate(work_dir, resources_dir, file_prefix, model_type, solver, library_inputs, label,
              protocol=None):
    try:
        return _generate_once(work_dir, resources_dir, file_prefix, model_type, solver,
                              library_inputs, label)
    except RuntimeError:
        unset = _unset(resources_dir, file_prefix)
        if not unset:
            raise
    given = {host_model_name(qname): (qname, value) for qname, value in (protocol or {}).items()}
    filled = [{'variable_name': given[name][0], 'units': units,
               'value': repr(float(given[name][1])),
               'data_reference': 'start of the step\'s protocol (obs_data params_to_change)'}
              for name, units in unset if name in given]
    missing = [name for name, _ in unset if name not in given]
    if missing:
        raise UnsetModelParameters(label, missing)
    parameters = os.path.join(resources_dir, f'{file_prefix}_parameters.csv')
    write_parameter_rows(parameters, read_parameter_rows(parameters) + filled)
    os.remove(os.path.join(resources_dir, f'{file_prefix}_parameters_unfinished.csv'))
    return _generate_once(work_dir, resources_dir, file_prefix, model_type, solver,
                          library_inputs, label)


def _generate_once(work_dir, resources_dir, file_prefix, model_type, solver, library_inputs,
                   label):
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
