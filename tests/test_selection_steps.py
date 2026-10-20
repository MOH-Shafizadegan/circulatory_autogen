"""Selection steps: a workflow step that calibrates instances of its module-array target as they
are in it (``{"select": [...]}``, libcuflynx.calibration_workflow.select).

One instance runs as its library module placed at that instance, as a step naming ``vessel``
does; several run as the module array they form, cut from the target. The fixture library is
libcuflynx.external_testing.workflow_library: in the ``chain`` array, C (z = x + k q^2) reads
x from A (y = 2 p^2) through a boundary condition, so C selected alone needs x from its
protocol, and with x = Y_A it gives q = sqrt(Z_C - Y_A) = 3, as calibrating C in the chain does.
"""
import csv
import json
import os

import numpy as np
import pytest

from libcuflynx.calibration_workflow import (UnsetParametersError, WorkflowError,
                                             plan_workflow, resolve_workflow,
                                             run_calibration_workflow, workflow_model)
from libcuflynx.calibration_workflow.spec import Target, parse_workflow
from libcuflynx.utilities.module_library import ModuleSources
from libcuflynx.utilities.supermodules import write_instance_parameters
from libcuflynx.external_testing.workflow_library import (OPTIMISER_SETTINGS, STD_A, STD_B,
                                                          Y_A, Y_B, Z_C, build_workflow_library)


@pytest.fixture
def library(tmp_path):
    return build_workflow_library(str(tmp_path / 'library'))


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


def _record(name, module_type, inp=(), out=(), parameterisation='default'):
    return {'name': name, 'module_type': module_type, 'module_subtype': 'v1',
            'parameterisation': parameterisation, 'inp_instances': list(inp),
            'out_instances': list(out)}


def _obs(path, items, params_to_change=None):
    _write_json(path, {
        'protocol_info': {'pre_times': [0.0], 'sim_times': [[1.0]],
                          'params_to_change': params_to_change or {}},
        'data_items': [{'data_item_name': name, 'data_type': 'constant', 'unit': 'dimensionless',
                        'operands': [operand], 'operation': 'mean', 'weight': 1.0,
                        'value': value, 'std': std} for name, operand, value, std in items],
        'prediction_items': []})
    return path


def _params(path, rows):
    with open(path, 'w') as f:
        f.write('vessel_name,param_name,min,max,name_for_plotting\n')
        f.writelines(f'{v},{p},{lo},{hi},{p}_{v}\n' for v, p, lo, hi in rows)
    return path


def _workflow(library, steps, target='chain_module_array.json'):
    return {'schema_version': 1, 'workflow_name': 'selection',
            'module_library_dirs': [library['modules']], 'settings': OPTIMISER_SETTINGS,
            'target': target, 'steps': steps}


def _write_workflow(library, steps, target='chain_module_array.json'):
    path = os.path.join(library['arrays'], 'selection_workflow.json')
    _write_json(path, _workflow(library, steps, target))
    return path


def _c_alone(library, x=None):
    '''C selected alone, its protocol setting x (or not).'''
    obs = _obs(os.path.join(library['arrays'], 'data', 'c_obs.json'), [('z_c', 'C/z', Z_C, 0.17)],
               {'C/x': [[x]]} if x is not None else None)
    params = _params(os.path.join(library['arrays'], 'data', 'c_params.csv'),
                     [('C', 'q', 0.1, 10.0)])
    return {'id': 'c-alone', 'model': {'select': ['C']}, 'obs_data': obs,
            'params_for_id': params}


def _parameters(path):
    with open(path) as f:
        return {r['variable_name']: float(r['value']) for r in csv.DictReader(f)}


# --------------------------------------------------------------------------------------------
# the file
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize('edit, message', [
    (lambda w: w.update(target={'select': ['A']}), "at target: .*'select' was unexpected"),
    (lambda w: w['steps'][0].update(vessel='C'), 'has no "submodule", "vessel" or "vessel_map"'),
    (lambda w: w['steps'][0]['model'].update(select=[]), 'should be non-empty|is too short'),
    (lambda w: w.update(target={'module_type': 'chain', 'version': 'v1',
                                'parameterisation': 'rest'}),
     'only a module-array target has'),
])
def test_a_selection_is_a_step_of_a_module_array_target(library, edit, message):
    raw = _workflow(library, [_c_alone(library, x=Y_A)])
    edit(raw)
    with pytest.raises(WorkflowError, match=message):
        parse_workflow(raw, path=os.path.join(library['arrays'], 'w.json'))


@pytest.mark.unit
def test_one_instance_runs_as_its_library_module_placed_there(library):
    resolved = resolve_workflow(parse_workflow(
        _workflow(library, [_c_alone(library, x=Y_A)]),
        path=os.path.join(library['arrays'], 'w.json')))
    step = resolved.steps['c-alone']
    assert step.model.target == Target('lin_c', 'v1', 'default')
    assert step.place == 'C' and step.renames_inputs
    # the target's values for C, named for its model, which is generated as "mod"
    assert {r['variable_name']: r['value'] for r in step.base_rows} == {'mod/q': '1.0',
                                                                         'mod/k': '1.0'}
    assert [(p.vessel, p.param) for p in step.parameters] == [('mod', 'q')]
    assert step.target_parameterisation_name(step.parameters[0]) == 'C/q'


@pytest.mark.unit
def test_several_instances_run_as_the_array_they_form(library):
    arrays = library['arrays']
    _write_json(os.path.join(arrays, 'abc_module_array.json'),
                [_record('A', 'lin_a', out=['C']), _record('C', 'lin_c', inp=['A']),
                 _record('B', 'lin_b')])
    with open(os.path.join(arrays, 'abc_parameters.csv'), 'w') as f:
        f.write('variable_name,units,value,data_reference\nA/p,dimensionless,1.5,x\n'
                'B/p,dimensionless,2.5,x\nC/q,dimensionless,1.0,x\n')
    obs = _obs(os.path.join(arrays, 'data', 'ab_obs.json'),
               [('y_a', 'A/y', Y_A, STD_A), ('y_b', 'B/y', Y_B, STD_B)])
    params = _params(os.path.join(arrays, 'data', 'ab_params.csv'),
                     [('A', 'p', 0.1, 5.0), ('B', 'p', 0.1, 5.0)])
    step = {'id': 'ab', 'model': {'select': ['A', 'B']}, 'obs_data': obs, 'params_for_id': params}
    resolved = resolve_workflow(parse_workflow(
        _workflow(library, [step], 'abc_module_array.json'), path=os.path.join(arrays, 'w.json')))
    ab = resolved.steps['ab']
    assert ab.place == '' and not ab.renames_inputs
    # A's link to C, outside the selection, is dropped
    assert [(r['name'], r['out_instances']) for r in ab.model.records] == [('A', []), ('B', [])]
    assert [r['variable_name'] for r in ab.base_rows] == ['A/p', 'B/p']
    plan = plan_workflow(parse_workflow(_workflow(library, [step], 'abc_module_array.json'),
                                        path=os.path.join(arrays, 'w.json')))
    assert plan['steps'][0]['selection'] == {'runs_as': 'module_array', 'records': ['A', 'B']}
    # CUFLynx finds a bare-path target's files here, to send them back to PhLynx
    assert plan['target'] == {'module_array': os.path.join(arrays, 'abc_module_array.json'),
                              'parameters': os.path.join(arrays, 'abc_parameters.csv')}


@pytest.mark.unit
@pytest.mark.parametrize('select, message', [
    (['D'], 'not an instance of the target'),
    (['A/C'], 'has no submodule'),      # A is not a super instance
])
def test_a_selection_names_instances_of_the_target(library, select, message):
    step = dict(_c_alone(library, x=Y_A), model={'select': select})
    with pytest.raises(WorkflowError, match=message):
        resolve_workflow(parse_workflow(_workflow(library, [step]),
                                        path=os.path.join(library['arrays'], 'w.json')))


def _nested_sys(library, select='sys/C'):
    '''A target ``sys`` of one chain instance, with its instance parameters file new from the
    chain version's, and a step selecting ``sys/C``, resolved.'''
    arrays = library['arrays']
    array = os.path.join(arrays, 'sys_module_array.json')
    if not os.path.isfile(array):
        _write_json(array, [_record('sys', 'chain')])
        write_instance_parameters(array, ModuleSources(
            {'module_library_dirs': [library['modules']]}).supermodule_registry())
    with open(os.path.join(arrays, 'sys_parameters.csv'), 'w') as f:
        f.write('variable_name,units,value,data_reference\nsys/C/q,dimensionless,2.0,host\n')
    obs = _obs(os.path.join(arrays, 'data', 'sys_c_obs.json'),
               [('z_c', 'sys__C/z', Z_C, 0.17)], {'sys__C/x': [[Y_A]]})
    params = _params(os.path.join(arrays, 'data', 'sys_c_params.csv'),
                     [('sys__C', 'q', 0.1, 10.0)])
    step = {'id': 'sys-c', 'model': {'select': [select]}, 'obs_data': obs,
            'params_for_id': params}
    raw = _workflow(library, [step], 'sys_module_array.json')
    return resolve_workflow(parse_workflow(raw, path=os.path.join(arrays, 'w.json')))


@pytest.mark.unit
def test_a_nested_instance_takes_its_super_instances_values(library):
    rs = _nested_sys(library).steps['sys-c']
    assert rs.model.target == Target('lin_c', 'v1', 'default') and rs.place == 'sys__C'
    # the chain's parameters file's C/q and C/k, under the host's sys/C/q
    assert {r['variable_name']: r['value'] for r in rs.base_rows} == {'mod/q': '2.0',
                                                                       'mod/k': '1.0'}
    assert rs.target_parameterisation_name(rs.parameters[0]) == 'sys__C/q'


def _version_file(library):
    return os.path.join(library['modules'], 'chain', 'versions', 'v1', 'chain_v1_parameters.json')


def _sys_file(library):
    return os.path.join(library['arrays'], 'sys_supermodule_parameters', 'sys.json')


def _sys_instance(library, instance):
    """The model's values for instance ``instance`` of its super instance sys."""
    return _read_json(_sys_file(library))['instances'][instance]


@pytest.mark.unit
def test_a_nested_instance_takes_the_model_s_values_not_the_version_s(library):
    """The model's sys.json gives C's k, whatever the chain version's file says."""
    _nested_sys(library)
    doc = _read_json(_sys_file(library))
    doc['instances']['C']['parameters']['k']['value'] = 1.5
    _write_json(_sys_file(library), doc)
    version = _read_json(_version_file(library))
    version['instances']['C']['parameters']['k']['value'] = 9.0
    _write_json(_version_file(library), version)
    rows = _nested_sys(library).steps['sys-c'].base_rows
    assert {r['variable_name']: r['value'] for r in rows} == {'mod/q': '2.0', 'mod/k': '1.5'}


@pytest.mark.unit
def test_a_super_instance_without_its_parameters_file_is_refused(library):
    _nested_sys(library)
    os.remove(_sys_file(library))
    with pytest.raises(WorkflowError, match=r'sys\.json does not exist.*cuflynx-upgrade '
                                            r'supermodule-instances'):
        _nested_sys(library)


@pytest.mark.unit
def test_a_selected_super_instance_is_generated_with_the_model_s_values(library):
    """Selecting sys itself runs the chain version as mod, whose parameters file is the model's
    sys.json, not a new instance's."""
    _nested_sys(library)
    doc = _read_json(_sys_file(library))
    doc['instances']['C']['parameters']['k']['value'] = 1.5
    _write_json(_sys_file(library), doc)
    selection = _nested_sys(library, select='sys').steps['sys-c'].selection
    assert selection.instance_parameters == doc


@pytest.mark.unit
def test_a_nested_instance_starts_from_the_parameterisation_its_supermodule_records(library):
    """The model's sys.json records that C started from lin_c's "fit_c"; the step model is that
    parameterisation, with the file's values over it."""
    _nested_sys(library)
    path = _sys_file(library)
    doc = _read_json(path)
    doc['instances']['C']['parameterisation'] = 'fit_c'
    _write_json(path, doc)
    fit_c = os.path.join(library['modules'], 'lin_c', 'versions', 'v1', 'parameterisations',
                         'fit_c')
    os.makedirs(fit_c)
    with open(os.path.join(fit_c, 'fit_c_parameters.csv'), 'w') as f:
        f.write('variable_name,units,value,data_reference\nq,dimensionless,5.0,fit\n')
    arrays = library['arrays']
    resolved = resolve_workflow(parse_workflow(
        _workflow(library, [{'id': 'sys-c', 'model': {'select': ['sys/C']},
                             'obs_data': os.path.join(arrays, 'data', 'sys_c_obs.json'),
                             'params_for_id': os.path.join(arrays, 'data', 'sys_c_params.csv')}],
                  'sys_module_array.json'), path=os.path.join(arrays, 'w.json')))
    assert resolved.steps['sys-c'].model.target == Target('lin_c', 'v1', 'fit_c')


@pytest.mark.unit
def test_calibrated_values_inside_a_super_instance_go_to_the_model_s_instance_file(
        library, tmp_path):
    """sys/C/q and sys/C/k are written to the model's parameters file of sys (instance C), and
    the chain version's file, a new instance's template, is left alone; the written host file
    updates only sys/C/q, the row the host file has."""
    from libcuflynx.calibration_workflow.runner import write_calibrated_parameters
    resolved = _nested_sys(library)
    merged = [{'model_name': f'{v}_sys__C', 'parameterisation_name': f'sys__C/{v}', 'value': x,
               'units': 'dimensionless', 'from_step': 'sys-c'} for v, x in (('q', 3.0), ('k', 4.0))]
    version = open(_version_file(library)).read()
    out = write_calibrated_parameters(resolved.target, merged, 'w', path=str(tmp_path / 'out.csv'))
    assert open(_version_file(library)).read() == version
    instance = _sys_instance(library, 'C')
    assert instance['parameters']['q']['value'] == 3.0 and instance['parameters']['k']['value'] == 4.0
    assert instance['parameters']['q']['data_reference'].startswith('calibrated by workflow w, step sys-c')
    assert _parameters(out) == {'sys/C/q': 3.0}


@pytest.mark.unit
def test_a_packed_selection_keeps_its_instances_and_packs_its_inputs(library, tmp_path):
    from libcuflynx.calibration_workflow import pack_workflow

    path = _write_workflow(library, [_c_alone(library, x=Y_A)])
    dest = str(tmp_path / 'packed')
    pack_workflow(path, dest)
    packed = _read_json(os.path.join(dest, 'calibration_workflow.json'))
    step = packed['steps'][0]
    assert step['model'] == {'select': ['C']}
    assert os.path.isfile(os.path.join(dest, step['obs_data']))
    assert os.path.isfile(os.path.join(dest, packed['target']['module_array']))


@pytest.mark.unit
def test_a_packed_target_carries_its_super_instances_parameters_files(library, tmp_path):
    from libcuflynx.calibration_workflow import pack_workflow
    _nested_sys(library)
    path = os.path.join(library['arrays'], 'sys_workflow.json')
    _write_json(path, _workflow(library, [{
        'id': 'sys-c', 'model': {'select': ['sys/C']}, 'obs_data': 'data/sys_c_obs.json',
        'params_for_id': 'data/sys_c_params.csv'}], 'sys_module_array.json'))
    dest = str(tmp_path / 'packed')
    files = pack_workflow(path, dest)['files']
    array = _read_json(os.path.join(dest, 'calibration_workflow.json'))['target']['module_array']
    packed = os.path.join(os.path.dirname(array), 'sys_supermodule_parameters', 'sys.json')
    assert packed in files
    assert _read_json(os.path.join(dest, packed)) == _read_json(_sys_file(library))


# --------------------------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_a_boundary_left_open_needs_a_protocol(library, tmp_path):
    path = _write_workflow(library, [_c_alone(library)])
    with pytest.raises(UnsetParametersError) as caught:
        workflow_model(path, 'c-alone', output_dir=str(tmp_path / 'run'),
                       work_dir=str(tmp_path / 'views'))
    assert caught.value.names == ['C/x'] and caught.value.step_id == 'c-alone'
    assert 'protocol_info.params_to_change' in str(caught.value)


@pytest.mark.integration
def test_one_instance_calibrated_from_its_protocol_lands_on_that_instance(library, tmp_path):
    path = _write_workflow(library, [_c_alone(library, x=Y_A)])
    view = workflow_model(path, 'c-alone', output_dir=str(tmp_path / 'run'),
                          work_dir=str(tmp_path / 'views'))
    assert view['place'] == 'C'
    assert _parameters(view['parameters_path'])['x_mod'] == Y_A
    # the tab's obs_data names the model's vessel
    assert _read_json(view['obs_data_path'])['data_items'][0]['operands'] == ['mod/z']

    result = run_calibration_workflow(path, output_dir=str(tmp_path / 'run'),
                                      write_calibrated=True)
    assert result['complete']
    merged = {r['parameterisation_name']: r['value'] for r in result['merged_parameters']}
    assert list(merged) == ['C/q']
    assert merged['C/q'] == pytest.approx(np.sqrt(Z_C - Y_A), rel=1e-4)
    # the step's inputs, as it ran them, are kept with its results
    step_dir = tmp_path / 'run' / 'c-alone'
    assert (step_dir / 'inputs' / 'c-alone_params_for_id.json').is_file()
    calibrated = _parameters(os.path.join(library['arrays'],
                                          'chain_calibrated_parameters.csv'))
    assert calibrated['C/q'] == pytest.approx(3.0, rel=1e-4)
    assert calibrated['A/p'] == 1.0     # outside the selection: unchanged


@pytest.mark.integration
def test_a_nested_instance_is_calibrated_into_its_super_instance(library, tmp_path):
    _nested_sys(library)
    path = os.path.join(library['arrays'], 'sys_workflow.json')
    _write_json(path, _workflow(library, [{
        'id': 'sys-c', 'model': {'select': ['sys/C']}, 'obs_data': 'data/sys_c_obs.json',
        'params_for_id': 'data/sys_c_params.csv'}], 'sys_module_array.json'))
    result = run_calibration_workflow(path, output_dir=str(tmp_path / 'run'),
                                      write_calibrated=True)
    merged = {r['parameterisation_name']: r['value'] for r in result['merged_parameters']}
    assert list(merged) == ['sys__C/q']
    assert merged['sys__C/q'] == pytest.approx(np.sqrt(Z_C - Y_A), rel=1e-4)
    # in the model's parameters file of sys, and in the host's row that would win over it; the
    # chain version's file is not touched
    assert _sys_instance(library, 'C')['parameters']['q']['value'] == \
        pytest.approx(np.sqrt(Z_C - Y_A), rel=1e-4)
    assert _read_json(_version_file(library))['instances']['C']['parameters']['q']['value'] == 1.0
    calibrated = _parameters(os.path.join(library['arrays'], 'sys_calibrated_parameters.csv'))
    assert calibrated == {'sys/C/q': pytest.approx(np.sqrt(Z_C - Y_A), rel=1e-4)}


@pytest.mark.integration
def test_several_instances_are_calibrated_as_the_array_they_form(library, tmp_path):
    test_several_instances_run_as_the_array_they_form(library)
    path = os.path.join(library['arrays'], 'ab_workflow.json')
    _write_json(path, _workflow(library, [{
        'id': 'ab', 'model': {'select': ['A', 'B']}, 'obs_data': 'data/ab_obs.json',
        'params_for_id': 'data/ab_params.csv'}], 'abc_module_array.json'))
    result = run_calibration_workflow(path, output_dir=str(tmp_path / 'run'),
                                      write_calibrated=True)
    merged = {r['parameterisation_name']: r['value'] for r in result['merged_parameters']}
    assert merged == pytest.approx({'A/p': np.sqrt(Y_A / 2), 'B/p': np.sqrt(Y_B / 3)}, rel=1e-4)
    calibrated = _parameters(os.path.join(library['arrays'], 'abc_calibrated_parameters.csv'))
    assert calibrated['C/q'] == 1.0     # outside the selection: unchanged
