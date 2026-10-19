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


@pytest.mark.unit
def test_a_nested_instance_takes_its_super_instances_values(library):
    arrays = library['arrays']
    _write_json(os.path.join(arrays, 'sys_module_array.json'), [_record('sys', 'chain')])
    with open(os.path.join(arrays, 'sys_parameters.csv'), 'w') as f:
        f.write('variable_name,units,value,data_reference\nsys/C/q,dimensionless,2.0,host\n')
    obs = _obs(os.path.join(arrays, 'data', 'sys_c_obs.json'),
               [('z_c', 'sys__C/z', Z_C, 0.17)], {'sys__C/x': [[Y_A]]})
    params = _params(os.path.join(arrays, 'data', 'sys_c_params.csv'),
                     [('sys__C', 'q', 0.1, 10.0)])
    step = {'id': 'sys-c', 'model': {'select': ['sys/C']}, 'obs_data': obs,
            'params_for_id': params}
    raw = _workflow(library, [step], 'sys_module_array.json')
    resolved = resolve_workflow(parse_workflow(raw, path=os.path.join(arrays, 'w.json')))
    rs = resolved.steps['sys-c']
    assert rs.model.target == Target('lin_c', 'v1', 'default') and rs.place == 'sys__C'
    # the chain parameterisation's C/q and C/k, under the host's sys/C/q
    assert {r['variable_name']: r['value'] for r in rs.base_rows} == {'mod/q': '2.0',
                                                                       'mod/k': '1.0'}
    assert rs.target_parameterisation_name(rs.parameters[0]) == 'sys__C/q'


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
    test_a_nested_instance_takes_its_super_instances_values(library)
    path = os.path.join(library['arrays'], 'sys_workflow.json')
    _write_json(path, _workflow(library, [{
        'id': 'sys-c', 'model': {'select': ['sys/C']}, 'obs_data': 'data/sys_c_obs.json',
        'params_for_id': 'data/sys_c_params.csv'}], 'sys_module_array.json'))
    result = run_calibration_workflow(path, output_dir=str(tmp_path / 'run'))
    merged = {r['parameterisation_name']: r['value'] for r in result['merged_parameters']}
    assert list(merged) == ['sys__C/q']
    assert merged['sys__C/q'] == pytest.approx(np.sqrt(Z_C - Y_A), rel=1e-4)


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
