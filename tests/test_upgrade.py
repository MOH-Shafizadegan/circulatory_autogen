"""cuflynx-upgrade (libcuflynx/upgrade.py): each command converts a file in an older form into
one the readers accept, leaves a current file alone, and ``--dry-run`` writes nothing."""
import json
import os
import shutil

import pytest

from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
from libcuflynx.upgrade import main as upgrade
from libcuflynx.utilities.config_schemas import (load_supermodule_registry, normalise_module_config,
                                                 read_module_array_records)

pytestmark = pytest.mark.unit

OLD_COMPONENT = {'vessel_type': 'src', 'BC_type': 'nn', 'module_format': 'cellml',
                 'module_file': 'm.cellml', 'module_type': 'src_type', 'default_instance': 'p',
                 'entrance_ports': [],
                 'exit_ports': [{'port_type': 'flow_port', 'variables': ['v'],
                                 'multi_port': 'False'}],
                 'general_ports': [], 'variables_and_units': [['v', 'm3_per_s', 'access', 'variable']]}
OLD_SUPERMODULE = {'vessel_type': 'pair', 'BC_type': 'supermodule', 'module_format': 'supermodule',
                   'default_parameters': 'pair_defaults.csv',
                   'submodules': [{'name': 'a', 'vessel_type': 'src', 'BC_type': 'nn',
                                   'out_vessels': ['b']},
                                  {'name': 'b', 'vessel_type': 'src', 'BC_type': 'nn',
                                   'inp_vessels': ['a']}]}


def _write(path, obj):
    path.write_text(json.dumps(obj, indent=2))
    return str(path)


def test_a_module_config_is_converted(tmp_path):
    path = _write(tmp_path / 'm_modules_config.json', [OLD_COMPONENT, OLD_SUPERMODULE])
    (tmp_path / 'pair_defaults.csv').write_text('variable_name,units,value,data_reference\n'
                                                'a/v,m3_per_s,1,r\n')
    with pytest.raises(ValueError, match='cuflynx-upgrade module-config'):
        normalise_module_config(json.loads(open(path).read()), path)
    assert upgrade(['module-config', path]) == 0
    entries = normalise_module_config(json.loads(open(path).read()), path)
    component, supermodule = entries
    assert (component['module_type'], component['module_subtype'],
            component['component_file'], component['component_type']) == \
        ('src', 'nn', 'm.cellml', 'src_type')
    assert component['default_parameterisation'] == 'p'
    assert 'multi_port' not in component['exit_ports'][0]
    assert supermodule['default_parameterisation'] == 'default'
    # the inline submodules: their versions in the entry, the records in its module array
    assert 'submodules' not in supermodule
    assert supermodule['submodule_versions'] == [{'module_type': 'src', 'module_subtype': 'nn'}]
    records = read_module_array_records(str(tmp_path / 'pair_supermodule_module_array.json'))
    assert [(r['name'], r['module_type'], r['module_subtype'], r['inp_instances'],
             r['out_instances']) for r in records] == [('a', 'src', 'nn', [], ['b']),
                                                       ('b', 'src', 'nn', ['a'], [])]
    registered = load_supermodule_registry([path])[('pair', 'supermodule')]
    assert [s['name'] for s in registered['submodules']] == ['a', 'b']
    moved = tmp_path / 'parameterisations' / 'default' / 'default_parameters.csv'
    assert moved.read_text().endswith('a/v,m3_per_s,1,r\n')
    assert not (tmp_path / 'pair_defaults.csv').exists()
    before = open(path).read()
    assert upgrade(['module-config', path]) == 0
    assert open(path).read() == before


def test_a_module_array_is_converted_and_renamed(tmp_path):
    json_path = _write(tmp_path / 'm_vessel_array.json', [
        {'name': 'heart', 'vessel_type': 'heart', 'BC_type': 'vp',
         'inp_vessels': 'venous_svc', 'out_vessels': ['aortic_root']},
        {'name': 'aortic_root', 'vessel_type': 'arterial', 'BC_type': 'vv', 'inp_vessels': ['heart']}])
    csv_path = tmp_path / 'c_vessel_array.csv'
    # cells padded with spaces, as many CSV arrays are
    csv_path.write_text('name,BC_type,vessel_type,inp_vessels,out_vessels\n'
                        'a,  nn, src,,b   \nb,nn,src,a,\nh, vp, heart_LVprop, a, b\n')
    assert upgrade(['module-array', json_path, str(csv_path)]) == 0
    assert not (tmp_path / 'm_vessel_array.json').exists()
    assert not csv_path.exists()
    records = read_module_array_records(str(tmp_path / 'm_module_array.json'))
    heart = records[0]
    assert heart['inp_instances'] == ['venous_svc', 'pvn']
    assert heart['out_instances'] == ['aortic_root', 'par']
    assert [(r['name'], r['module_type'], r['inp_instances'], r['out_instances'])
            for r in records[2:]] == [('par', 'arterial_simple', ['heart'], ['pvn']),
                                       ('pvn', 'arterial_simple', ['par'], ['heart'])]
    rows = read_module_array_records(str(tmp_path / 'c_module_array.csv'))
    assert [(r['name'], r['module_type'], r['out_instances']) for r in rows] == \
        [('a', 'src', ['b']), ('b', 'src', []), ('h', 'heart_LVprop', ['b', 'par']),
         ('par', 'arterial_simple', ['pvn']), ('pvn', 'arterial_simple', ['h'])]


@pytest.mark.integration
def test_a_one_output_heart_is_refused_until_converted(tmp_path):
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture
    resources = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'resources')
    work = tmp_path / 'resources'
    work.mkdir()
    for suffix in ('_module_array.csv', '_parameters.csv'):
        shutil.copy(os.path.join(resources, 'cvs_model_0d' + suffix), work)
    array = work / 'cvs_model_0d_module_array.csv'
    rows = [r for r in array.read_text().splitlines() if not r.startswith(('par,', 'pvn,'))]
    rows = [r.replace(' pvn,', ',').replace(' par', '') if r.startswith('heart,') else r
            for r in rows]
    array.write_text('\n'.join(rows) + '\n')
    config = {'file_prefix': 'cvs_model_0d', 'input_param_file': 'cvs_model_0d_parameters.csv',
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': str(work),
              'generated_models_dir': str(tmp_path / 'generated'), 'DEBUG': False}
    with pytest.raises(ValueError, match='heart "heart" has one output.*cuflynx-upgrade module-array'):
        generate_with_new_architecture(False, config)
    assert upgrade(['module-array', str(array)]) == 0
    assert any(r.startswith('par,') for r in array.read_text().splitlines())


@pytest.mark.parametrize('doc', [
    [{'data_item_name': 'x', 'data_type': 'constant', 'unit': 'm', 'operands': ['a/x'],
      'value': 1.0, 'std': 0.1}],
    {'data_items': [{'data_item_name': 'x', 'data_type': 'constant', 'unit': 'm',
                     'operands': ['a/x'], 'value': 1.0, 'std': 0.1}]},
])
def test_an_obs_data_without_protocol_info_is_converted(tmp_path, doc):
    path = _write(tmp_path / 'm_obs_data.json', doc)
    with pytest.raises(ValueError, match='no protocol_info.*cuflynx-upgrade obs-data'):
        ObsAndParamDataParser().parse_obs_data_json(param_id_obs_path=path)
    inputs = tmp_path / 'user_inputs.yaml'
    inputs.write_text('pre_time: 20\nsim_time: 2\n')
    assert upgrade(['obs-data', path, '--dry-run', '--user-inputs', str(inputs)]) == 0
    assert json.loads(open(path).read()) == doc
    assert upgrade(['obs-data', path, '--user-inputs', str(inputs)]) == 0
    parsed = ObsAndParamDataParser().parse_obs_data_json(param_id_obs_path=path)
    assert parsed['protocol_info']['pre_times'] == [20.0]
    assert parsed['protocol_info']['sim_times'] == [[2.0]]
    assert len(parsed['gt_df']) == 1


def test_obs_data_needs_the_times(tmp_path):
    path = _write(tmp_path / 'm_obs_data.json', [])
    with pytest.raises(SystemExit, match='needs pre_time and sim_time'):
        upgrade(['obs-data', path])


def test_explicit_times_win_over_user_inputs(tmp_path):
    path = _write(tmp_path / 'm_obs_data.json', [])
    inputs = tmp_path / 'user_inputs.yaml'
    inputs.write_text('pre_time: 20\nsim_time: 2\n')
    assert upgrade(['obs-data', path, '--user-inputs', str(inputs), '--pre-time', '5']) == 0
    protocol_info = json.load(open(path))['protocol_info']
    assert protocol_info['pre_times'] == [5.0]
    assert protocol_info['sim_times'] == [[2.0]]


def test_a_heart_with_no_outputs_is_left_alone(tmp_path):
    """A heart module tested on its own has no neighbours; only a one-output heart lacks its
    pulmonary circuit."""
    path = _write(tmp_path / 'm_module_array.json',
                  [{'name': 'mod', 'module_type': 'heart', 'module_subtype': 'vp_simple'}])
    before = open(path).read()
    assert upgrade(['module-array', path]) == 0
    assert open(path).read() == before


# ------------------------------------------------------------------- supermodule submodules

def _rows(path):
    import csv
    with open(path, newline='') as f:
        return [(r['variable_name'], r['value']) for r in csv.DictReader(f)]


def _parameters(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('variable_name,units,value,data_reference\n'
                    + ''.join(f'{n},u,{v},ref\n' for n, v in rows))


def _library(root):
    """A module library: component ``src`` v1 (``k`` local, ``T`` a global constant) with
    parameterisations ``default`` and ``fast``, supermodule ``inner`` v1 (current form)
    with parameterisation ``p``, and a version still in an older form, which the readers
    refuse and the converter reads past."""
    src = root / 'src' / 'versions' / 'v1'
    src.mkdir(parents=True)
    _write(src / 'src_v1_modules_config.json', [{
        'module_type': 'src', 'module_subtype': 'v1', 'component_file': 'src.cellml',
        'component_type': 'src_type', 'default_parameterisation': 'default',
        'variables_and_units': [['k', 'm', 'access', 'constant'],
                                ['T', 'K', 'access', 'global_constant']]}])
    _parameters(src / 'parameterisations' / 'default' / 'default_parameters.csv',
                [('k', '1'), ('T', '300')])
    _parameters(src / 'parameterisations' / 'fast' / 'fast_parameters.csv',
                [('k', '5'), ('T', '310'), ('global/G', '2')])
    inner = root / 'inner' / 'versions' / 'v1'
    inner.mkdir(parents=True)
    _write(inner / 'inner_v1_modules_config.json', [{
        'module_type': 'inner', 'module_subtype': 'v1', 'module_format': 'supermodule',
        'default_parameterisation': 'd',
        'submodule_versions': [{'module_type': 'src', 'module_subtype': 'v1'}]}])
    _write(inner / 'inner_v1_module_array.json', [{'name': 'g', 'module_type': 'src',
                                                    'module_subtype': 'v1'}])
    _parameters(inner / 'parameterisations' / 'p' / 'p_parameters.csv',
                [('g/k', '7'), ('T', '320')])
    legacy = root / 'legacy' / 'versions' / 'v1'
    legacy.mkdir(parents=True)
    _write(legacy / 'legacy_v1_modules_config.json', [dict(OLD_COMPONENT, BC_type='v1')])
    return str(root)


def _sub(name, module_type='src', parameterisation=None, **extra):
    record = {'name': name, 'module_type': module_type, 'module_subtype': 'v1', **extra}
    if parameterisation is not None:
        record['parameterisation'] = parameterisation
    return record


def _supermodule(tmp_path, submodules, **entry):
    directory = tmp_path / 'sup' / 'versions' / 'v1'
    directory.mkdir(parents=True)
    return _write(directory / 'sup_v1_modules_config.json', [dict(
        {'module_type': 'sup', 'module_subtype': 'v1', 'module_format': 'supermodule',
         'submodules': submodules}, **entry)]), directory


def test_a_named_parameterisation_is_folded_into_every_supermodule_parameterisation(tmp_path):
    library = _library(tmp_path / 'lib')
    path, directory = _supermodule(
        tmp_path, [_sub('a', parameterisation='fast', out_instances=['b']),
                   _sub('b', parameterisation='default', inp_instances=['a']), _sub('c')],
        default_parameterisation='base')
    _parameters(directory / 'parameterisations' / 'base' / 'base_parameters.csv',
                [('c/k', '9'), ('T', '305')])
    _parameters(directory / 'parameterisations' / 'alt' / 'alt_parameters.csv', [('a/k', '8')])
    assert upgrade(['module-config', path, '--module-library-dir', library,
                    '--no-builtin-modules']) == 0
    entry = json.loads(open(path).read())[0]
    assert entry['submodule_versions'] == [{'module_type': 'src', 'module_subtype': 'v1'}]
    assert entry['default_parameterisation'] == 'base'
    records = read_module_array_records(str(directory / 'sup_v1_module_array.json'))
    assert [r['name'] for r in records] == ['a', 'b', 'c']
    assert not any('parameterisation' in r for r in records)
    assert records[0]['out_instances'] == ['b']
    # a's "fast" rows, as <sub>/<var> (a global plainly), after the rows each already had;
    # b named its version's default, which it gets anyway: nothing of it is folded
    assert _rows(directory / 'parameterisations' / 'base' / 'base_parameters.csv') == [
        ('c/k', '9'), ('T', '305'), ('a/k', '5'), ('G', '2')]
    assert _rows(directory / 'parameterisations' / 'alt' / 'alt_parameters.csv') == [
        ('a/k', '8'), ('T', '310'), ('G', '2')]
    load_supermodule_registry([path])


def test_a_supermodule_without_parameterisations_gets_a_default_one(tmp_path):
    library = _library(tmp_path / 'lib')
    path, directory = _supermodule(tmp_path, [_sub('a', parameterisation='fast')])
    assert upgrade(['module-config', path, '--module-library-dir', library,
                    '--no-builtin-modules']) == 0
    entry = json.loads(open(path).read())[0]
    assert entry['default_parameterisation'] == 'default'
    assert _rows(directory / 'parameterisations' / 'default' / 'default_parameters.csv') == [
        ('a/k', '5'), ('T', '310'), ('G', '2')]


def test_only_a_default_named_folds_nothing(tmp_path):
    library = _library(tmp_path / 'lib')
    path, directory = _supermodule(tmp_path, [_sub('a', parameterisation='default')])
    assert upgrade(['module-config', path, '--module-library-dir', library,
                    '--no-builtin-modules']) == 0
    entry = json.loads(open(path).read())[0]
    assert 'default_parameterisation' not in entry
    assert not (directory / 'parameterisations').exists()
    assert 'parameterisation' not in read_module_array_records(
        str(directory / 'sup_v1_module_array.json'))[0]


def test_a_nested_supermodule_s_rows_are_prefixed_with_the_submodule(tmp_path):
    library = _library(tmp_path / 'lib')
    path, directory = _supermodule(
        tmp_path, [_sub('in', module_type='inner', parameterisation='p'), _sub('s')])
    assert upgrade(['module-config', path, '--module-library-dir', library,
                    '--no-builtin-modules']) == 0
    assert json.loads(open(path).read())[0]['submodule_versions'] == [
        {'module_type': 'inner', 'module_subtype': 'v1'},
        {'module_type': 'src', 'module_subtype': 'v1'}]
    # g/k is inner's submodule g: in/g/k; T is a global and keeps its name
    assert _rows(directory / 'parameterisations' / 'default' / 'default_parameters.csv') == [
        ('in/g/k', '7'), ('T', '320')]


def test_a_named_parameterisation_without_its_library_is_refused_and_nothing_written(tmp_path):
    path, directory = _supermodule(tmp_path, [_sub('a', parameterisation='fast')])
    before = open(path).read()
    with pytest.raises(SystemExit, match=r'submodule "a" names parameterisation "fast" of '
                                         r"\('src', 'v1'\).*pass --module-library-dir"):
        upgrade(['module-config', path, '--no-builtin-modules'])
    assert open(path).read() == before
    assert not (directory / 'sup_v1_module_array.json').exists()


def test_an_existing_module_array_is_not_overwritten(tmp_path):
    path, directory = _supermodule(tmp_path, [_sub('a')])
    (directory / 'sup_v1_module_array.json').write_text('[]')
    with pytest.raises(SystemExit, match='exists already'):
        upgrade(['module-config', path])
    assert (directory / 'sup_v1_module_array.json').read_text() == '[]'


def test_a_dry_run_writes_no_module_array(tmp_path):
    library = _library(tmp_path / 'lib')
    path, directory = _supermodule(tmp_path, [_sub('a', parameterisation='fast')])
    before = open(path).read()
    assert upgrade(['module-config', path, '--dry-run', '--module-library-dir', library,
                    '--no-builtin-modules']) == 0
    assert open(path).read() == before
    assert sorted(os.listdir(directory)) == ['sup_v1_modules_config.json']
