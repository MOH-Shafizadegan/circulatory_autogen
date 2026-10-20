"""Module versions and their parameterisations (utilities/parameterisations.py).

A module library lays a module version out as ``<module_type>/versions/<version>/`` with its
CellML, a one-entry ``*_modules_config.json`` (module_subtype == version) and named parameter
sets, *parameterisations*, in ``parameterisations/<name>/<name>_parameters.csv``. A
module-array record picks one with ``"parameterisation"``; without it, the entry's
``default_parameterisation`` is used when its file exists. A supermodule's submodule names none:
its values are the supermodule instance's parameters file in the model
(``<prefix>_supermodule_parameters/<instance>.json``, written from the version's parameters file),
then its version's default fills the rest; the supermodule's parameterisation sets globals only.
Rows ``{var}`` become ``{var}_{vessel}``, except the module's global constants, and are default
parameters: the host parameters file wins, then a supermodule's parameterisation, then the
instance's parameters file, then the submodules' version defaults. The older ``instances/`` directory and
``default_instance`` key are refused.

obs_data files may carry a top-level ``obs_data_name``; one filed under
``parameterisations/<name>/`` is warned about when the name differs.

Most tests run against the real module library, circulatory-autogen-modules, at the commit CI
pins (``tests/MODULE_LIBRARY_REF``; found as ``tests/_module_library.py`` describes, skipped
without it). They read what they expect from the library's own instance files, so a value that
is recalibrated there changes nothing here. The modules they use:

* ``i_M`` (``Argus2026_v01``): instances ``default`` and ``davis2020_wistar`` among others, whose
  ``rho_M`` differ; ``F`` and ``T`` are global constants;
* ``soma`` (``sympathetic``): a supermodule with a parameterisation;
* ``neuron`` (``sympathetic``): a supermodule with ``soma`` nested in it.

A small synthetic library is kept for what a real library cannot contain -- its own structure
tests forbid it: a ``default_parameterisation`` whose file is missing and a config without a
``parameterisations/`` directory (``pulse_src`` v2), a supermodule with three parameterisations
(``pair``, ``base`` and ``alt``, and ``legacy`` with a ``<sub>/<var>`` row, which is refused), and two instances that set one global constant to
different values (``pulse_src`` v1 ``default`` and ``other``; no real module's instances
disagree on a global). Its reader modules are those of test_config_schemas.py.
"""
import csv
import json
import os
import shutil
import types
import warnings

import pytest

import _module_library as module_library
from _supermodules import split_supermodule, write_supermodule
from test_config_schemas import (_assert_same_generated_models, _prescribed, _write_library,
                                 modules_config)

from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.utilities.config_schemas import (load_component_registry,
                                                 load_expanded_vessel_records,
                                                 load_supermodule_registry,
                                                 normalise_module_config_entry,
                                                 normalise_vessel_record)
from libcuflynx.utilities.parameterisations import (ConflictingGlobalWarning,
                                                   available_parameterisations, global_constants,
                                                   parameterisation_parameters_path, read_parameter_rows)
from libcuflynx.utilities.parameter_names import supermodule_row_name
from libcuflynx.utilities.config_schemas import supermodule_instance_parameters_path
from libcuflynx.utilities.supermodules import (submodule_path_parts, write_instance_parameters)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(REPO_ROOT, 'resources')

# --------------------------------------------------------------------------------------------
# the real module library
# --------------------------------------------------------------------------------------------

I_M = ('i_M', 'Argus2026_v01')
I_M_NAMED = 'davis2020_wistar'
SOMA = ('soma', 'sympathetic')
NEURON = ('neuron', 'sympathetic')
HEART = ('heart', 'Argus2026_v01')


@pytest.fixture(scope='module')
def real_library():
    """The library's ``modules`` dir, its config files and both registries."""
    modules = str(module_library.modules_dir_or_skip())
    from libcuflynx.utilities.module_library import ModuleSources
    # the library holds its own copies of the built-in modules, so it is used alone, as its
    # own harness does
    files = ModuleSources({'module_library_dirs': [modules],
                           'use_builtin_modules': False}).config_files
    return types.SimpleNamespace(modules=modules, files=files,
                                 components=load_component_registry(files),
                                 supermodules=load_supermodule_registry(files))


def _entry(lib, key):
    return lib.supermodules.get(key) or lib.components[key]


def _parameterisation_values(lib, key, instance):
    """``{variable_name: (value, data_reference)}`` of one real instance's file."""
    path = parameterisation_parameters_path(_entry(lib, key)['config_path'], instance)
    return {row['variable_name']: (row['value'], row['data_reference'])
            for row in read_parameter_rows(path, 'test')}


def _real_load(tmp_path, lib, records):
    path = str(tmp_path / 'm_module_array.json')
    _write_json(path, records)
    write_instance_parameters(path, lib.supermodules)
    return load_expanded_vessel_records(path, lib.supermodules, lib.components)


# --------------------------------------------------------------------------------------------
# the synthetic library (only what a real library cannot contain; see the module docstring)
# --------------------------------------------------------------------------------------------

PULSE_CELLML = (
    "<?xml version='1.0' encoding='UTF-8'?>\n"
    '<model name="pulse_src_v1_modules" xmlns="http://www.cellml.org/cellml/1.1#" '
    'xmlns:cellml="http://www.cellml.org/cellml/1.1#">\n'
    + _prescribed('pulse_src_type', 'v', 'm3_per_s')
    + '</model>\n'
)


def _pulse_entry(version, **extra):
    entry = {
        'module_type': 'pulse_src', 'module_subtype': version,
        'component_file': f'pulse_src_{version}_modules.cellml',
        'component_type': 'pulse_src_type', 'module_format': 'cellml',
        'entrance_ports': [], 'exit_ports': [{'port_type': 'flow_port', 'variables': ['v']}],
        'general_ports': [],
        'variables_and_units': [['v', 'm3_per_s', 'access', 'variable'],
                                ['mean', 'm3_per_s', 'access', 'constant'],
                                ['amp', 'dimensionless', 'access', 'constant'],
                                ['omega', 'per_s', 'access', 'global_constant']],
    }
    entry.update(extra)
    return entry


def _sub(name, **extra):
    record = {'name': name, 'module_type': 'pulse_src', 'module_subtype': 'v1',
              'inp_instances': [], 'out_instances': []}
    record.update(extra)
    return record


# with its instance records inline; on disk, PAIR_ENTRY and its module array
PAIR = {'module_type': 'pair', 'module_subtype': 'v1', 'module_format': 'supermodule',
        'submodules': [_sub('a'), _sub('b')], 'default_parameterisation': 'base',
        # its parameters file: between its parameterisation and the submodules' defaults
        'instance_parameters': {'a': {'amp': (0.7, 'dimensionless', 'file_ref')},
                                'b': {'amp': (0.2, 'dimensionless', 'file_ref')}}}
PAIR_ENTRY, _, _ = split_supermodule(PAIR)

# instance -> rows (variable_name, units, value)
PULSE_INSTANCES = {
    'default': [('mean', 'm3_per_s', '1e-05'), ('amp', 'dimensionless', '0.5'),
                ('omega', 'per_s', '6.0')],
    'other': [('mean', 'm3_per_s', '2e-05'), ('amp', 'dimensionless', '0.3'),
              ('omega', 'per_s', '4.0')],
}
PAIR_INSTANCES = {
    'base': [('omega', 'per_s', '5.0')],
    'alt': [],
    # a <sub>/<var> row, which a supermodule parameterisation may no longer hold
    'legacy': [('b/amp', 'dimensionless', '0.1')],
}


def _write_parameters(path, rows, reference, sourced=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('variable_name,units,value,data_reference' + (',sourced' if sourced else '') + '\n')
        for name, units, value in rows:
            f.write(f'{name},{units},{value},{reference}' + (',yes' if sourced else '') + '\n')


def _write_instances(version_dir, instances):
    for instance, rows in instances.items():
        _write_parameters(os.path.join(version_dir, 'parameterisations', instance,
                                       f'{instance}_parameters.csv'),
                          rows, f'{instance}_ref', sourced=instance == 'other')


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


@pytest.fixture(scope='module')
def library(tmp_path_factory):
    """``(module_library_dir, external_modules_dir)``"""
    root = tmp_path_factory.mktemp('instance_library')
    modules = os.path.join(str(root), 'modules')
    for version, extra in (('v1', {'default_parameterisation': 'default'}),
                           ('v2', {'default_parameterisation': 'missing'})):
        version_dir = os.path.join(modules, 'sources', 'pulse_src', 'versions', version)
        os.makedirs(version_dir)
        with open(os.path.join(version_dir, f'pulse_src_{version}_modules.cellml'), 'w') as f:
            f.write(PULSE_CELLML.replace('pulse_src_v1_modules', f'pulse_src_{version}_modules'))
        _write_json(os.path.join(version_dir, f'pulse_src_{version}_modules_config.json'),
                    [_pulse_entry(version, **extra)])
    _write_instances(os.path.join(modules, 'sources', 'pulse_src', 'versions', 'v1'),
                     PULSE_INSTANCES)
    pair_dir = os.path.join(modules, 'systems', 'pair', 'versions', 'v1')
    write_supermodule(pair_dir, PAIR)
    _write_instances(pair_dir, PAIR_INSTANCES)
    readers = _write_library(str(root / 'readers'),
                             {'schema_test_modules_config.json': modules_config()})
    return modules, readers


def _config_files(library):
    modules, readers = library
    found = []
    for directory, _, files in os.walk(modules):
        found += [os.path.join(directory, f) for f in sorted(files)
                  if f.endswith('_modules_config.json')]
    return sorted(found) + [os.path.join(readers, 'schema_test_modules_config.json')]


def _rec(name, module_type, subtype='nn', inp=(), out=(), **extra):
    record = {'name': name, 'module_type': module_type, 'module_subtype': subtype,
              'inp_instances': list(inp), 'out_instances': list(out)}
    record.update(extra)
    return record


def _load(tmp_path, library, records, instances=True):
    path = str(tmp_path / 'm_module_array.json')
    _write_json(path, records)
    files = _config_files(library)
    supermodules = load_supermodule_registry(files)
    if instances:
        write_instance_parameters(path, supermodules)
    return load_expanded_vessel_records(path, supermodules, load_component_registry(files))


def _values(rows):
    return {row['variable_name']: (row['value'], row['data_reference']) for row in rows}


def _two_sources(first_instance='other', second_instance=None):
    s1 = _rec('s1', 'pulse_src', 'v1', out=['coll'])
    s2 = _rec('s2', 'pulse_src', 'v1', out=['coll'])
    if first_instance is not None:
        s1['parameterisation'] = first_instance
    if second_instance is not None:
        s2['parameterisation'] = second_instance
    return [s1, s2, _rec('coll', 'collector', inp=['s1', 's2'])]


def _pair_host(**instance):
    return [_rec('pr', 'pair', 'v1', out=['coll'],
                 per_submodule_outputs={'a': ['coll'], 'b': ['coll']}, **instance),
            _rec('coll', 'collector', inp=['pr'])]



# --------------------------------------------------------------------------------------------
# component instances
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_a_named_parameterisation_and_the_default_one_are_applied_with_the_vessel_suffix(
        tmp_path, real_library):
    lib = real_library
    named = _parameterisation_values(lib, I_M, I_M_NAMED)
    default = _parameterisation_values(lib, I_M, _entry(lib, I_M)['default_parameterisation'])
    assert named['rho_M'] != default['rho_M'], 'the library no longer tells them apart'
    globals_ = global_constants(_entry(lib, I_M))
    assert globals_, 'i_M has no global constants left to test with'

    _, rows = _real_load(tmp_path, lib, [_rec('s1', *I_M, parameterisation=I_M_NAMED), _rec('s2', *I_M)])

    expected = {}
    for vessel, values in (('s1', named), ('s2', default)):
        for name, value in values.items():
            # a global keeps its plain name and comes once, from the first record
            expected.setdefault(name if name in globals_ else f'{name}_{vessel}', value)
    assert _values(rows) == expected
    names = [r['variable_name'] for r in rows]
    for name in globals_ & set(named):
        assert names.count(name) == 1 and f'{name}_s1' not in names
    # only the four parameter columns are kept (the library's "sourced" column is ignored)
    assert all(set(r) == {'variable_name', 'units', 'value', 'data_reference'} for r in rows)


def _conflicts(caught):
    return [w.message for w in caught if isinstance(w.message, ConflictingGlobalWarning)]


def _load_caught(tmp_path, library, records):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        records, rows = _load(tmp_path, library, records)
    return rows, _conflicts(caught)


@pytest.mark.unit
def test_the_first_record_sets_a_global_shared_by_several_parameterisations(tmp_path, library):
    # synthetic: no two instances in the real library set a global to different values
    rows, conflicts = _load_caught(tmp_path, library, _two_sources(None, 'other'))
    assert _values(rows)['omega'] == ('6.0', 'default_ref')
    assert _values(rows)['mean_s2'] == ('2e-05', 'other_ref')
    # ...and the disagreement is reported, naming both values and who set them
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.name == 'omega'
    assert [v for v, _, _ in conflict.values] == ['6.0', '4.0']
    message = str(conflict)
    assert ('"s1" (parameterisation "default")' in message
            and '"s2" (parameterisation "other")' in message)
    assert 'host parameters file' in message


@pytest.mark.unit
def test_parameterisations_that_agree_on_a_global_do_not_warn(tmp_path, library):
    _, conflicts = _load_caught(tmp_path, library, _two_sources('other', 'other'))
    assert not conflicts


@pytest.mark.unit
def test_a_global_a_supermodule_parameterisation_sets_is_not_a_conflict_of_its_submodules(tmp_path,
                                                                                  library):
    # pair's submodules' version default sets omega to 6.0, but pair's own parameterisation
    # sets 5.0, which outranks it
    rows, conflicts = _load_caught(tmp_path, library, _pair_host())
    assert _values(rows)['omega'][0] == '5.0'
    assert not conflicts


@pytest.mark.unit
def test_a_supermodule_s_global_used_for_modules_outside_it_warns(tmp_path, library):
    # pr1 (parameterisation base) sets omega 5.0; pr2 (alt) sets none, so its submodules' version
    # default chose 6.0 -- but a global is one value for the model, and pr2's modules run at 5.0
    records = [_rec('pr1', 'pair', 'v1', out=['coll'], parameterisation='base',
                    per_submodule_outputs={'a': ['coll'], 'b': ['coll']}),
               _rec('pr2', 'pair', 'v1', out=['coll'], parameterisation='alt',
                    per_submodule_outputs={'a': ['coll'], 'b': ['coll']}),
               _rec('coll', 'collector', inp=['pr1', 'pr2'])]
    rows, conflicts = _load_caught(tmp_path, library, records)
    assert _values(rows)['omega'][0] == '5.0'
    assert len(conflicts) == 1
    assert [v for v, _, _ in conflicts[0].values] == ['5.0', '6.0', '6.0']
    message = str(conflicts[0])
    assert 'supermodule "pr1" (parameterisation "base")' in message and '"pr2__a"' in message
    # pr1's own submodules (6.0 from their version default) are overridden on purpose: not listed
    assert '"pr1__a"' not in message and '"pr1__b"' not in message


@pytest.mark.unit
def test_the_host_file_settles_a_conflicting_global():
    from libcuflynx.utilities.parameterisations import reissue_warnings, warn_conflicting_globals
    settings = [('T', '295.15', 'kelvin', '"a"', 'a'), ('T', '310', 'kelvin', '"b"', 'b')]
    for settled, expected in (((), 1), ({'T'}, 0)):
        with warnings.catch_warnings(record=True) as held:
            warnings.simplefilter('always')
            warn_conflicting_globals(settings, 'm')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            reissue_warnings(held, settled=settled)
        assert len(_conflicts(caught)) == expected
    # different units: most likely two quantities sharing a name
    with warnings.catch_warnings(record=True) as held:
        warnings.simplefilter('always')
        warn_conflicting_globals([('T', '295.15', 'kelvin', '"cell"', 'cell'),
                                  ('T', '1', 'second', '"clock"', 'clock')], 'm')
    assert 'units differ' in str(_conflicts(held)[0])
    # an outer supermodule overriding the global of one nested in it is not a conflict
    with warnings.catch_warnings(record=True) as held:
        warnings.simplefilter('always')
        warn_conflicting_globals([('T', '1', 'K', 'outer', 'h'), ('T', '2', 'K', 'inner', 'h_soma')],
                                 'm')
    assert not _conflicts(held)


@pytest.mark.unit
def test_nothing_is_loaded_without_a_parameterisation_or_an_existing_default(tmp_path, library):
    # synthetic: v2 declares default_parameterisation "missing" but has no parameterisations/:
    # nothing (the real library's structure test forbids such a version)
    records = [_rec('s', 'pulse_src', 'v2', out=['coll']), _rec('coll', 'collector', inp=['s'])]
    _, rows = _load(tmp_path, library, records)
    assert rows == []


@pytest.mark.unit
def test_an_unknown_parameterisation_names_the_version_directory_and_its_parameterisations(tmp_path,
                                                                            real_library):
    with pytest.raises(ValueError) as error:
        _real_load(tmp_path, real_library, [_rec('s1', *I_M, parameterisation='nope')])
    message = str(error.value)
    assert 'unknown parameterisation "nope"' in message and '"s1"' in message
    assert os.path.join('i_M', 'versions', 'Argus2026_v01') in message
    existing = available_parameterisations(_entry(real_library, I_M)['config_path'])
    assert I_M_NAMED in existing and str(existing) in message


@pytest.mark.unit
def test_a_parameterisation_of_a_config_without_parameterisations_is_an_error(tmp_path, library):
    # synthetic: every real module version has a parameterisations/ directory
    records = [_rec('s', 'pulse_src', 'v2', out=['coll'], parameterisation='default'),
               _rec('coll', 'collector', inp=['s'])]
    with pytest.raises(ValueError,
                       match=r'has no parameterisations/ directory'):
        _load(tmp_path, library, records)


@pytest.mark.unit
def test_a_parameterisation_of_a_type_with_no_config_is_an_error(tmp_path, real_library):
    with pytest.raises(ValueError, match=r'no module config entry of that type'):
        _real_load(tmp_path, real_library, [_rec('s', 'i_M', 'v9', parameterisation='default')])


@pytest.mark.unit
@pytest.mark.parametrize('value', ['', '   ', 3, 'a/b', '..'])
def test_a_bad_parameterisation_name_is_rejected(value):
    with pytest.raises(ValueError, match='parameterisation'):
        normalise_vessel_record(_rec('s', 'pulse_src', 'v1', parameterisation=value), 'test', 0)


@pytest.mark.unit
@pytest.mark.parametrize('entry', [_pulse_entry('v1', default_parameterisation=''),
                                   dict(PAIR_ENTRY, default_parameterisation=['base'])])
def test_a_bad_default_parameterisation_is_rejected(entry):
    with pytest.raises(ValueError, match='default_parameterisation'):
        normalise_module_config_entry(entry, 'test')


@pytest.mark.unit
def test_merge_default_parameters_uses_the_first_of_repeated_extra_rows():
    import numpy as np
    from libcuflynx.parsers.ModelParsers import merge_default_parameters
    dtype = [(c, '<U80') for c in ('variable_name', 'units', 'value', 'data_reference')]
    host = np.array([('a', 'm', '1', 'host')], dtype=dtype)
    merged = merge_default_parameters(host, [
        {'variable_name': 'g', 'units': 'm', 'value': '2', 'data_reference': 'first'},
        {'variable_name': 'g', 'units': 'm', 'value': '3', 'data_reference': 'second'}])
    assert merged['variable_name'].tolist() == ['a', 'g']
    assert merged['value'].tolist() == ['1', '2']


# --------------------------------------------------------------------------------------------
# supermodules
# --------------------------------------------------------------------------------------------

def _file_values(supermodule, instance):
    """``{model name: value}`` of the real ``supermodule``'s parameters file, for the record
    ``instance``: each instance's parameter <var> (or nested <path>/<var>) renamed."""
    out = {}
    for sub, entry in supermodule['instance_parameters'].items():
        for var, parameter in entry['parameters'].items():
            *path, leaf = var.split('/')
            out[f'{leaf}_{"__".join([instance, sub] + path)}'] = float(parameter['value'])
    return out


@pytest.mark.unit
def test_a_supermodule_s_parameters_file_beats_its_submodules_defaults(tmp_path, real_library):
    lib = real_library
    soma = lib.supermodules[SOMA]
    subs = [s['name'] for s in soma['submodules']]
    records, rows = _real_load(tmp_path, lib, [_rec('soma', *SOMA)])
    assert [r['name'] for r in records] == [f'soma__{s}' for s in subs]
    values = {name: float(value) for name, (value, _) in _values(rows).items()}

    paths = submodule_path_parts(soma, lib.supermodules)
    # the supermodule's own parameterisation: its globals (and any <submodule>/<var> rows)
    own = {supermodule_row_name(name, 'soma', paths): float(value) for name, (value, _) in
           _parameterisation_values(lib, SOMA, soma['default_parameterisation']).items()}
    for name, value in own.items():
        assert values[name] == value, name
    in_file = _file_values(soma, 'soma')
    assert in_file, 'soma\'s parameters file sets nothing'
    for name, value in in_file.items():
        assert values[name] == own.get(name, value), name

    # a submodule's version's default fills only what the file leaves unset
    for sub in soma['submodules']:
        key = (sub['module_type'], sub['module_subtype'])
        globals_ = global_constants(_entry(lib, key))
        for name, (value, _) in _parameterisation_values(
                lib, key, _entry(lib, key).get('default_parameterisation')).items():
            model_name = f'{name}_soma__{sub["name"]}'
            if name not in globals_ and model_name not in in_file and model_name not in own:
                assert values[model_name] == float(value), model_name


@pytest.mark.unit
def test_the_heart_s_chambers_keep_their_own_values(tmp_path, real_library):
    """The heart's parameters file: lv started from the chamber's default with its own E_A, and
    ra from the chamber's parameterisation "ra"; the generated rows are the file's."""
    lib = real_library
    heart = lib.supermodules[HEART]
    instances = heart['instance_parameters']
    assert instances['lv']['parameterisation'] == 'default'
    assert instances['ra']['parameterisation'] == 'ra'
    assert instances['lv']['parameters']['E_A']['value'] == 366575000
    _, rows = _real_load(tmp_path, lib, [_rec('heart', *HEART)])
    values = _values(rows)
    assert float(values['E_A_heart__lv'][0]) == 366575000
    assert float(values['E_A_heart__ra'][0]) == instances['ra']['parameters']['E_A']['value']


@pytest.mark.unit
def test_a_nested_supermodule_s_parameters_file_is_renamed_through_the_nesting(tmp_path, real_library):
    lib = real_library
    soma = lib.supermodules[SOMA]
    subs = [s['name'] for s in soma['submodules']]
    neuron = lib.supermodules[NEURON]
    soma_path = next(s['name'] for s in neuron['submodules']
                     if (s['module_type'], s['module_subtype']) == SOMA)
    records, rows = _real_load(tmp_path, lib, [_rec('neuron', *NEURON)])
    names = [r['name'] for r in records]
    assert f'neuron__{soma_path}__{subs[0]}' in names
    values = _values(rows)
    neuron_own = set(_file_values(neuron, 'neuron'))
    checked = 0
    for nested, value in _file_values(soma, f'neuron__{soma_path}').items():
        if nested in neuron_own:
            continue                     # the outer supermodule sets it
        assert float(values[nested][0]) == value, nested
        checked += 1
    assert checked


@pytest.mark.unit
def test_the_instance_s_file_sets_its_submodules_and_their_default_fills_the_rest(
        tmp_path, library):
    records, rows = _load(tmp_path, library, _pair_host())
    assert [r['name'] for r in records] == ['pr__a', 'pr__b', 'coll']
    assert all('parameterisation' not in r for r in records[:2])
    assert _values(rows) == {
        # the supermodule's default parameterisation "base": its global
        'omega': ('5.0', 'base_ref'),
        # the instance's parameters file, written from the version's
        'amp_pr__a': ('0.7', 'file_ref'), 'amp_pr__b': ('0.2', 'file_ref'),
        # what that leaves unset: the submodule version's default parameterisation
        'mean_pr__a': ('1e-05', 'default_ref'), 'mean_pr__b': ('1e-05', 'default_ref'),
    }


@pytest.mark.unit
def test_a_named_supermodule_parameterisation_replaces_its_default_s_globals(tmp_path, library):
    # synthetic: the real supermodules have one parameterisation each
    _, rows = _load(tmp_path, library, _pair_host(parameterisation='alt'))
    assert _values(rows) == {
        'amp_pr__a': ('0.7', 'file_ref'), 'amp_pr__b': ('0.2', 'file_ref'),
        # nothing of base: omega and mean are the submodules' version default's
        'omega': ('6.0', 'default_ref'),
        'mean_pr__a': ('1e-05', 'default_ref'), 'mean_pr__b': ('1e-05', 'default_ref'),
    }


# --------------------------------------------------------------------------------------------
# a model's supermodule instance parameters files
# --------------------------------------------------------------------------------------------

def _instance_file(tmp_path, instance='pr'):
    return supermodule_instance_parameters_path(str(tmp_path / 'm_module_array.json'), instance)


@pytest.mark.unit
def test_generation_reads_the_model_s_instance_file_not_the_version_s(tmp_path, library):
    """The model's copy is what is generated: a value edited there is used, and a value edited
    in the version's parameters file afterwards is not."""
    _load(tmp_path, library, _pair_host())
    path = _instance_file(tmp_path)
    assert path == str(tmp_path / 'm_supermodule_parameters' / 'pr.json')
    with open(path) as f:
        doc = json.load(f)
    doc['instances']['b']['parameters']['amp'] = {'value': 0.45, 'units': 'dimensionless',
                                                  'data_reference': 'model_ref'}
    doc['instances']['a']['parameters']['mean'] = {'value': 3e-05, 'units': 'm3_per_s',
                                                   'data_reference': 'model_ref'}
    with open(path, 'w') as f:
        json.dump(doc, f)
    version_file = os.path.join(library[0], 'systems', 'pair', 'versions', 'v1',
                                'pair_v1_parameters.json')
    original = open(version_file).read()
    try:
        with open(version_file, 'w') as f:
            f.write(original.replace('0.7', '0.11'))
        _, rows = _load(tmp_path, library, _pair_host())
    finally:
        with open(version_file, 'w') as f:
            f.write(original)
    values = _values(rows)
    assert values['amp_pr__b'] == ('0.45', 'model_ref')
    assert values['mean_pr__a'] == ('3e-05', 'model_ref')
    assert values['amp_pr__a'] == ('0.7', 'file_ref')


@pytest.mark.unit
def test_a_supermodule_instance_without_its_file_is_refused_naming_the_command(tmp_path,
                                                                              library):
    with pytest.raises(ValueError, match=r'supermodule instance "pr" .*its parameters file '
                                         r'.*m_supermodule_parameters/pr\.json does not exist\. '
                                         r'.*cuflynx-upgrade supermodule-instances '
                                         r'.*m_module_array\.json'):
        _load(tmp_path, library, _pair_host(), instances=False)


@pytest.mark.unit
def test_a_submodule_row_in_a_supermodule_parameterisation_is_refused(tmp_path, library):
    with pytest.raises(ValueError, match=r'legacy_parameters\.csv: the rows \[\'b/amp\'\] set '
                                         r'submodule parameters\. A supermodule\'s '
                                         r'parameterisation sets only globals.*cuflynx-upgrade '
                                         r'supermodule-instances'):
        _load_legacy(tmp_path, library)


def _load_legacy(tmp_path, library):
    """The pair host on "legacy", its instance file written from the version alone (as one
    written before the refusal would be)."""
    path = str(tmp_path / 'm_module_array.json')
    _write_json(path, _pair_host(parameterisation='legacy'))
    files = _config_files(library)
    supermodules = load_supermodule_registry(files)
    from libcuflynx.utilities.config_schemas import write_supermodule_parameters_file
    from libcuflynx.utilities.supermodules import instance_parameters_from_version
    write_supermodule_parameters_file(_instance_file(tmp_path), instance_parameters_from_version(
        supermodules[('pair', 'v1')], supermodules))
    return load_expanded_vessel_records(path, supermodules, load_component_registry(files))


@pytest.mark.unit
def test_writing_an_instance_moves_its_parameterisation_s_submodule_rows_into_its_file(
        tmp_path, library):
    """cuflynx-upgrade supermodule-instances: a parameterisation's <sub>/<var> rows go over the
    version's values in the new file; an existing file is kept unless replaced."""
    path = str(tmp_path / 'm_module_array.json')
    _write_json(path, _pair_host(parameterisation='legacy'))
    supermodules = load_supermodule_registry(_config_files(library))
    done = write_instance_parameters(path, supermodules)
    assert [(n, a) for n, _, a in done] == [('pr', 'written')]
    with open(_instance_file(tmp_path)) as f:
        doc = json.load(f)
    assert doc['instances']['b']['parameters']['amp'] == {
        'value': 0.1, 'units': 'dimensionless', 'data_reference': 'legacy_ref'}
    assert doc['instances']['a']['parameters']['amp']['value'] == 0.7
    assert [a for _, _, a in write_instance_parameters(path, supermodules)] == ['kept']
    assert [a for _, _, a in write_instance_parameters(path, supermodules, replace=True)] == \
        ['replaced']
    with pytest.raises(ValueError, match=r"\['coll'\] are not supermodule instances there"):
        write_instance_parameters(path, supermodules, instances=['coll'])


@pytest.mark.unit
def test_a_supermodule_row_in_the_older_form_is_refused():
    with pytest.raises(ValueError, match=r'"amp_a" is written \{variable\}_\{submodule\}.*'
                                         r'a/amp.*cuflynx-upgrade parameter-names'):
        supermodule_row_name('amp_a', 'pr', [('a',), ('b',)], where='p.csv: ')
    assert supermodule_row_name('a/amp', 'pr', [('a',), ('b',)]) == 'amp_pr__a'
    assert supermodule_row_name('omega', 'pr', [('a',), ('b',)]) == 'omega'


@pytest.mark.unit
def test_an_unknown_supermodule_parameterisation_lists_its_parameterisations(tmp_path, real_library):
    with pytest.raises(ValueError) as error:
        _real_load(tmp_path, real_library, [_rec('soma', *SOMA, parameterisation='nope')])
    message = str(error.value)
    existing = available_parameterisations(real_library.supermodules[SOMA]['config_path'])
    assert 'unknown parameterisation "nope"' in message and str(existing) in message
    assert os.path.join('soma', 'versions', 'sympathetic') in message


@pytest.mark.unit
@pytest.mark.parametrize('parameterisation', ['other', 'default'])
def test_a_submodule_naming_a_parameterisation_is_refused(tmp_path, parameterisation):
    """Even its version's own default: a submodule's parameters come from the supermodule."""
    config = write_supermodule(str(tmp_path / 'probe'), dict(
        PAIR, module_type='probe', submodules=[_sub('a', parameterisation=parameterisation),
                                               _sub('b')]))
    with pytest.raises(ValueError, match=r"probe_v1_module_array\.json: \['a'\] name a "
                                         r"parameterisation\. A submodule's values are in the "
                                         r"supermodule's parameters file .*probe_v1_parameters"
                                         r"\.json"):
        load_supermodule_registry([config])


# --------------------------------------------------------------------------------------------
# a supermodule's parameters file
# --------------------------------------------------------------------------------------------

def _probe(tmp_path, **entry):
    """A copy of PAIR named ``probe``, written to ``tmp_path/probe``; returns (config, dir)."""
    directory = str(tmp_path / 'probe')
    return write_supermodule(directory, dict(PAIR, module_type='probe', **entry)), directory


def _rewrite_parameters(directory, edit):
    path = os.path.join(directory, 'probe_v1_parameters.json')
    with open(path) as f:
        doc = json.load(f)
    edit(doc)
    with open(path, 'w') as f:
        json.dump(doc, f)


@pytest.mark.unit
def test_a_supermodule_without_its_parameters_file_is_an_error(tmp_path):
    config, directory = _probe(tmp_path)
    os.remove(os.path.join(directory, 'probe_v1_parameters.json'))
    with pytest.raises(ValueError, match=r'its parameters file .*probe_v1_parameters\.json, which '
                                         r'holds the parameter values of its instances, does '
                                         r'not exist'):
        load_supermodule_registry([config])


@pytest.mark.unit
@pytest.mark.parametrize('edit, message', [
    (lambda d: d['instances'].pop('b'), r"it has no entry for \['b'\]"),
    (lambda d: d['instances'].update(c={'parameterisation': 'default', 'parameters': {}}),
     r"\['c'\] are not instances there"),
], ids=['missing', 'unknown'])
def test_a_parameters_file_names_the_module_array_s_instances(tmp_path, edit, message):
    config, directory = _probe(tmp_path)
    _rewrite_parameters(directory, edit)
    with pytest.raises(ValueError, match=r'probe_v1_parameters\.json: its instances must be those '
                                         r'of the module array .*probe_v1_module_array\.json; '
                                         + message):
        load_supermodule_registry([config])


@pytest.mark.unit
@pytest.mark.parametrize('edit, where', [
    (lambda d: d['instances']['a']['parameters']['amp'].pop('units'), r'instances\.a\.parameters'),
    (lambda d: d['instances']['a'].pop('parameterisation'), r'instances\.a'),
    (lambda d: d.pop('schema_version'), r'\(the top level\)'),
], ids=['units', 'parameterisation', 'schema_version'])
def test_a_parameters_file_is_checked_against_its_schema(tmp_path, edit, where):
    pytest.importorskip('jsonschema')
    config, directory = _probe(tmp_path)
    _rewrite_parameters(directory, edit)
    with pytest.raises(ValueError, match=r'probe_v1_parameters\.json: 1 problem with the '
                                         r'supermodule parameters file\n  at ' + where):
        load_supermodule_registry([config])


@pytest.mark.unit
def test_a_supermodule_instance_s_values_are_paths_into_it(tmp_path, library):
    """An outer supermodule over PAIR: its parameters file sets pair's submodule a through the
    path a/amp, over PAIR's own file and its parameterisation."""
    outer = {'module_type': 'outer', 'module_subtype': 'v1', 'module_format': 'supermodule',
             'submodules': [_rec('p', 'pair', 'v1')],
             'instance_parameters': {'p': {'a/amp': (0.3, 'dimensionless', 'outer_ref')}}}
    config = write_supermodule(str(tmp_path / 'outer'), outer)
    files = _config_files(library) + [config]
    path = str(tmp_path / 'm_module_array.json')
    _write_json(path, [_rec('o', 'outer', 'v1')])
    supermodules = load_supermodule_registry(files)
    write_instance_parameters(path, supermodules)
    # the new instance's file holds pair's values too, as paths, under outer's
    with open(_instance_file(tmp_path, 'o')) as f:
        written = json.load(f)['instances']['p']['parameters']
    assert written['a/amp']['value'] == 0.3 and written['b/amp']['value'] == 0.2
    _, rows = load_expanded_vessel_records(path, supermodules, load_component_registry(files))
    assert _values(rows)['amp_o__p__a'] == ('0.3', 'outer_ref')
    assert _values(rows)['amp_o__p__b'] == ('0.2', 'file_ref')

    outer['instance_parameters'] = {'p': {'amp': (0.3, 'dimensionless')}}
    config = write_supermodule(str(tmp_path / 'outer2'), outer)
    supermodules = load_supermodule_registry(_config_files(library) + [config])
    write_instance_parameters(path, supermodules, replace=True)
    with pytest.raises(ValueError, match=r'instance "p" of its parameters file is a supermodule '
                                         r'instance, so its parameter "amp" must be a path into '
                                         r'it'):
        load_expanded_vessel_records(path, supermodules, load_component_registry(files))


# --------------------------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------------------------

@pytest.mark.integration
def test_generation_uses_parameterisations_and_the_host_file_wins(tmp_path, real_library):
    lib = real_library
    named = _parameterisation_values(lib, I_M, I_M_NAMED)
    default = _parameterisation_values(lib, I_M, _entry(lib, I_M)['default_parameterisation'])
    units = {row['variable_name']: row['units'] for row in read_parameter_rows(
        parameterisation_parameters_path(_entry(lib, I_M)['config_path'], I_M_NAMED), 'test')}
    prefix = 'mi_gen'
    resources = tmp_path / 'resources'
    _write_json(str(resources / f'{prefix}_module_array.json'),
                [_rec('s1', *I_M, parameterisation=I_M_NAMED), _rec('s2', *I_M)])
    # host values that differ from every instance's, under the module's own units
    _write_parameters(str(resources / f'{prefix}_parameters.csv'),
                      [('s1/tau_w_num', units['tau_w_num'], '999'),
                       ('T', units['T'], '300')], 'host_value')
    generated = tmp_path / 'generated_models'
    config = {'file_prefix': prefix, 'input_param_file': f'{prefix}_parameters.csv',
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': str(resources),
              'generated_models_dir': str(generated), 'module_library_dirs': [lib.modules],
              'use_builtin_modules': False, 'DEBUG': False}
    assert generate_with_new_architecture(False, config), f'generation of {prefix} failed'
    with open(generated / prefix / f'{prefix}_parameters.csv') as f:
        params = {row['variable_name']: (row['value'], row['data_reference'])
                  for row in csv.DictReader(f)}
    assert params['tau_w_num_s1'] == ('999', 'host_value')
    assert params['T'] == ('300', 'host_value')
    # values only: generation rewrites the library's data_reference text
    assert params['rho_M_s1'][0] == named['rho_M'][0]
    assert params['rho_M_s2'][0] == default['rho_M'][0]
    assert params['tau_w_num_s2'][0] == default['tau_w_num'][0]
    assert 'T_s1' not in params and 'T_s2' not in params


# --------------------------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_the_json_schemas_accept_parameterisations_and_obs_data_names(real_library):
    jsonschema = pytest.importorskip('jsonschema')
    from libcuflynx.schemas import (MODULE_CONFIG_SCHEMA, OBS_DATA_SCHEMA, MODULE_ARRAY_SCHEMA,
                                    load_schema)
    validators = {}
    for name in (MODULE_ARRAY_SCHEMA, MODULE_CONFIG_SCHEMA, OBS_DATA_SCHEMA):
        schema = load_schema(name)
        jsonschema.Draft202012Validator.check_schema(schema)
        validators[name] = jsonschema.Draft202012Validator(schema)
    arrays, configs, obs = (validators[MODULE_ARRAY_SCHEMA], validators[MODULE_CONFIG_SCHEMA],
                            validators[OBS_DATA_SCHEMA])

    # every module config in the library, and every instance obs_data
    for path in real_library.files:
        with open(path) as f:
            configs.validate(json.load(f))
    instance_obs = []
    for directory, _, files in os.walk(real_library.modules):
        if os.path.basename(os.path.dirname(directory)) == 'parameterisations':
            instance_obs += [os.path.join(directory, f) for f in files
                             if f.endswith('_obs_data.json')]
    assert instance_obs, 'the library has no instance obs_data to check'
    for path in instance_obs:
        with open(path) as f:
            obs.validate(json.load(f))

    arrays.validate([_rec('s1', *I_M, parameterisation=I_M_NAMED), _rec('s2', *I_M),
                     _rec('soma', *SOMA, parameterisation='default')])
    for bad in ('', ' ', 'a/b', '..', 3):
        assert not arrays.is_valid([_rec('s', *I_M, parameterisation=bad)]), bad
    assert not configs.is_valid([_pulse_entry('v1', default_parameterisation='a/b')])
    assert not configs.is_valid([_pulse_entry('v1', default_instance='p')])
    assert not configs.is_valid([PAIR])

    protocol = {'pre_times': [0.0], 'sim_times': [[1.0]]}
    obs.validate({'obs_data_name': 'other', 'data_items': [], 'protocol_info': protocol})
    assert not obs.is_valid({'data_items': []})
    assert not obs.is_valid([{'data_item_name': 'x', 'data_type': 'constant', 'unit': 'm',
                              'operands': ['a/x'], 'value': 1.0, 'std': 0.1}])
    assert not obs.is_valid({'obs_data_name': 3, 'data_items': []})
    assert not obs.is_valid({'obs_data_name': '', 'data_items': []})


# --------------------------------------------------------------------------------------------
# obs_data_name
# --------------------------------------------------------------------------------------------

def _obs_doc(**top):
    doc = {'protocol_info': {'pre_times': [0.0], 'sim_times': [[1.0]], 'params_to_change': {}},
           'data_items': [{'data_item_name': 'mean of x', 'data_type': 'constant',
                           'unit': 'dimensionless', 'operands': ['main/x'], 'operation': 'mean',
                           'weight': 1.0, 'value': 1.0, 'std': 0.1}]}
    doc.update(top)
    return doc


def _parse(doc=None, path=None):
    from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser
    return ObsAndParamDataParser().parse_obs_data_json(param_id_obs_path=path, obs_data_dict=doc)


def _name_warnings(caught):
    return [w for w in caught if 'obs_data_name' in str(w.message)]


@pytest.mark.unit
def test_obs_data_name_is_accepted_and_returned():
    assert _parse(_obs_doc(obs_data_name='other'))['obs_data_name'] == 'other'
    parsed = _parse(_obs_doc())
    assert parsed['obs_data_name'] is None and len(parsed['gt_df']) == 1


@pytest.mark.unit
@pytest.mark.parametrize('bad', [3, '', ['other']])
def test_a_bad_obs_data_name_is_rejected(bad):
    with pytest.raises(ValueError, match='obs_data_name'):
        _parse(_obs_doc(obs_data_name=bad))


@pytest.mark.unit
@pytest.mark.parametrize('directory', ['parameterisations'])
@pytest.mark.parametrize('top, warns', [({'obs_data_name': 'other'}, False),
                                        ({'obs_data_name': 'default'}, True),
                                        ({}, True)])
def test_obs_data_in_a_parameterisation_directory_is_checked_against_its_name(tmp_path, top, warns,
                                                                       directory):
    path = str(tmp_path / 'v1' / directory / 'other' / 'other_obs_data.json')
    _write_json(path, _obs_doc(**top))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        parsed = _parse(path=path)
    assert len(parsed['gt_df']) == 1
    found = _name_warnings(caught)
    if warns:
        assert len(found) == 1 and "parameterisation 'other'" in str(found[0].message)
    else:
        assert found == []


@pytest.mark.unit
def test_a_real_parameterisation_s_obs_data_names_it(real_library):
    path = os.path.join(os.path.dirname(parameterisation_parameters_path(
        _entry(real_library, I_M)['config_path'], I_M_NAMED)), f'{I_M_NAMED}_obs_data.json')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert _parse(path=path)['obs_data_name'] == I_M_NAMED
    assert _name_warnings(caught) == []


@pytest.mark.unit
def test_obs_data_outside_a_parameterisation_directory_is_not_checked(tmp_path):
    path = str(tmp_path / 'resources' / 'm_obs_data.json')
    _write_json(path, _obs_doc(obs_data_name='anything'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert _parse(path=path)['obs_data_name'] == 'anything'
    assert _name_warnings(caught) == []


# --------------------------------------------------------------------------------------------
# the older names are refused
# --------------------------------------------------------------------------------------------

@pytest.mark.unit
def test_an_instances_directory_is_refused(tmp_path):
    from libcuflynx.utilities.parameterisations import parameterisations_dir
    config = tmp_path / 'm_v1_modules_config.json'
    config.write_text('[]')
    assert parameterisations_dir(str(config)) == str(tmp_path / 'parameterisations')
    _write_parameters(str(tmp_path / 'instances' / 'p' / 'p_parameters.csv'),
                      [('a', 'm', '1')], 'instances')
    with pytest.raises(ValueError, match='instances has the older name.*rename it'):
        parameterisations_dir(str(config))


@pytest.mark.unit
def test_a_record_names_its_parameterisation_with_parameterisation_only(tmp_path, library):
    normalised = normalise_vessel_record(_rec('s', 'pulse_src', 'v1', parameterisation=' other '),
                                         'test', 0)
    assert normalised['parameterisation'] == 'other' and 'instance' not in normalised
    for record in (_rec('s', 'pulse_src', 'v1', instance='other'),
                   _rec('s', 'pulse_src', 'v1', instance='other', parameterisation='other')):
        with pytest.raises(ValueError, match=r'has "instance".*"parameterisation"; rename'):
            normalise_vessel_record(record, 'test', 0)
    config = write_supermodule(str(tmp_path / 'probe'),
                               dict(PAIR, submodules=[_sub('a', instance='x'), _sub('b')]))
    with pytest.raises(ValueError, match=r'pair_v1_module_array\.json, record 0 \("a"\) has '
                                         r'"instance"'):
        load_supermodule_registry([config])
    records = _two_sources(None, None)
    records[0]['instance'] = 'other'
    with pytest.raises(ValueError, match=r'"s1".*has "instance"'):
        _load(tmp_path, library, records)


@pytest.mark.unit
def test_default_instance_is_refused():
    for entry in (_pulse_entry('v1', default_instance='default'),
                  dict(PAIR_ENTRY, default_instance='base')):
        with pytest.raises(ValueError, match=r"\['default_instance'\].*cuflynx-upgrade "
                                             r"module-config"):
            normalise_module_config_entry(entry, 'lib.json')
    pair = normalise_module_config_entry(PAIR_ENTRY)
    assert pair['default_parameterisation'] == 'base'
    assert pair['submodule_versions'] == [{'module_type': 'pulse_src', 'module_subtype': 'v1'}]
    # a bad name is reported under the key it was given as
    with pytest.raises(ValueError, match=r'"parameterisation" is "a/b"'):
        normalise_vessel_record(_rec('s', 'pulse_src', 'v1', parameterisation='a/b'), 'test', 0)
    with pytest.raises(ValueError, match='default_parameterisation'):
        normalise_module_config_entry(_pulse_entry('v1', default_parameterisation=''))


@pytest.mark.unit
def test_the_json_schemas_require_parameterisation_on_records():
    jsonschema = pytest.importorskip('jsonschema')
    from libcuflynx.schemas import MODULE_CONFIG_SCHEMA, MODULE_ARRAY_SCHEMA, load_schema
    arrays = jsonschema.Draft202012Validator(load_schema(MODULE_ARRAY_SCHEMA))
    configs = jsonschema.Draft202012Validator(load_schema(MODULE_CONFIG_SCHEMA))
    arrays.validate([_rec('s1', 'pulse_src', 'v1', parameterisation='other')])
    configs.validate([_pulse_entry('v1', default_parameterisation='default'), PAIR_ENTRY])
    assert not arrays.is_valid([_rec('s2', 'pulse_src', 'v1', instance='other')])
    assert not configs.is_valid([PAIR])
    for bad in ('', ' ', 'a/b', '..', 3):
        assert not arrays.is_valid([_rec('s', 'pulse_src', 'v1', parameterisation=bad)]), bad
    assert not configs.is_valid([_pulse_entry('v1', default_parameterisation='a/b')])
    assert not configs.is_valid([dict(PAIR_ENTRY, default_parameterisation='')])


# --------------------------------------------------------------------------------------------
# existing models are unchanged
# --------------------------------------------------------------------------------------------

def _resource_prefixes():
    from test_json_module_arrays import RESOURCE_PREFIXES
    return RESOURCE_PREFIXES


def _generate_resource(work_dir, prefix):
    resources = os.path.join(str(work_dir), 'resources')
    os.makedirs(resources, exist_ok=True)
    for suffix in ('_parameters.csv', '_module_array.csv'):
        shutil.copy(os.path.join(RESOURCES_DIR, prefix + suffix), resources)
    config = {'file_prefix': prefix, 'input_param_file': f'{prefix}_parameters.csv',
              'model_type': 'cellml', 'solver': 'CVODE_myokit', 'resources_dir': resources,
              'generated_models_dir': os.path.join(str(work_dir), 'generated_models'),
              'DEBUG': False}
    try:
        ok = generate_with_new_architecture(False, config)
    except (Exception, SystemExit):
        ok = False
    return ok, os.path.join(str(work_dir), 'generated_models', prefix, f'{prefix}.cellml')


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures('skip_generated_model_checks')
@pytest.mark.parametrize('prefix', _resource_prefixes())
def test_resources_models_generate_identically_without_the_parameterisation_lookup(
        tmp_path, monkeypatch, prefix):
    if not os.path.exists(os.path.join(RESOURCES_DIR, f'{prefix}_parameters.csv')):
        pytest.skip(f'{prefix} has no parameters file')
    ok, with_lookup = _generate_resource(tmp_path / 'instances', prefix)
    if not ok:
        pytest.skip(f'{prefix} does not generate')
    # load_model reads the registry through ModuleSources; None means no instance lookup
    from libcuflynx.utilities.module_library import ModuleSources
    monkeypatch.setattr(ModuleSources, 'component_registry', lambda self: None)
    ok, without = _generate_resource(tmp_path / 'plain', prefix)
    assert ok
    _assert_same_generated_models(with_lookup, without)


# --- rows libcuflynx appends itself must fit module arrays with extra columns ("instance", ...) ---

def test_appended_vessel_rows_fill_extra_columns():
    import pandas as pd
    from libcuflynx.parsers.ModelParsers import _append_vessel_row
    df = pd.DataFrame(columns=['name', 'module_subtype', 'module_type', 'inp_instances', 'out_instances', 'instance'])
    _append_vessel_row(df, 'par', 'vp', 'arterial_simple', ['heart'], ['pvn'])
    assert list(df.loc[0]) == ['par', 'vp', 'arterial_simple', ['heart'], ['pvn'], '']
    # a frame with the columns in another order still gets each value in its column
    df2 = pd.DataFrame(columns=['instance', 'name', 'module_subtype', 'module_type', 'inp_instances', 'out_instances'])
    _append_vessel_row(df2, 'volume_sum_1D', 'nn', 'FV1D_volume_sum', '', 'total')
    assert df2.loc[0, 'name'] == 'volume_sum_1D' and df2.loc[0, 'instance'] == ''

def _generate_cpp_1d(work_dir, module_array_name, records=None):
    import contextlib
    import io
    res = os.path.join(work_dir, 'res')
    os.makedirs(res, exist_ok=True)
    shutil.copy(os.path.join(RESOURCES_DIR, 'aortic_bif_1d_parameters.csv'), res)
    if records is None:
        shutil.copy(os.path.join(RESOURCES_DIR, 'aortic_bif_1d_module_array.csv'), res)
    else:
        _write_json(os.path.join(res, module_array_name), records)
    cfg = {'file_prefix': 'aortic_bif_1d', 'input_param_file': 'aortic_bif_1d_parameters.csv',
           'model_type': 'cpp', 'solver': 'RK4', 'couple_to_1d': True, 'resources_dir': res,
           'generated_models_dir': os.path.join(work_dir, 'gen'), 'cpp_generated_models_dir': os.path.join(work_dir, 'cpp'),
           'cpp_1d_model_config_path': None, 'dt': 0.001,
           'solver_info': {'dt_solver': 1e-4, 'MaximumNumberOfSteps': 5000, 'solver': 'RK4'}, 'DEBUG': False}
    with contextlib.redirect_stdout(io.StringIO()):
        assert generate_with_new_architecture(False, cfg)
    with open(os.path.join(work_dir, 'cpp', 'model0d.cc')) as f:
        return f.read()


def test_0d_1d_split_accepts_records_with_extra_keys(tmp_path):
    '''split_0d_1d_module_array appends volume_sum_1D / 1D-coupling rows; with an extra record key
    (such as "instance") a positional 5-value row used to raise "cannot set a row with mismatched
    columns". The generated C++ is the same as from the plain array.'''
    from libcuflynx.utilities.config_schemas import read_module_array_records
    plain = _generate_cpp_1d(str(tmp_path / 'plain'), None)
    records = [dict(r, comment='extra column') for r in
               read_module_array_records(os.path.join(RESOURCES_DIR, 'aortic_bif_1d_module_array.csv'))]
    extra = _generate_cpp_1d(str(tmp_path / 'extra'), 'aortic_bif_1d_module_array.json', records)
    assert extra == plain


def test_an_empty_optional_csv_cell_is_not_set(tmp_path):
    '''An empty "parameterisation" cell in a CSV module array (e.g. the 0D part libcuflynx
    writes itself for the 1D split, where appended rows have none) leaves the key unset rather
    than giving an invalid empty name.'''
    from libcuflynx.utilities.config_schemas import read_module_array_records
    p = tmp_path / 'm_module_array.csv'
    p.write_text('name,module_subtype,module_type,inp_instances,out_instances,parameterisation\n'
                 'a,nn,pulse_src,,b,other\n'
                 'b,nn,volume_sum,a,,\n')
    recs = read_module_array_records(str(p))
    assert recs[0]['parameterisation'] == 'other'
    assert 'parameterisation' not in recs[1]
