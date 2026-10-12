"""Boundary conditions (params_to_change) as emulator inputs -- ``emulated_params_to_change``.

An emulator trained with the protocol fixed has every boundary condition baked in as a hidden
constant. Listing a BC under ``emulator_settings.emulated_params_to_change`` makes it an input
instead, so one emulator serves any obs_data whose BC lies in the trained range.

Three layers, cheapest first:

* the BC spec, values, protocol writes and fingerprint, on plain dicts (``unit``);
* the trainer and the use side over a stub emulator and no solver (``unit``);
* two exemplars with the real library (``integration``/``slow``, need autoemulate): an emulator
  without the BC input is wrong as soon as the BC moves, one with it is not.

The ODE exemplar is ``Simple_ODE_BC`` in ``tests/test_inputs`` -- the benchmark with a BC
multiplying a calibrated parameter::

    dx/dt = -x + p * u_bc      ->  x_ss = p * u_bc
    dy/dt = -3y + q            ->  y_ss = q / 3

``u_bc`` is set by ``params_to_change["benchmark/u_bc"]`` and calibrated ``(p, q)`` come from
``Simple_ODE_BC_params_for_id.csv``. The second exemplar is ``3compartment`` with the terminal
(outflow) resistance ``systemic_T/R_T`` as the BC.
"""
import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from libcuflynx.emulators.bc_inputs import (BCInputError, apply_bc_values, bc_values,
                                            preserved_protocol, resolve_bc_inputs)
from libcuflynx.emulators.emulator_bundle import (EmulatorBoundsError, EmulatorBundle,
                                                  EmulatorQualityError, fingerprint)
from libcuflynx.emulators.emulator_trainer import EmulatorTrainer, resolve_emulator_dir
from libcuflynx.param_id.paramID import CVS0DParamID, emulated_feature_labels
from libcuflynx.parsers.PrimitiveParsers import (ANALYSIS_OPTIONS, ObsAndParamDataParser,
                                                 YamlFileParser, param_entry_labels)

try:
    from mpi4py import MPI
except ImportError:                                       # pragma: no cover - env dependent
    MPI = None

TEST_INPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_inputs')
BC_MODULES = os.path.join(TEST_INPUTS, 'emulator_bc_modules')
BC_KEY = 'benchmark/u_bc'
BC_RANGE = [0.5, 2.0]


# =========================================================================== fixtures

@pytest.fixture
def mpi_comm():
    if MPI is None:
        pytest.skip('mpi4py is required for the emulator training tests')
    return MPI.COMM_WORLD


def _protocol(**extra):
    """A small protocol with one numeric BC, one shape-driven BC and a ramp."""
    protocol = {
        'pre_times': [0.0, 0.0],
        'sim_times': [[2.0], [2.0]],
        'params_to_change': {
            'comp/r': [[1.0], [1.0]],
            'comp/g': [[0.3], [0.3]],
            'comp/stim': [['pulse'], ['pulse']],
            'comp/sweep': [['ramp'], ['ramp']],
        },
        'protocol_shapes': {
            'pulse': {'type': 'pacing',
                      'events': [{'level': 2.0, 'start': 0.1, 'length': 0.2, 'period': 1.0}]},
            'ramp': {'type': 'ramp', 'from': 0.0, 'to': 4.0},
            'unused': {'type': 'ramp', 'from': 0.0, 'to': 1.0, 'duration': 2.0},
        },
    }
    protocol.update(extra)
    return protocol


def _infos():
    param_id_info = {'param_names': [['comp/alpha']], 'param_mins': np.array([0.0]),
                     'param_maxs': np.array([1.0])}
    obs_info = {'operands': [['comp/x']], 'operations': ['max'], 'operation_kwargs': [None],
                'data_types': ['constant'], 'experiment_idxs': [0], 'subexperiment_idxs': [0]}
    return param_id_info, obs_info


def _bc_config(base_user_inputs, tmp_path, obs_path=None, **overrides):
    """A Simple_ODE_BC config. ``tmp_path`` holds outputs, generated model and emulator."""
    config = base_user_inputs.copy()
    config.update({
        'file_prefix': 'Simple_ODE_BC',
        'input_param_file': 'Simple_ODE_BC_parameters.csv',
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'resources_dir': TEST_INPUTS,
        'external_modules_dir': BC_MODULES,
        'param_id_obs_path': obs_path or os.path.join(TEST_INPUTS, 'Simple_ODE_BC_obs_data.json'),
        'params_for_id_path': os.path.join(TEST_INPUTS, 'Simple_ODE_BC_params_for_id.csv'),
        'param_id_output_dir': str(tmp_path / 'param_id_output'),
        'generated_models_dir': str(tmp_path / 'generated_models'),
        'param_id_method': 'genetic_algorithm',
        'pre_time': 0.0,
        'sim_time': 8.0,
        'dt': 0.05,
        'DEBUG': True,
        'do_uq': False,
        'do_ia': False,
        'plot_predictions': False,
        'solver_info': {'MaximumStep': 0.01, 'MaximumNumberOfSteps': 5000},
        'emulator_settings': {'emulator_dir': str(tmp_path / 'emulator'), 'min_r2': 0.5},
    })
    config.update(overrides)
    return YamlFileParser().parse_user_inputs_file(
        config, obs_path_needed=True, do_generation_with_fit_parameters=False)


def _obs_with_bc(tmp_path, value, name='obs'):
    """The Simple_ODE_BC obs_data with ``u_bc`` set to ``value`` (a scalar or a per-exp list)."""
    with open(os.path.join(TEST_INPUTS, 'Simple_ODE_BC_obs_data.json')) as file:
        obs = json.load(file)
    obs['protocol_info']['params_to_change'][BC_KEY] = (
        value if isinstance(value, list) else [[float(value)]])
    path = tmp_path / f'Simple_ODE_BC_{name}.json'
    path.write_text(json.dumps(obs))
    return str(path)


def _parse_infos(config, scratch):
    os.makedirs(scratch, exist_ok=True)
    parser = ObsAndParamDataParser()
    parsed = parser.parse_obs_data_json(
        param_id_obs_path=config['param_id_obs_path'], pre_time=config['pre_time'],
        sim_time=config['sim_time'], model_type=config['model_type'],
        method=config['solver_info'].get('method'))
    obs_info = parser.process_obs_info(gt_df=parsed['gt_df'], output_dir=scratch,
                                       dt=config['dt'])
    protocol_info = parser.process_protocol_and_weights(
        gt_df=parsed['gt_df'], protocol_info=parsed['protocol_info'], dt=config['dt'])
    return obs_info, protocol_info, parser.get_param_id_info(config['params_for_id_path'])


class _AnalyticFit:
    """A 'fitted emulator' that is exactly the benchmark: inputs (p, q, u) in the unit box."""

    def __init__(self, x_shift, x_span):
        self.x_shift = np.asarray(x_shift, dtype=float)
        self.x_span = np.asarray(x_span, dtype=float)

    def predict(self, x_scaled):
        x = np.asarray(x_scaled, dtype=float) * self.x_span + self.x_shift
        return np.column_stack([x[:, 0] * x[:, 2], x[:, 1] / 3.0])


def _write_bc_bundle(config, bc_range=BC_RANGE):
    """A stub bundle over (p, q, u_bc), the way training with the BC as an input writes it."""
    directory = config['emulator_settings']['emulator_dir']
    obs_info, protocol_info, param_id_info = _parse_infos(config, os.path.join(directory, '_p'))
    bc_inputs = resolve_bc_inputs(protocol_info, {BC_KEY: {'range': bc_range}})
    mins = list(param_id_info['param_mins']) + [bc_range[0]]
    maxs = list(param_id_info['param_maxs']) + [bc_range[1]]
    labels = emulated_feature_labels(obs_info)
    shift, span = np.asarray(mins, dtype=float), np.asarray(maxs) - np.asarray(mins)
    meta = {
        'param_entry_labels': list(param_entry_labels(param_id_info)) + [BC_KEY],
        'param_mins': [float(v) for v in mins],
        'param_maxs': [float(v) for v in maxs],
        'bc_inputs': bc_inputs,
        'param_names': [[str(n) for n in entry] for entry in param_id_info['param_names']],
        'param_defaults': {str(n): 1.0 for entry in param_id_info['param_names'] for n in entry},
        'feature_labels': labels,
        'feature_r2': [0.99] * len(labels),
        'feature_rmse': [0.01] * len(labels),
        'x_scale': {'shift': shift.tolist(), 'span': span.tolist()},
        'y_scale': {'shift': [0.0] * len(labels), 'span': [1.0] * len(labels)},
        'fingerprint': fingerprint(param_id_info, obs_info, protocol_info,
                                   bc_inputs=bc_inputs),
    }
    bundle = EmulatorBundle(_AnalyticFit(shift, span), meta)
    bundle.save(directory)
    return bundle


# =========================================================================== the spec

@pytest.mark.unit
def test_a_params_to_change_key_is_one_input_shared_by_every_experiment():
    inputs = resolve_bc_inputs(_protocol(), {'comp/r': {'range': [0.5, 1.5]}})
    assert len(inputs) == 1, 'one input per BC, not one per (experiment, sub-experiment)'
    item = inputs[0]
    assert (item['label'], item['kind'], item['min'], item['max']) == ('comp/r', 'value', 0.5, 1.5)


@pytest.mark.unit
def test_only_listed_bcs_become_inputs_in_the_order_listed():
    emulated = {'ramp.to': {'range': [3.0, 5.0]},
                'comp/g': {'range': [0.1, 0.5]},
                'pulse.events[0].level': {'range': [1.0, 3.0]}}
    inputs = resolve_bc_inputs(_protocol(), emulated)
    assert [item['label'] for item in inputs] == list(emulated)
    assert [item['kind'] for item in inputs] == ['ramp', 'value', 'event']
    assert resolve_bc_inputs(_protocol(), None) == []
    assert resolve_bc_inputs(_protocol(), {}) == []


@pytest.mark.unit
@pytest.mark.parametrize('emulated, message', [
    ({'comp/nope': {'range': [0, 1]}}, 'names neither'),
    ({'comp/r': [0.5, 1.5]}, 'needs a range'),
    ({'comp/r': {'range': [1.5, 0.5]}}, 'min < max'),
    ({'comp/r': {'range': [0.5, 1.5], 'per_experiment': True}}, 'unknown setting'),
    ({'comp/stim': {'range': [0, 1]}}, 'no single number'),
    ({'unused.to': {'range': [0, 1]}}, 'not used by any'),
    ({'pulse.events[0].multiplier': {'range': [0, 1]}}, 'cannot be varied'),
    ({'pulse.events[3].level': {'range': [0, 1]}}, 'no event 3'),
    ({'pulse.from': {'range': [0, 1]}}, 'not a ramp'),
    ({'ramp.events[0].level': {'range': [0, 1]}}, 'is a ramp'),
])
def test_an_emulated_bc_that_cannot_be_an_input_is_refused_by_name(emulated, message):
    with pytest.raises(BCInputError, match=message):
        resolve_bc_inputs(_protocol(), emulated)


@pytest.mark.unit
def test_a_bc_that_differs_between_experiments_is_refused():
    """One shared input cannot hold two values; per-experiment BCs are not supported yet."""
    protocol = _protocol()
    protocol['params_to_change']['comp/r'] = [[1.0], [2.0]]
    with pytest.raises(BCInputError, match='different values across experiments'):
        resolve_bc_inputs(protocol, {'comp/r': {'range': [0.5, 2.5]}})


@pytest.mark.unit
def test_bc_values_reads_numbers_shape_fields_and_the_duration_alias():
    protocol = _protocol()
    protocol['protocol_shapes']['pulse']['events'][0]['duration'] = (
        protocol['protocol_shapes']['pulse']['events'][0].pop('length'))
    inputs = resolve_bc_inputs(protocol, {'comp/g': {'range': [0, 1]},
                                          'pulse.events[0].length': {'range': [0.1, 0.3]},
                                          'ramp.to': {'range': [3, 5]}})
    assert bc_values(protocol, inputs) == pytest.approx([0.3, 0.2, 4.0])


# =========================================================================== protocol writes

@pytest.mark.unit
def test_apply_writes_every_leaf_and_re_expands_the_touched_shape():
    from libcuflynx.utilities.protocol_shapes import materialise_shapes

    protocol = _protocol()
    materialise_shapes(protocol)
    before = copy.deepcopy(protocol['protocol_traces']['pulse'])
    inputs = resolve_bc_inputs(protocol, {'comp/r': {'range': [0.5, 1.5]},
                                          'pulse.events[0].level': {'range': [1, 5]}})
    apply_bc_values(protocol, inputs, [1.25, 4.0])

    assert protocol['params_to_change']['comp/r'] == [[1.25], [1.25]]
    assert protocol['protocol_shapes']['pulse']['events'][0]['level'] == 4.0
    after = protocol['protocol_traces']['pulse']
    assert max(after['values']) == pytest.approx(4.0), 'the trace must follow the new level'
    assert max(before['values']) == pytest.approx(2.0)


@pytest.mark.unit
def test_preserved_protocol_puts_the_obs_data_back_in_place():
    """The solver helper holds the very dict, so it must be restored, not replaced."""
    protocol = _protocol()
    params_to_change = protocol['params_to_change']
    inputs = resolve_bc_inputs(protocol, {'comp/r': {'range': [0.5, 1.5]},
                                          'ramp.to': {'range': [3, 5]}})
    with preserved_protocol(protocol):
        apply_bc_values(protocol, inputs, [0.7, 3.3])
        assert protocol['params_to_change']['comp/r'] == [[0.7], [0.7]]
    assert protocol['params_to_change'] is params_to_change
    assert protocol['params_to_change']['comp/r'] == [[1.0], [1.0]]
    assert protocol['protocol_shapes']['ramp']['to'] == 4.0
    assert 'protocol_traces' not in protocol


# =========================================================================== fingerprint

@pytest.mark.unit
def test_a_fixed_bc_value_is_still_part_of_the_fingerprint():
    """The emulator was trained *at* a BC it does not take as input; changing it is stale."""
    param_id_info, obs_info = _infos()
    base = fingerprint(param_id_info, obs_info, _protocol())
    changed = _protocol()
    changed['params_to_change']['comp/r'] = [[2.0], [2.0]]
    assert fingerprint(param_id_info, obs_info, changed) != base

    # ... and still is when some *other* BC is an input.
    inputs = resolve_bc_inputs(_protocol(), {'comp/g': {'range': [0.1, 0.5]}})
    assert (fingerprint(param_id_info, obs_info, changed, bc_inputs=inputs)
            != fingerprint(param_id_info, obs_info, _protocol(), bc_inputs=inputs))


@pytest.mark.unit
def test_an_emulated_bc_value_is_not_part_of_the_fingerprint_but_its_range_is():
    param_id_info, obs_info = _infos()
    emulated = {'comp/r': {'range': [0.5, 2.5]}, 'pulse.events[0].level': {'range': [1, 3]}}
    inputs = resolve_bc_inputs(_protocol(), emulated)
    base = fingerprint(param_id_info, obs_info, _protocol(), bc_inputs=inputs)

    revalued = _protocol()
    apply_bc_values(revalued, inputs, [2.0, 2.9])
    assert fingerprint(param_id_info, obs_info, revalued, bc_inputs=inputs) == base

    widened = resolve_bc_inputs(_protocol(), dict(emulated, **{'comp/r': {'range': [0.5, 3]}}))
    assert fingerprint(param_id_info, obs_info, _protocol(), bc_inputs=widened) != base


@pytest.mark.unit
def test_without_bc_inputs_the_fingerprint_is_what_it_always_was():
    """No upgrade may report every existing emulator stale."""
    param_id_info, obs_info = _infos()
    assert (fingerprint(param_id_info, obs_info, _protocol(), bc_inputs=[])
            == fingerprint(param_id_info, obs_info, _protocol()))


# =========================================================================== settings

@pytest.mark.unit
def test_the_setting_is_in_the_schema():
    names = {d['name']: d for d in ANALYSIS_OPTIONS['emulation']['options']}
    assert names['emulated_params_to_change']['type'] == 'dict'
    assert 'vary_params_to_change' not in names and 'params_to_change_ranges' not in names


@pytest.mark.unit
@pytest.mark.parametrize('settings, message', [
    ({'emulated_params_to_change': ['comp/r']}, 'must map each boundary condition'),
    ({'vary_params_to_change': True}, 'replaced by emulated_params_to_change'),
    ({'params_to_change_ranges': {'comp/r': [[[0, 1]]]}}, 'replaced by emulated_params_to_change'),
])
def test_malformed_or_retired_settings_are_refused(base_user_inputs, tmp_path, settings, message):
    with pytest.raises(ValueError, match=message):
        _bc_config(base_user_inputs, tmp_path, emulator_settings=settings)


# =========================================================================== trainer (stub)

def _stub_trainer(protocol_info, emulated, num_train_samples=8):
    param_id_info, obs_info = _infos()
    pid = SimpleNamespace(sim_helper=SimpleNamespace(emulates_features=False),
                          obs_info=obs_info, protocol_info=protocol_info,
                          param_id_info=param_id_info, model_path=None)
    return EmulatorTrainer(pid, {'emulated_params_to_change': emulated,
                                 'num_train_samples': num_train_samples, 'random_seed': 0})


@pytest.mark.unit
def test_the_design_covers_theta_then_every_bc_range():
    emulated = {'comp/r': {'range': [0.5, 1.5]}, 'ramp.to': {'range': [3.0, 5.0]}}
    trainer = _stub_trainer(_protocol(), emulated, num_train_samples=32)
    assert trainer.input_labels() == ['comp/alpha', 'comp/r', 'ramp.to']
    mins, maxs = trainer.input_box()
    assert mins.tolist() == [0.0, 0.5, 3.0] and maxs.tolist() == [1.0, 1.5, 5.0]

    design = trainer.design()
    assert design.shape == (32, 3)
    assert np.all(design >= mins) and np.all(design <= maxs)
    # Space-filling in the BC columns too, not pinned at the obs_data value.
    assert np.ptp(design[:, 1]) > 0.5 and np.ptp(design[:, 2]) > 1.0


@pytest.mark.unit
def test_every_sample_is_simulated_at_its_own_bc_and_the_protocol_is_restored(monkeypatch):
    import libcuflynx.param_id.fd_backend as fd_backend

    protocol = _protocol()
    trainer = _stub_trainer(protocol, {'comp/r': {'range': [0.5, 1.5]},
                                       'pulse.events[0].level': {'range': [1.0, 3.0]}})
    seen = []

    def fake_features(pid, theta):
        seen.append((float(theta[0]), pid.protocol_info['params_to_change']['comp/r'][1][0],
                     max(pid.protocol_info['protocol_traces']['pulse']['values'])))
        return np.array([theta[0]])

    monkeypatch.setattr(fd_backend, 'observable_features', fake_features)
    design = trainer.design()
    x, y = trainer.evaluate(design)

    assert np.allclose([s[0] for s in seen], design[:, 0]), 'theta excludes the BC columns'
    assert np.allclose([s[1] for s in seen], design[:, 1]), 'every experiment gets the BC'
    assert np.allclose([s[2] for s in seen], design[:, 2]), 'shape traces are re-expanded'
    assert np.array_equal(x, design)
    assert protocol['params_to_change']['comp/r'] == [[1.0], [1.0]]
    assert protocol['protocol_shapes']['pulse']['events'][0]['level'] == 2.0


# =========================================================================== use side (stub)

@pytest.mark.unit
def test_calibration_hands_the_emulator_theta_plus_the_obs_data_bc(base_user_inputs, tmp_path):
    train_config = _bc_config(base_user_inputs, tmp_path)
    _write_bc_bundle(train_config)

    # A different obs_data, BC moved inside the trained range: same emulator, no retraining.
    config = _bc_config(base_user_inputs, tmp_path, obs_path=_obs_with_bc(tmp_path, 1.8),
                        use_emulator=True)
    engine = CVS0DParamID.init_from_dict(config).param_id
    theta = np.array([1.5, 1.2])
    assert np.isfinite(engine.get_cost_from_params(theta))
    helper = engine.sim_helper
    assert helper._theta.tolist() == pytest.approx([1.5, 1.2, 1.8])
    assert helper._features == pytest.approx([1.5 * 1.8, 1.2 / 3.0])

    # The ensemble path (MCMC) appends the same BC to every walker.
    helper.predict_ensemble(np.array([[1.0, 1.0], [2.0, 0.6]]))
    assert helper._ensemble_thetas[:, 2].tolist() == [1.8, 1.8]


@pytest.mark.unit
def test_a_bc_outside_the_trained_range_is_refused(base_user_inputs, tmp_path):
    _write_bc_bundle(_bc_config(base_user_inputs, tmp_path))
    config = _bc_config(base_user_inputs, tmp_path, obs_path=_obs_with_bc(tmp_path, 2.5),
                        use_emulator=True)
    engine = CVS0DParamID.init_from_dict(config).param_id
    with pytest.raises(EmulatorBoundsError, match=BC_KEY):
        engine.sim_helper.bundle.predict(engine.sim_helper._full_input(np.array([1.0, 1.0])))


@pytest.mark.unit
def test_an_obs_data_giving_the_bc_per_experiment_values_is_refused(base_user_inputs, tmp_path):
    with open(os.path.join(TEST_INPUTS, 'Simple_ODE_BC_obs_data.json')) as file:
        obs = json.load(file)
    info = obs['protocol_info']
    info['pre_times'], info['sim_times'] = [0.0, 0.0], [[8.0], [8.0]]
    info['experiment_labels'] = ['a', 'b']

    def write(values, name):
        info['params_to_change'][BC_KEY] = values
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(obs))
        return str(path)

    # Trained on two experiments sharing one BC value -- fine ...
    _write_bc_bundle(_bc_config(base_user_inputs, tmp_path,
                                obs_path=write([[1.0], [1.0]], 'shared')))
    # ... used on the same two experiments with different values -- not representable.
    config = _bc_config(base_user_inputs, tmp_path, obs_path=write([[1.0], [1.5]], 'split'),
                        use_emulator=True)
    with pytest.raises(BCInputError, match='different values across experiments'):
        CVS0DParamID.init_from_dict(config)


@pytest.mark.unit
def test_the_helper_refuses_to_guess_a_bc_it_was_never_given(base_user_inputs, tmp_path):
    from libcuflynx.solver_wrappers.emulator_solver_helper import SimulationHelper

    bundle = _write_bc_bundle(_bc_config(base_user_inputs, tmp_path))
    helper = SimulationHelper(None, bundle=bundle)
    with pytest.raises(RuntimeError, match='set_bc_values'):
        helper.set_theta([1.0, 1.0])
    helper.set_bc_values([1.2])
    helper.set_theta([1.0, 1.0])
    assert helper._theta.tolist() == [1.0, 1.0, 1.2]


# =========================================================================== exemplar: ODE

def _relative_error(predicted, truth):
    return np.abs(np.asarray(predicted) - np.asarray(truth)) / np.abs(np.asarray(truth))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_the_bc_input_is_what_makes_the_emulator_right_at_a_new_bc(
        base_user_inputs, temp_output_dir, mpi_comm):
    """Two emulators, same model, same samples budget -- one with ``u_bc`` as an input.

    Both are trained on an obs_data with ``u_bc = 1``. At that BC both are accurate. Moved to
    ``u_bc = 1.8`` (a new patient or protocol) the one without the input still answers
    ``x_ss = p * 1`` -- 44% off -- and the fingerprint refuses it, which is the right outcome
    when the alternative is that. The one with the input stays accurate and is accepted.
    """
    pytest.importorskip('autoemulate')
    from libcuflynx.param_id.fd_backend import observable_features
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture

    # Kept after the run (tests/test_outputs/<test id>/) so the plot and emulators can be read.
    root = __import__('pathlib').Path(temp_output_dir)
    settings = {'num_train_samples': 64, 'sample_type': 'sobol', 'random_seed': 0,
                'min_r2': 0.95, 'n_iter': 2, 'n_splits': 2}
    fixed_cfg = _bc_config(base_user_inputs, root, do_emulation=True, emulator_settings=dict(
        settings, emulator_dir=str(root / 'emulator_fixed')))
    bc_cfg = _bc_config(base_user_inputs, root, do_emulation=True, emulator_settings=dict(
        settings, emulator_dir=str(root / 'emulator_bc'),
        emulated_params_to_change={BC_KEY: {'range': BC_RANGE}}))

    if mpi_comm.Get_rank() == 0:
        assert generate_with_new_architecture(False, fixed_cfg)
    mpi_comm.Barrier()
    fixed = EmulatorTrainer.init_from_dict(fixed_cfg, comm=mpi_comm).train()
    trainer = EmulatorTrainer.init_from_dict(bc_cfg, comm=mpi_comm)
    with_bc = trainer.train()
    mpi_comm.Barrier()
    if mpi_comm.Get_rank() != 0:
        return

    assert with_bc.param_entry_labels == ['benchmark/p', 'benchmark/q', BC_KEY]
    assert with_bc.meta['param_mins'][-1] == BC_RANGE[0]
    assert with_bc.meta['param_maxs'][-1] == BC_RANGE[1]
    assert trainer.pid.protocol_info['params_to_change'][BC_KEY] == [[1.0]], \
        'training must leave the obs_data protocol as it found it'

    pid = trainer.pid
    rng = np.random.default_rng(7)
    mins = np.asarray(pid.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(pid.param_id_info['param_maxs'], dtype=float)
    thetas = mins + rng.random((10, mins.size)) * (maxs - mins)

    errors = {}
    for u_bc in (1.0, 1.8):
        truth = []
        with preserved_protocol(pid.protocol_info):
            pid.protocol_info['params_to_change'][BC_KEY] = [[u_bc]]
            for theta in thetas:
                truth.append(observable_features(pid, theta))
        truth = np.asarray(truth)
        pred_fixed = fixed.predict(thetas)
        pred_bc = with_bc.predict(np.column_stack([thetas, np.full(len(thetas), u_bc)]))
        errors[u_bc] = {'fixed': _relative_error(pred_fixed, truth),
                        'bc': _relative_error(pred_bc, truth),
                        'truth': truth, 'pred_fixed': pred_fixed, 'pred_bc': pred_bc}

    # Where it was trained, the fixed-BC emulator is fine -- the comparison is fair.
    assert errors[1.0]['fixed'].max() < 0.03, errors[1.0]['fixed']
    assert errors[1.0]['bc'].max() < 0.03, errors[1.0]['bc']
    # At the new BC, x_ss (column 0) is wrong by ~44% without the input, ~exact with it ...
    assert np.median(errors[1.8]['fixed'][:, 0]) > 0.3, errors[1.8]['fixed'][:, 0]
    assert errors[1.8]['bc'].max() < 0.03, errors[1.8]['bc']
    # ... and y_ss, which does not depend on the BC, is right either way.
    assert errors[1.8]['fixed'][:, 1].max() < 0.03

    _parity_plot(errors[1.8], root / 'bc_input_parity_ode.png',
                 feature_names=[r'$x_{ss} = p\,u_{bc}$', r'$y_{ss} = q/3$'],
                 bc_text=r'$u_{bc}$ = 1.8')

    # Through the front door: a run on the new obs_data.
    new_obs = _obs_with_bc(root, 1.8, name='u18')
    use_bc = _bc_config(base_user_inputs, root, obs_path=new_obs, use_emulator=True,
                        emulator_settings={'emulator_dir': str(root / 'emulator_bc')})
    engine = CVS0DParamID.init_from_dict(use_bc).param_id
    assert np.isfinite(engine.get_cost_from_params(thetas[0]))
    assert engine.sim_helper._theta[-1] == pytest.approx(1.8)
    use_fixed = _bc_config(base_user_inputs, root, obs_path=new_obs, use_emulator=True,
                           emulator_settings={'emulator_dir': str(root / 'emulator_fixed')})
    with pytest.raises(EmulatorQualityError, match='stale'):
        CVS0DParamID.init_from_dict(use_fixed)


def _parity_plot(result, path, feature_names, bc_text):
    """Predicted vs simulated at the new BC, without and with the BC input, side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    truth = result['truth']
    n = truth.shape[1]
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for i, ax in enumerate(axes.flat[:n]):
        lo, hi = truth[:, i].min(), truth[:, i].max()
        pad = 0.1 * (hi - lo or 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='0.6', lw=1, label='y = x')
        ax.scatter(truth[:, i], result['pred_fixed'][:, i], marker='x',
                   label=f'BC fixed (max err {result["fixed"][:, i].max():.1%})')
        ax.scatter(truth[:, i], result['pred_bc'][:, i], facecolors='none', edgecolors='C1',
                   label=f'BC as input (max err {result["bc"][:, i].max():.1%})')
        ax.set_xlabel('simulated')
        ax.set_ylabel('emulator')
        ax.set_title(f'{feature_names[i]}\nat {bc_text}'.replace('_{', '\x00').replace('_', ' ')
                     .replace('\x00', '_{'))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'[bc exemplar] parity plot: {path}')
    assert os.path.isfile(path)


# =========================================================================== exemplar: 3compartment

R_T_KEY = 'systemic_T/R_T'
R_T_NOMINAL = 1.1e8


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mpi
def test_terminal_resistance_as_an_input_on_the_3compartment_model(
        base_user_inputs, resources_dir, temp_output_dir, mpi_comm):
    """The same comparison on a circulation model, with the outflow resistance as the BC.

    ``R_T`` is the archetypal boundary condition of a 0D network. Moving it 30% shifts mean
    aortic pressure ~15% and flows ~10%, which an emulator trained at the nominal ``R_T`` cannot
    know about. The response is nonlinear, so the assertion is relative: with ``R_T`` as an
    input the error at the new BC is several times smaller than without.
    """
    pytest.importorskip('autoemulate')
    from libcuflynx.param_id.fd_backend import observable_features
    from libcuflynx.scripts.script_generate_with_new_architecture import \
        generate_with_new_architecture

    root = temp_output_dir

    def config(**settings):
        cfg = base_user_inputs.copy()
        cfg.update({
            'file_prefix': '3compartment',
            'input_param_file': '3compartment_parameters.csv',
            'resources_dir': resources_dir,
            'param_id_obs_path': os.path.join(TEST_INPUTS, '3compartment_bc_obs_data.json'),
            'params_for_id_path': os.path.join(resources_dir, '3compartment_params_for_id.csv'),
            'param_id_output_dir': os.path.join(root, 'param_id_output'),
            'generated_models_dir': os.path.join(root, 'generated_models'),
            'param_id_method': 'genetic_algorithm',
            'DEBUG': True,
            'do_emulation': True,
            'plot_predictions': False,
            'emulator_settings': dict({'num_train_samples': 128, 'sample_type': 'sobol',
                                       'random_seed': 0, 'min_r2': 0.5, 'n_iter': 2,
                                       'n_splits': 2}, **settings),
        })
        return YamlFileParser().parse_user_inputs_file(
            cfg, obs_path_needed=True, do_generation_with_fit_parameters=False)

    fixed_cfg = config(emulator_dir=os.path.join(root, 'emulator_fixed'))
    bc_cfg = config(emulator_dir=os.path.join(root, 'emulator_bc'),
                    emulated_params_to_change={
                        R_T_KEY: {'range': [0.8 * R_T_NOMINAL, 1.4 * R_T_NOMINAL]}})
    if mpi_comm.Get_rank() == 0:
        assert generate_with_new_architecture(False, fixed_cfg)
    mpi_comm.Barrier()
    fixed = EmulatorTrainer.init_from_dict(fixed_cfg, comm=mpi_comm).train()
    trainer = EmulatorTrainer.init_from_dict(bc_cfg, comm=mpi_comm)
    with_bc = trainer.train()
    mpi_comm.Barrier()
    if mpi_comm.Get_rank() != 0:
        return

    pid = trainer.pid
    mins = np.asarray(pid.param_id_info['param_mins'], dtype=float)
    maxs = np.asarray(pid.param_id_info['param_maxs'], dtype=float)
    # The middle half of the box: this compares BC handling, not edge extrapolation.
    rng = np.random.default_rng(3)
    thetas = mins + (0.25 + 0.5 * rng.random((8, mins.size))) * (maxs - mins)
    new_r = 1.3 * R_T_NOMINAL
    truth = []
    with preserved_protocol(pid.protocol_info):
        pid.protocol_info['params_to_change'][R_T_KEY] = [[new_r]]
        for theta in thetas:
            truth.append(observable_features(pid, theta))
    truth = np.asarray(truth)
    result = {'truth': truth, 'pred_fixed': fixed.predict(thetas),
              'pred_bc': with_bc.predict(np.column_stack([thetas, np.full(len(thetas), new_r)]))}
    result['fixed'] = _relative_error(result['pred_fixed'], truth)
    result['bc'] = _relative_error(result['pred_bc'], truth)

    fixed_err, bc_err = float(np.mean(result['fixed'])), float(np.mean(result['bc']))
    print(f'[bc exemplar 3compartment] mean relative error at R_T = 1.3 x nominal: '
          f'BC fixed {fixed_err:.2%}, BC as input {bc_err:.2%}')
    assert fixed_err > 0.04, 'R_T should visibly move the features, or this proves nothing'
    assert bc_err * 4 < fixed_err, (fixed_err, bc_err)

    labels = [label.split('/')[-1] for label in with_bc.feature_labels]
    _parity_plot(result, os.path.join(root, 'bc_input_parity_3compartment.png'),
                 feature_names=labels, bc_text=r'$R_{T}$ = 1.3 x nominal')
