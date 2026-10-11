"""BE_batched_c: kernels from libCellML's analysis (kernel_ir), their C (generators/cpp/batched) and the solver.

The models are small CellML 2.0 files written here, so every test is quick:
- ``OPS``: one block of two states through most MathML operations and algebraic variables, for the exact Jacobian;
- ``CELLS``: two unconnected cells (one kernel, two blocks) and a coupled oscillator (another kernel, one block),
  with known solutions x = x0 exp(-k t) and p = cos(w t).
"""
import math
import os
import shutil

import numpy as np
import pytest

from libcuflynx.generators import kernel_ir

MATHML = 'xmlns="http://www.w3.org/1998/Math/MathML" xmlns:cellml="http://www.cellml.org/cellml/2.0#"'


def _cellml(components, units=''):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<model xmlns="http://www.cellml.org/cellml/2.0#" xmlns:cellml="http://www.cellml.org/cellml/2.0#" name="m">
  <units name="per_s"><unit units="second" exponent="-1"/></units>
  {units}
  {components}
</model>
"""


def _ode(state, rhs):
    return (f'<apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>{state}</ci></apply>{rhs}</apply>')


def _cn(v, units='dimensionless'):
    return f'<cn cellml:units="{units}">{v}</cn>'


def _cell(name, k, x0):
    return f"""
  <component name="{name}">
    <variable name="t" units="second" interface="public"/>
    <variable name="x0" units="dimensionless" initial_value="{x0}"/>
    <variable name="x" units="dimensionless" initial_value="x0"/>
    <variable name="k" units="per_s" initial_value="{k}"/>
    <variable name="flux" units="per_s"/>
    <math {MATHML}>
      <apply><eq/><ci>flux</ci><apply><times/><ci>k</ci><ci>x</ci></apply></apply>
      {_ode('x', '<apply><minus/><ci>flux</ci></apply>')}
    </math>
  </component>"""


CELLS = _cellml(_cell('cell_a', 0.5, 2.0) + _cell('cell_b', 0.2, 1.0) + f"""
  <component name="pair">
    <variable name="t" units="second" interface="public"/>
    <variable name="p" units="dimensionless" initial_value="1"/>
    <variable name="q" units="per_s" initial_value="0"/>
    <variable name="w" units="per_s" initial_value="3"/>
    <math {MATHML}>
      {_ode('p', '<ci>q</ci>')}
      {_ode('q', '<apply><times/><apply><minus/><ci>w</ci></apply><ci>w</ci><ci>p</ci></apply>')}
    </math>
  </component>
  <component name="environment"><variable name="t" units="second" interface="public"/></component>
  <connection component_1="environment" component_2="cell_a"><map_variables variable_1="t" variable_2="t"/></connection>
  <connection component_1="environment" component_2="cell_b"><map_variables variable_1="t" variable_2="t"/></connection>
  <connection component_1="environment" component_2="pair"><map_variables variable_1="t" variable_2="t"/></connection>
""", units='<units name="per_s2"><unit units="second" exponent="-2"/></units>')

# Dimensionless throughout (time too), so every operation is allowed on every variable. The piecewise w is
# continuous at x = 1, and the rates keep x above -2 (where sqrt(x + 2) would end the solution).
OPS = _cellml(f"""
  <component name="ops">
    <variable name="t" units="dimensionless"/>
    <variable name="x" units="dimensionless" initial_value="0.7"/>
    <variable name="y" units="dimensionless" initial_value="1.3"/>
    <variable name="a" units="dimensionless" initial_value="0.8"/>
    <variable name="b" units="dimensionless" initial_value="1.5"/>
    <variable name="u" units="dimensionless"/>
    <variable name="v" units="dimensionless"/>
    <variable name="w" units="dimensionless"/>
    <math {MATHML}>
      <apply><eq/><ci>u</ci><apply><divide/><apply><times/><ci>a</ci><ci>x</ci><ci>y</ci></apply>
        <apply><plus/>{_cn(1)}<apply><power/><ci>y</ci>{_cn(2)}</apply></apply></apply></apply>
      <apply><eq/><ci>v</ci><apply><plus/><apply><exp/><apply><minus/><ci>u</ci></apply></apply>
        <apply><root/><apply><plus/><ci>x</ci>{_cn(2)}</apply></apply>
        <apply><ln/><apply><plus/>{_cn(1)}<apply><times/><ci>y</ci><ci>y</ci></apply></apply></apply>
        <apply><power/><ci>y</ci><ci>b</ci></apply></apply></apply>
      <apply><eq/><ci>w</ci><piecewise>
        <piece><apply><minus/>{_cn(2)}<apply><divide/>{_cn(1)}<ci>x</ci></apply></apply><apply><gt/><ci>x</ci>{_cn(1)}</apply></piece>
        <otherwise><apply><plus/><ci>x</ci><apply><sin/><apply><minus/><ci>x</ci>{_cn(1)}</apply></apply></apply></otherwise>
      </piecewise></apply>
      {_ode('x', f'<apply><minus/><apply><plus/><apply><tanh/><ci>v</ci></apply><apply><times/>{_cn(0.3)}<ci>w</ci></apply><apply><times/>{_cn(0.2)}<apply><max/><ci>x</ci>{_cn(0.2)}</apply></apply></apply><apply><times/>{_cn(1.5)}<ci>x</ci></apply></apply>')}
      {_ode('y', f'<apply><minus/><apply><plus/><apply><times/><ci>u</ci><apply><cos/><ci>y</ci></apply></apply><apply><times/>{_cn(0.5)}<apply><min/><ci>y</ci>{_cn(2)}</apply></apply><apply><arctan/><ci>x</ci></apply></apply><apply><plus/><apply><divide/><ci>y</ci><apply><plus/><ci>b</ci><apply><abs/><apply><minus/><ci>x</ci><ci>y</ci></apply></apply></apply></apply><apply><times/>{_cn(0.5)}<ci>y</ci></apply></apply></apply>')}
    </math>
  </component>""")

NLA = _cellml(f"""
  <component name="c">
    <variable name="t" units="dimensionless"/>
    <variable name="x" units="dimensionless" initial_value="1"/>
    <variable name="z" units="dimensionless" initial_value="1"/>
    <math {MATHML}>
      <apply><eq/><apply><plus/><ci>z</ci><apply><exp/><ci>z</ci></apply></apply><ci>x</ci></apply>
      {_ode('x', '<apply><minus/><ci>z</ci></apply>')}
    </math>
  </component>""")


def _write(tmp_path, name, text):
    path = tmp_path / f'{name}.cellml'
    path.write_text(text)
    return str(path)


def _kernel_model(tmp_path, name, text):
    _, am = kernel_ir.analyse_cellml(_write(tmp_path, name, text))
    return am, kernel_ir.build_kernel_model(am)


needs_c = pytest.mark.skipif(shutil.which('cmake') is None or (shutil.which('cc') or shutil.which('gcc')) is None,
                             reason='needs CMake and a C compiler')


def _helper(tmp_path, text, sim_time=1.0, dt=0.1, **solver_info):
    from libcuflynx.solver_wrappers import get_simulation_helper
    solver_info.setdefault('build_dir', str(tmp_path / 'build'))
    return get_simulation_helper(_write(tmp_path, 'model', text), 'BE_batched_c', 'cellml', dt=dt,
                                 sim_time=sim_time, solver_info=solver_info, pre_time=0.0)


@pytest.mark.unit
def test_unconnected_cells_are_blocks_of_one_kernel_and_coupled_states_one_block(tmp_path):
    _, km = _kernel_model(tmp_path, 'cells', CELLS)
    by_names = {tuple(km.states[i].component for i in b.states): b for b in km.blocks}
    assert set(by_names) == {('cell_a',), ('cell_b',), ('pair', 'pair')}
    assert by_names[('cell_a',)].kernel == by_names[('cell_b',)].kernel != by_names[('pair', 'pair')].kernel
    assert len(km.kernels) == 2
    cell = km.kernels[by_names[('cell_a',)].kernel]
    # the rate reads k through the algebraic flux, a temporary; the Jacobian is d(-k x)/dx = -k
    assert cell.rates.assignments == [('a0', ('mul', ('in', 0), ('y', 0)))]
    assert [(r, c) for r, c, _ in cell.jacobian.entries] == [(0, 0)]
    # each cell reads its own k, so no input of the cells' kernel is shared
    assert cell.shared_inputs == ()


@pytest.mark.unit
def test_algebraic_loop_is_refused(tmp_path):
    with pytest.raises(kernel_ir.KernelIRError, match='algebraic loop'):
        _kernel_model(tmp_path, 'nla', NLA)


@pytest.mark.unit
def test_derivative_rules_of_the_intermediate_form():
    tangent = lambda name, j: None                                # noqa: E731
    x = ('y', 0)
    assert kernel_ir.derivative(('mul', ('in', 0), x), 0, tangent) == ('in', 0)
    assert kernel_ir.derivative(('mul', ('in', 0), x), 1, tangent) is None
    # the quotient rule as (du - (u/v) dv) / v
    assert kernel_ir.derivative(('div', ('num', 1.0), x), 0, tangent) == \
        ('div', ('neg', ('mul', ('div', ('num', 1.0), x), ('num', 1.0))), x)


@needs_c
@pytest.mark.solver
def test_exact_jacobian_matches_central_differences(tmp_path):
    from libcuflynx.generators.cpp import batched
    am, km = _kernel_model(tmp_path, 'ops', OPS)
    assert len(km.blocks) == 1
    path, _ = batched.build(batched.sources(am, km), str(tmp_path))
    lib = batched.BatchedLibrary(path)
    states, rates, variables = np.zeros(km.state_count), np.zeros(km.state_count), np.zeros(km.variable_count)
    lib.bk_initialise(batched.ptr(states), batched.ptr(rates), batched.ptr(variables))
    lib.bk_compute_computed_constants(batched.ptr(variables))
    block = km.blocks[0]
    inputs = variables[block.inputs].copy()
    n = len(block.states)

    def f(y):
        out = np.zeros(n)
        lib.bk_kernel_rates(block.kernel, 0.0, batched.ptr(y), batched.ptr(inputs), batched.ptr(out))
        return out

    rng = np.random.default_rng(1)
    for y in [states[block.states].copy()] + [rng.uniform(0.2, 2.0, n) for _ in range(5)]:
        exact = np.zeros(n * n)
        lib.bk_kernel_jacobian(block.kernel, 0.0, batched.ptr(y), batched.ptr(inputs), batched.ptr(exact))
        central = np.zeros((n, n))
        for j in range(n):
            d = 1e-6 * max(abs(y[j]), 1.0)
            up, down = y.copy(), y.copy()
            up[j] += d
            down[j] -= d
            central[:, j] = (f(up) - f(down)) / (2 * d)
        assert np.allclose(exact.reshape(n, n), central, rtol=1e-6, atol=1e-8), (y, exact, central)


@needs_c
@pytest.mark.solver
def test_solutions_converge_onto_the_analytic_ones(tmp_path):
    t = np.linspace(0.0, 3.0, 31)
    previous = None
    for rtol in (1e-4, 1e-6, 1e-8):
        helper = _helper(tmp_path, CELLS, sim_time=3.0, rtol=rtol, atol=1e-12)
        assert helper.run()
        a, b, p = (helper.get_results([[n]])[0][0] for n in ('cell_a/x', 'cell_b/x', 'pair/p'))
        worst = max(np.max(np.abs(a - 2.0 * np.exp(-0.5 * t))) / 2.0, np.max(np.abs(b - np.exp(-0.2 * t))),
                    np.max(np.abs(p - np.cos(3.0 * t))))
        if previous is not None:
            assert worst < 0.3 * previous
        previous = worst
    assert worst < 1e-3
    # algebraic variables come from libCellML's computeVariables at each output
    flux = helper.get_results([['cell_a/flux']])[0][0]
    assert np.allclose(flux, 0.5 * a, rtol=1e-12)


@needs_c
@pytest.mark.solver
def test_results_are_the_same_on_any_number_of_threads(tmp_path):
    results = []
    for threads in (1, 3):
        helper = _helper(tmp_path, CELLS, sim_time=5.0, rtol=1e-5, threads=threads)
        assert helper.run()
        results.append(np.array(helper.get_all_results(flatten=True)))
    assert np.array_equal(results[0], results[1])


@needs_c
@pytest.mark.solver
def test_a_block_whose_inputs_did_not_change_continues_its_steps(tmp_path):
    """Changing cell_b's rate constant between sub-experiments restarts cell_b only: cell_a's outputs are the
    same bits as in one unbroken run, and cell_b follows the new constant from where it was."""
    from libcuflynx.protocol_runners.protocol_executor import ProtocolExecutor
    whole = _helper(tmp_path, CELLS, sim_time=4.0, dt=0.5, rtol=1e-6)
    assert whole.run()
    unbroken = whole.get_results([['cell_a/x']])[0][0]

    helper = _helper(tmp_path, CELLS, sim_time=2.0, dt=0.5, rtol=1e-6)
    protocol = {'pre_times': [0.0], 'sim_times': [[2.0, 2.0]],
                'params_to_change': {'cell_b/k': [[0.2, 1.0]]}}
    steps_before = helper.stats()['steps']
    ok, by_sub, _, t_by_exp = ProtocolExecutor(helper).run_protocol(protocol, result_variables=[['cell_a/x',
                                                                                                   'cell_b/x']])
    assert ok
    a = np.concatenate([by_sub[(0, 0)][0][0], by_sub[(0, 1)][0][0][1:]])
    b = np.concatenate([by_sub[(0, 0)][0][1], by_sub[(0, 1)][0][1][1:]])
    assert np.array_equal(a, unbroken)
    t = t_by_exp[0]
    expected = np.where(t <= 2.0, np.exp(-0.2 * t), np.exp(-0.4) * np.exp(-1.0 * (t - 2.0)))
    assert np.max(np.abs(b - expected)) < 2e-3
    assert helper.stats()['steps'] > steps_before


@needs_c
@pytest.mark.solver
def test_a_state_initialised_by_a_parameter_follows_it(tmp_path):
    helper = _helper(tmp_path, CELLS, sim_time=1.0, rtol=1e-6)
    helper.set_param_vals(['cell_a/x0'], [5.0])
    assert helper.run()
    x = helper.get_results([['cell_a/x']])[0][0]
    assert x[0] == 5.0
    assert abs(x[-1] - 5.0 * math.exp(-0.5)) < 5e-3 * 5.0
    with pytest.raises(ValueError, match='cannot set states'):
        helper.set_param_vals(['cell_a/x'], [1.0], change_states=False)


@needs_c
@pytest.mark.solver
def test_agrees_with_cvode_myokit(tmp_path):
    from libcuflynx.solver_wrappers import get_simulation_helper
    path = _write(tmp_path, 'ops_model', OPS)
    names = [['ops/x', 'ops/y', 'ops/v']]
    out = {}
    for solver, info in (('CVODE_myokit', {'rtol': 1e-10, 'atol': 1e-12}),
                         ('BE_batched_c', {'rtol': 1e-7, 'build_dir': str(tmp_path / 'build')})):
        helper = get_simulation_helper(path, solver, 'cellml', dt=0.05, sim_time=4.0, solver_info=info)
        assert helper.run()
        out[solver] = np.array(helper.get_results(names)[0])
    scale = np.max(np.abs(out['CVODE_myokit']), axis=1, keepdims=True)
    assert np.max(np.abs(out['BE_batched_c'] - out['CVODE_myokit']) / scale) < 2e-3


@pytest.mark.unit
def test_be_batched_c_is_a_cellml_solver_in_the_schema():
    from libcuflynx.parsers.PrimitiveParsers import SOLVER_INFO_FIELDS, SOLVER_SCHEMA
    assert 'BE_batched_c' in SOLVER_SCHEMA['solvers_by_model_type']['cellml']
    assert SOLVER_SCHEMA['stiff_suitable_methods']['BE_batched_c'] == ['backward_euler']
    from libcuflynx.solver_wrappers import batched_c_helper
    defaults = {f['name']: f['default'] for f in SOLVER_INFO_FIELDS['BE_batched_c']}
    assert defaults['rtol'] == batched_c_helper.DEFAULT_RTOL
    assert defaults['atol'] == batched_c_helper.DEFAULT_ATOL
    assert defaults['MaximumNumberOfSteps'] == batched_c_helper.DEFAULT_MAX_STEPS
