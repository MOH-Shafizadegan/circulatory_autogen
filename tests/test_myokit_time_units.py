"""
CVODE_myokit on models whose modules use other time units than the model (issue #604).

CellML takes a module's derivatives w.r.t. its own time variable, so ``dx/dt = 1 [1/ms]``
in a module whose ``t`` is in ms makes x grow by 1000 per second of model time. Myokit
converts the module's ``t`` but takes every derivative w.r.t. the free variable; the
Myokit helper rescales those derivatives after import.
"""
import os
import textwrap

import numpy as np
import pytest

from libcuflynx.scripts.script_generate_with_new_architecture import generate_with_new_architecture
from libcuflynx.solver_wrappers import get_simulation_helper

MIXED_TIME_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="mixed_time" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
  <units name="ms"><unit prefix="milli" units="second"/></units>
  <units name="per_ms"><unit prefix="milli" units="second" exponent="-1"/></units>
  <units name="per_s"><unit units="second" exponent="-1"/></units>
  <component name="environment">
    <variable name="time" units="second" public_interface="out"/>
  </component>
  <component name="fast">
    <variable name="t" units="ms" public_interface="in"/>
    <variable name="x" units="dimensionless" initial_value="0"/>
    <variable name="t_copy" units="ms"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply><cn cellml:units="per_ms">1</cn></apply>
      <apply><eq/><ci>t_copy</ci><ci>t</ci></apply>
    </math>
  </component>
  <component name="slow">
    <variable name="t" units="second" public_interface="in"/>
    <variable name="y" units="dimensionless" initial_value="0"/>
    <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><eq/><apply><diff/><bvar><ci>t</ci></bvar><ci>y</ci></apply><cn cellml:units="per_s">1</cn></apply>
    </math>
  </component>
  <connection><map_components component_1="environment" component_2="fast"/><map_variables variable_1="time" variable_2="t"/></connection>
  <connection><map_components component_1="environment" component_2="slow"/><map_variables variable_1="time" variable_2="t"/></connection>
</model>
"""


def test_derivative_wrt_ms_time_is_scaled_to_model_seconds(tmp_path):
    path = tmp_path / 'mixed_time.cellml'
    path.write_text(MIXED_TIME_CELLML)
    helper = get_simulation_helper(model_path=str(path), solver='CVODE_myokit',
                                   model_type='cellml', dt=0.5, sim_time=1.0, pre_time=0.0)
    helper.run()
    x, t_copy, y = (np.asarray(r).ravel() for r in helper.get_results(
        [['fast/x'], ['fast/t_copy'], ['slow/y']], flatten=True))
    t = np.asarray(helper.get_time())
    np.testing.assert_allclose(x, 1000.0 * t, rtol=1e-6)       # 1 per ms
    np.testing.assert_allclose(t_copy, 1000.0 * t, rtol=1e-12)  # t itself in ms
    np.testing.assert_allclose(y, t, rtol=1e-6)                # the seconds module unchanged


MS_DECAY_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="modules" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
    <component name="ms_decay_type">
        <variable name="t" public_interface="in" units="ms_test"/>
        <variable name="x_init" public_interface="in" units="dimensionless"/>
        <variable name="k" public_interface="in" units="per_ms_test"/>
        <variable initial_value="x_init" name="x" public_interface="out" units="dimensionless"/>
        <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply>
                <eq/>
                <apply><diff/><bvar><ci>t</ci></bvar><ci>x</ci></apply>
                <apply><minus/><apply><times/><ci>k</ci><ci>x</ci></apply></apply>
            </apply>
        </math>
    </component>
</model>
"""

MS_DECAY_CONFIG = """\
[
  {
    "module_type": "ms_decay",
    "module_subtype": "nn",
    "module_format": "cellml",
    "component_file": "ms_decay_modules.cellml",
    "component_type": "ms_decay_type",
    "entrance_ports": [],
    "exit_ports": [],
    "general_ports": [],
    "variables_and_units": [
      ["x_init", "dimensionless", "access", "constant"],
      ["k", "per_ms_test", "access", "constant"],
      ["x", "dimensionless", "access", "variable"]
    ]
  }
]
"""

MS_UNITS_CELLML = """\
<?xml version='1.0' encoding='UTF-8'?>
<model name="Units" xmlns="http://www.cellml.org/cellml/1.1#" xmlns:cellml="http://www.cellml.org/cellml/1.1#">
    <units name="ms_test">
        <unit prefix="milli" units="second"/>
    </units>
    <units name="per_ms_test">
        <unit prefix="milli" units="second" exponent="-1"/>
    </units>
</model>
"""


@pytest.mark.integration
def test_generated_model_with_ms_module_runs_at_model_time(tmp_path):
    """A module written in ms, generated into CA's seconds-based model, decays at its own
    rate: k = 5e-4 /ms is 0.5 /s."""
    external_dir = tmp_path / 'external_modules'
    external_dir.mkdir()
    (external_dir / 'ms_decay_modules.cellml').write_text(MS_DECAY_CELLML)
    (external_dir / 'ms_decay_modules_config.json').write_text(MS_DECAY_CONFIG)
    (external_dir / 'ms_decay_units.cellml').write_text(MS_UNITS_CELLML)

    resources_dir = tmp_path / 'resources'
    resources_dir.mkdir()
    (resources_dir / 'ms_decay_module_array.csv').write_text(textwrap.dedent("""\
        name,module_subtype,module_type,inp_instances,out_instances
        decay,nn,ms_decay,,
        """))
    (resources_dir / 'ms_decay_parameters.csv').write_text(textwrap.dedent("""\
        variable_name,units,value,data_reference
        decay/x_init,dimensionless,2.0,test
        decay/k,per_ms_test,5e-4,test
        """))

    config = {
        'file_prefix': 'ms_decay',
        'input_param_file': 'ms_decay_parameters.csv',
        'model_type': 'cellml',
        'solver': 'CVODE_myokit',
        'resources_dir': str(resources_dir),
        'generated_models_dir': str(tmp_path / 'generated_models'),
        'external_modules_dir': str(external_dir),
        'use_builtin_modules': False,
        'DEBUG': False,
    }
    assert generate_with_new_architecture(False, config)

    model_path = os.path.join(config['generated_models_dir'], 'ms_decay', 'ms_decay.cellml')
    helper = get_simulation_helper(model_path=model_path, solver='CVODE_myokit',
                                   model_type='cellml', dt=0.01, sim_time=2.0, pre_time=0.0)
    helper.run()
    x = np.asarray(helper.get_results([['decay/x']], flatten=True)).ravel()
    t = np.asarray(helper.get_time())
    np.testing.assert_allclose(x, 2.0 * np.exp(-0.5 * t), rtol=1e-4)
