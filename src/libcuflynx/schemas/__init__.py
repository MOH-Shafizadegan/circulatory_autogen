"""Machine-readable JSON Schemas of libcuflynx's input files, shipped as package data.

* ``module_array.schema.json`` -- ``<prefix>_module_array.json``: a list of instance records,
  in the current (PhLynx) or the older libcuflynx keys, with the optional ``per_submodule_inputs`` /
  ``per_submodule_outputs`` of a supermodule instance and the optional ``parameterisation``
  (an ``instance`` key is refused).
* ``module_config.schema.json`` -- ``*_modules_config.json``: component entries in either key
  style, with their ports and ``variables_and_units`` (and an optional
  ``default_parameterisation``, older name ``default_instance``), and supermodule entries.
* ``obs_data.schema.json`` -- ``<name>_obs_data.json``: the optional ``obs_data_name``, the
  ``data_items``, ``prediction_items`` and ``protocol_info`` (with ``protocol_traces`` and
  ``protocol_shapes``).
* ``params_for_id.schema.json`` -- ``<prefix>_params_for_id.json``: the JSON form of
  params_for_id (version 1), which a ``params_for_id.csv`` is converted to on read
  (``ObsAndParamDataParser.params_for_id_csv_to_json``).
* ``calibration_workflow.schema.json`` -- ``calibration_workflow.json``: ordered calibrations
  of module parameterisations merged into a supermodule parameterisation
  (``libcuflynx.calibration_workflow``, whose ``spec.py`` and ``resolve.py`` check the same
  rules and those that need the library).

libcuflynx does not need a JSON Schema library: the readers check the same rules when they
load the files (``utilities/config_schemas.py`` for module configs and arrays,
``parsers/PrimitiveParsers.py`` for obs_data and params_for_id). The schemas are for editors, other tools (PhLynx) and tests.
Use :func:`schema_file` or :func:`load_schema` to get at them.
"""

import json

from libcuflynx.utilities.package_resources import package_data_file

MODULE_ARRAY_SCHEMA = 'module_array.schema.json'
MODULE_CONFIG_SCHEMA = 'module_config.schema.json'
OBS_DATA_SCHEMA = 'obs_data.schema.json'
PARAMS_FOR_ID_SCHEMA = 'params_for_id.schema.json'
CALIBRATION_WORKFLOW_SCHEMA = 'calibration_workflow.schema.json'


def schema_file(name):
    """The shipped schema ``name`` as an :class:`importlib.abc.Traversable`."""
    return package_data_file('libcuflynx.schemas', name)


def load_schema(name):
    """The shipped schema ``name`` (e.g. ``MODULE_ARRAY_SCHEMA``) as a dict."""
    return json.loads(schema_file(name).read_text(encoding='utf-8'))
