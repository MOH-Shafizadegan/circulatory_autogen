'''
Reading and checking a ``calibration_workflow.json`` (schema:
``libcuflynx/schemas/calibration_workflow.schema.json``).

A workflow is an ordered set of calibrations whose results are merged into one model, the
workflow's ``target``. Each step names the three inputs of an ordinary param_id run: a
``model``, an ``obs_data`` file and a ``params_for_id`` file. A model is either

* a module array of CA (``"resources/aorta_module_array.csv"``, or
  ``{"module_array": ..., "parameters": ...}`` when its parameters file is not
  ``<prefix>_parameters.csv`` beside it), or
* a module of a module library, ``{"module_type", "version", "parameterisation"}``.

Paths are relative to the workflow file. A step may take the calibrated values of earlier steps
as fixed parameter values (``fixed_from``) or, from steps that stored their posterior
(``store_distribution``), as priors (``priors_from``).

Only the file is checked here. Whether its files and modules exist, and where a step's model sits
in the target, is checked in ``resolve.py``.
'''

import copy
import json
import os
from dataclasses import dataclass, field

from libcuflynx.utilities.parameterisations import (check_parameterisation_name,
                                                    reject_instance_key)

SCHEMA_VERSION = 1
WORKFLOW_FILE_NAME = 'calibration_workflow.json'

# The user_inputs keys a workflow (or a step) may set. Everything else in a step's
# user_inputs is decided by the workflow: the model, the files, the output directories.
ALLOWED_SETTINGS = (
    'param_id_method', 'optimiser_options', 'debug_optimiser_options',
    'model_type', 'solver', 'solver_info', 'dt', 'pre_time', 'sim_time',
    'DEBUG', 'do_ad', 'UQ_options', 'debug_UQ_options',
    'use_builtin_modules', 'external_modules_dir',
    'operation_funcs_external_path', 'cost_funcs_external_path', 'modifier_funcs_external_path',
    'write_cuflynx_study',
)

PRIOR_KINDS = ('mvnormal', 'normal', 'kde')
DEFAULT_PRIOR_KIND = 'mvnormal'

_TOP_KEYS = {'schema_version', 'workflow_name', 'description', 'target', 'module_library_dirs',
             'settings', 'steps', 'comment'}
_STEP_KEYS = {'id', 'model', 'obs_data', 'params_for_id', 'submodule', 'vessel', 'vessel_map',
              'fixed_from', 'priors_from', 'store_distribution', 'settings', 'description',
              'comment'}
_TARGET_KEYS = {'module_type', 'version', 'parameterisation'}
_ARRAY_KEYS = {'module_array', 'parameters'}
# a module array file's name is <prefix><suffix>
ARRAY_SUFFIXES = ('_module_array.csv', '_module_array.json', '_vessel_array.csv',
                  '_vessel_array.json')
_PRIOR_KEYS = {'step', 'kind', 'inflate', 'parameters'}


class WorkflowError(ValueError):
    '''A calibration workflow that cannot be run as written.'''


@dataclass(frozen=True)
class Target:
    '''A model that is a module of a module library: one parameterisation of it.'''
    module_type: str
    version: str
    parameterisation: str

    kind = 'library'

    def label(self):
        return f'{self.module_type}/{self.version}/{self.parameterisation}'

    def as_dict(self):
        return {'module_type': self.module_type, 'version': self.version,
                'parameterisation': self.parameterisation}


LibraryModule = Target


@dataclass(frozen=True)
class ModuleArray:
    '''A model that is a module array of CA, with its parameters file.'''
    module_array: str
    parameters: str

    kind = 'array'

    @property
    def prefix(self):
        return array_prefix(self.module_array)

    def label(self):
        return self.prefix

    def as_dict(self):
        return {'module_array': self.module_array, 'parameters': self.parameters}


def array_prefix(path):
    '''The file prefix of a module array file (``aorta`` for ``aorta_module_array.csv``).'''
    name = os.path.basename(path)
    for suffix in ARRAY_SUFFIXES:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]


@dataclass(frozen=True)
class PriorSource:
    step: str
    kind: str = DEFAULT_PRIOR_KIND
    inflate: float = 1.0
    parameters: tuple = None   # None: every parameter of the stored distribution

    def as_dict(self):
        out = {'step': self.step, 'kind': self.kind, 'inflate': self.inflate}
        if self.parameters is not None:
            out['parameters'] = list(self.parameters)
        return out


@dataclass
class Step:
    id: str
    model: object                  # Target (a library module) or ModuleArray
    obs_data: str
    params_for_id: str
    submodule: str = None          # where a library module sits in a supermodule target
    vessel: str = None             # where a library module sits in a module-array target
    vessel_map: dict = field(default_factory=dict)   # module-array step: its vessel -> target's
    depends_on: list = field(default_factory=list)   # fixed_from and priors_from steps
    fixed_from: list = field(default_factory=list)
    priors_from: list = field(default_factory=list)
    store_distribution: bool = False
    settings: dict = field(default_factory=dict)
    description: str = ''


@dataclass
class Workflow:
    name: str
    target: object                 # Target or ModuleArray
    steps: list
    settings: dict = field(default_factory=dict)
    module_library_dirs: list = field(default_factory=list)
    description: str = ''
    path: str = None
    raw: dict = None

    def step(self, step_id):
        for step in self.steps:
            if step.id == step_id:
                return step
        raise WorkflowError(f'workflow "{self.name}" has no step "{step_id}" '
                            f'(steps: {[s.id for s in self.steps]}).')

    def step_settings(self, step):
        '''The workflow's settings with the step's on top (one level of dict merge, so a
        step's optimiser_options add to the workflow's rather than replace them).'''
        merged = copy.deepcopy(self.settings)
        for key, value in step.settings.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **copy.deepcopy(value)}
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def topological_order(self):
        '''The steps, each after everything it depends on; ties keep file order.'''
        done, order = set(), []
        remaining = list(self.steps)
        while remaining:
            ready = [s for s in remaining if all(d in done for d in s.depends_on)]
            if not ready:
                cycle = [s.id for s in remaining]
                raise WorkflowError(f'workflow "{self.name}": fixed_from/priors_from form a cycle among '
                                    f'steps {cycle}.')
            step = ready[0]
            order.append(step)
            done.add(step.id)
            remaining.remove(step)
        return order



# --------------------------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------------------------

def load_workflow(source, module_library_dirs=None):
    '''The checked :class:`Workflow` in ``source`` (a path to a calibration_workflow.json, or
    its content as a dict). ``module_library_dirs`` are added after the file's own (which
    are relative to the file).'''
    if isinstance(source, dict):
        raw, path = copy.deepcopy(source), None
    else:
        path = os.path.abspath(str(source))
        try:
            with open(path, encoding='utf-8-sig') as f:
                raw = json.load(f)
        except json.JSONDecodeError as exc:
            raise WorkflowError(f'{path} is not valid JSON: {exc}') from exc
    return parse_workflow(raw, path=path, module_library_dirs=module_library_dirs)


def parse_workflow(raw, path=None, module_library_dirs=None):
    where = path or 'calibration workflow'
    if not isinstance(raw, dict):
        raise WorkflowError(f'{where}: a calibration workflow is a JSON object.')
    _no_unknown_keys(raw, _TOP_KEYS, where)
    version = raw.get('schema_version', SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise WorkflowError(f'{where}: schema_version {version!r} is not supported by this '
                            f'libcuflynx (it reads version {SCHEMA_VERSION}).')
    name = raw.get('workflow_name')
    if not isinstance(name, str) or not name.strip():
        raise WorkflowError(f'{where}: "workflow_name" must be a non-empty string.')

    base_dir = os.path.dirname(path) if path else os.getcwd()
    library_dirs = [d if os.path.isabs(d) else os.path.normpath(os.path.join(base_dir, d))
                    for d in _string_list(raw, 'module_library_dirs', where)]
    for extra in module_library_dirs or []:
        extra = os.path.abspath(str(extra))
        if extra not in library_dirs:
            library_dirs.append(extra)

    if 'target' not in raw:
        raise WorkflowError(
            f'{where}: no "target". Name the model the results are merged into: a module array '
            f'("resources/<prefix>_module_array.csv") or a library module '
            f'({{"module_type", "version", "parameterisation"}}).')
    target = _model(raw['target'], f'{where}: "target"', base_dir)

    settings = _settings(raw.get('settings', {}), f'{where}: "settings"')

    raw_steps = raw.get('steps')
    if not isinstance(raw_steps, list) or not raw_steps:
        raise WorkflowError(f'{where}: "steps" must be a non-empty list.')
    steps = [_step(s, i, where, base_dir) for i, s in enumerate(raw_steps)]

    workflow = Workflow(name=name.strip(), target=target, steps=steps, settings=settings,
                        module_library_dirs=library_dirs,
                        description=raw.get('description', '') or '', path=path, raw=raw)
    check_workflow(workflow)
    return workflow


def _path(value, where, base_dir):
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError(f'{where} must be a path (relative to the workflow file).')
    value = os.path.expanduser(value.strip())
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(base_dir, value))


def _model(value, where, base_dir):
    '''A ``model`` (or the workflow ``target``): a module array path, ``{"module_array",
    "parameters"}``, or a library module ``{"module_type", "version", "parameterisation"}``.'''
    if isinstance(value, str):
        array = _path(value, where, base_dir)
        return ModuleArray(array, os.path.join(os.path.dirname(array),
                                               f'{array_prefix(array)}_parameters.csv'))
    if isinstance(value, dict) and 'module_array' in value:
        _no_unknown_keys(value, _ARRAY_KEYS, where)
        array = _path(value['module_array'], f'{where}: "module_array"', base_dir)
        parameters = (_path(value['parameters'], f'{where}: "parameters"', base_dir)
                      if 'parameters' in value else
                      os.path.join(os.path.dirname(array), f'{array_prefix(array)}_parameters.csv'))
        return ModuleArray(array, parameters)
    return _target(value, where)


def _no_unknown_keys(obj, allowed, where):
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise WorkflowError(f'{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}.')


def _string_list(obj, key, where):
    value = obj.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise WorkflowError(f'{where}: "{key}" must be a list of non-empty strings.')
    return list(value)


def _target(value, where):
    if not isinstance(value, dict):
        raise WorkflowError(f'{where} must be a module array path, or a library module '
                            f'{sorted(_TARGET_KEYS)}.')
    try:
        reject_instance_key(value, where, what='A workflow target')
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    _no_unknown_keys(value, _TARGET_KEYS, where)
    missing = sorted(_TARGET_KEYS - set(value))
    if missing:
        raise WorkflowError(f'{where} is missing {missing}.')
    for key in ('module_type', 'version'):
        if not isinstance(value[key], str) or not value[key].strip():
            raise WorkflowError(f'{where}: "{key}" must be a non-empty string.')
    try:
        parameterisation = check_parameterisation_name(value['parameterisation'], where)
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    return Target(value['module_type'].strip(), value['version'].strip(), parameterisation)


def _settings(value, where):
    if not isinstance(value, dict):
        raise WorkflowError(f'{where} must be an object.')
    unknown = sorted(set(value) - set(ALLOWED_SETTINGS))
    if unknown:
        raise WorkflowError(
            f'{where}: {unknown} cannot be set by a workflow. The model, obs_data, '
            f'params_for_id and output directories of each step are chosen by the workflow '
            f'itself; settings may be {list(ALLOWED_SETTINGS)}.')
    return copy.deepcopy(value)


def _prior_source(value, where):
    if isinstance(value, str):
        return PriorSource(step=value)
    if not isinstance(value, dict):
        raise WorkflowError(f'{where}: a priors_from entry is a step id or an object with '
                            f'{sorted(_PRIOR_KEYS)}.')
    _no_unknown_keys(value, _PRIOR_KEYS, where)
    if not isinstance(value.get('step'), str):
        raise WorkflowError(f'{where}: "step" must name a step.')
    kind = value.get('kind', DEFAULT_PRIOR_KIND)
    if kind not in PRIOR_KINDS:
        raise WorkflowError(f'{where}: prior kind "{kind}" is not one of {list(PRIOR_KINDS)}.')
    inflate = value.get('inflate', 1.0)
    if isinstance(inflate, bool) or not isinstance(inflate, (int, float)) or inflate <= 0:
        raise WorkflowError(f'{where}: "inflate" must be a positive number (a variance '
                            f'multiplier), not {inflate!r}.')
    parameters = value.get('parameters')
    if parameters is not None:
        if not isinstance(parameters, list) or not all(isinstance(p, str) for p in parameters):
            raise WorkflowError(f'{where}: "parameters" must be a list of parameter names.')
        parameters = tuple(parameters)
    return PriorSource(step=value['step'], kind=kind, inflate=float(inflate),
                       parameters=parameters)


def _step(value, index, where, base_dir):
    label = f'{where}: steps[{index}]'
    if not isinstance(value, dict):
        raise WorkflowError(f'{label} must be an object.')
    if 'target' in value:
        raise WorkflowError(f'{label}: a step names its "model", "obs_data" and '
                            f'"params_for_id"; "target" is the workflow\'s, not a step\'s.')
    if 'depends_on' in value:
        raise WorkflowError(f'{label}: "depends_on" is not a key. A step runs after the steps '
                            f'in its "fixed_from" and "priors_from", the only ways one step '
                            f'affects another; remove "depends_on".')
    _no_unknown_keys(value, _STEP_KEYS, label)
    step_id = value.get('id')
    if not isinstance(step_id, str) or not step_id.strip():
        raise WorkflowError(f'{label}: "id" must be a non-empty string.')
    try:
        check_parameterisation_name(step_id, label, key='id')  # it names an output directory
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    label = f'{where}: step "{step_id}"'
    missing = [k for k in ('model', 'obs_data', 'params_for_id') if k not in value]
    if missing:
        raise WorkflowError(f'{label} has no {missing}. A step names the three inputs of a '
                            f'calibration: "model", "obs_data" and "params_for_id".')
    model = _model(value['model'], f'{label}: "model"', base_dir)
    for key in ('submodule', 'vessel'):
        given = value.get(key)
        if given is not None and (not isinstance(given, str) or not given.strip()):
            raise WorkflowError(f'{label}: "{key}" must be a non-empty string.')
        if given is not None and model.kind != 'library':
            raise WorkflowError(f'{label}: "{key}" places a library module in the target; '
                                f'a module-array model maps its vessels with "vessel_map".')
    vessel_map = value.get('vessel_map', {})
    if not isinstance(vessel_map, dict) or not all(
            isinstance(k, str) and k and isinstance(v, str) and v for k, v in vessel_map.items()):
        raise WorkflowError(f'{label}: "vessel_map" maps vessel names to vessel names.')
    if vessel_map and model.kind != 'array':
        raise WorkflowError(f'{label}: "vessel_map" is for a module-array model; a library '
                            f'module is placed with "submodule" or "vessel".')
    submodule, vessel = value.get('submodule'), value.get('vessel')
    store = value.get('store_distribution', False)
    if not isinstance(store, bool):
        raise WorkflowError(f'{label}: "store_distribution" must be true or false.')
    priors = value.get('priors_from', [])
    if not isinstance(priors, list):
        raise WorkflowError(f'{label}: "priors_from" must be a list.')
    fixed_from = _string_list(value, 'fixed_from', label)
    priors_from = [_prior_source(p, label) for p in priors]
    return Step(id=step_id.strip(), model=model,
                obs_data=_path(value['obs_data'], f'{label}: "obs_data"', base_dir),
                params_for_id=_path(value['params_for_id'], f'{label}: "params_for_id"',
                                    base_dir),
                submodule=submodule.strip() if submodule else None,
                vessel=vessel.strip() if vessel else None,
                vessel_map={k.strip(): v.strip() for k, v in vessel_map.items()},
                depends_on=list(dict.fromkeys(fixed_from + [p.step for p in priors_from])),
                fixed_from=fixed_from,
                priors_from=priors_from,
                store_distribution=store,
                settings=_settings(value.get('settings', {}), f'{label}: "settings"'),
                description=value.get('description', '') or '')


# --------------------------------------------------------------------------------------------
# checks that need only the file
# --------------------------------------------------------------------------------------------

def check_workflow(workflow):
    ids = [s.id for s in workflow.steps]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise WorkflowError(f'workflow "{workflow.name}": step ids {duplicates} are used more '
                            f'than once.')
    for step in workflow.steps:
        for dep in step.depends_on:
            if dep not in ids:
                raise WorkflowError(f'step "{step.id}" takes values from unknown step "{dep}".')
            if dep == step.id:
                raise WorkflowError(f'step "{step.id}" takes values from itself.')
    workflow.topological_order()  # raises on a cycle

    for step in workflow.steps:
        prior_steps = [p.step for p in step.priors_from]
        for source in prior_steps:
            if not workflow.step(source).store_distribution:
                raise WorkflowError(
                    f'step "{step.id}" takes priors from "{source}", but "{source}" does not '
                    f'store its distribution. Set "store_distribution": true on "{source}".')
            if source in step.fixed_from:
                raise WorkflowError(
                    f'step "{step.id}" takes "{source}" both as fixed values and as priors; '
                    f'a parameter is either fixed or calibrated, not both.')
        if len(set(prior_steps)) != len(prior_steps):
            raise WorkflowError(f'step "{step.id}" lists a step in priors_from twice.')
        settings = workflow.step_settings(step)
        if step.store_distribution and not settings.get('UQ_options') and \
                not settings.get('debug_UQ_options'):
            raise WorkflowError(
                f'step "{step.id}" stores its distribution, which is sampled by MCMC after '
                f'the fit, but no "UQ_options" are set for it (at least num_steps and '
                f'num_walkers; see the UQ section of the parameter-identification docs).')
        if step.priors_from and not (step.store_distribution or uses_log_posterior(settings)):
            raise WorkflowError(
                f'step "{step.id}" takes priors, but neither samples a posterior '
                f'("store_distribution": true) nor optimises the log posterior '
                f'(optimiser_options.objective_function: "likelihood"). Its calibration would '
                f'ignore the priors.')
    return workflow


def uses_log_posterior(settings):
    '''Whether the step's optimiser minimises the negative log posterior (priors included)
    rather than the cost. Only the genetic algorithm reads objective_function.'''
    for key in ('optimiser_options', 'debug_optimiser_options'):
        if (settings.get(key) or {}).get('objective_function') == 'likelihood':
            return True
    return False
