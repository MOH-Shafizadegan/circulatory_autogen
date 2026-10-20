'''
Selection steps: a step whose model is ``{"select": [...]}``, instances of the workflow's
module-array target calibrated as they are in it. A selection runs as one of the step models
a workflow already has:

* one instance (a record, or ``heart/lv`` inside a super instance) is its library module,
  placed at that instance (``naming.Placed``), as a step naming ``vessel`` would be. Its
  parameterisation is the record's; a submodule's is the one its supermodule's parameters file
  records (``"default"``: its version's default);
* several records are the module array they form, cut from the target (``naming.Renamed``,
  names unchanged). A link to a record outside the selection is dropped, which leaves that
  port open.

Either way the step starts from the target's values for those instances. The target's
parameters file rows for them (plus, for a nested instance, the values its super instances'
parameters files and parameterisations give it) are the step model's parameters, under any
values fixed from earlier steps. A cut port's boundary condition then takes its value from the step's
obs_data protocol (``params_to_change``), the parameterisation, or the target's parameters
file; one with none of these is an :class:`UnsetParametersError`.

The step's obs_data and params_for_id name the instances as the target does. A single
instance's model is generated as ``mod``, so :func:`write_step_inputs` writes renamed copies
of them into the step's directory.
'''

import copy
import json
import os
from dataclasses import dataclass, field

from libcuflynx.calibration_workflow import naming
from libcuflynx.calibration_workflow.spec import LibraryModule, ModuleArray, WorkflowError
from libcuflynx.utilities.parameter_names import (flat_name, is_path_name, split_path_name)
from libcuflynx.utilities.parameterisations import (PARAMETERISATION_KEY,
                                                    parameterisation_rows)
from libcuflynx.utilities.supermodules import instance_parameter_rows


class UnsetParametersError(WorkflowError):
    '''A step model's parameters that nothing gives a value: ``names``, as the target's
    parameters file names them (``C/x``). A cut boundary condition is the usual one; set it with
    a protocol in the step's obs_data (``params_to_change``) or a row in the target's
    parameters file.'''

    def __init__(self, step_id, names):
        self.step_id, self.names = step_id, list(names)
        super().__init__(
            f'step "{step_id}": {len(self.names)} parameter(s) have no value: '
            f'{", ".join(self.names)}. A boundary condition left open by the selection needs '
            f'a protocol: add it to the step\'s obs_data under protocol_info.params_to_change '
            f'(or give it a row in the target\'s parameters file).')


@dataclass(frozen=True)
class SelectedArray(ModuleArray):
    '''The module array several selected records form, cut from the target's
    (``module_array``, ``parameters``: the target's files).'''
    names: tuple = ()
    records: list = field(default_factory=list, compare=False)

    def label(self):
        return f'selection {", ".join(self.names)} of {self.prefix}'

    def as_dict(self):
        return {'select': list(self.names)}


@dataclass
class ResolvedSelection:
    '''What a selection step runs: ``model`` (a LibraryModule or SelectedArray), the target
    ``vessel`` a single instance sits at (None for several), and ``rows``, the target's
    parameter rows for the selection named as the step model's parameters file names them.'''
    model: object
    vessel: str
    rows: list


def _parts(name):
    return tuple(p.strip() for p in name.split('/')) if '/' in name else (name,)


def resolve_selection(step, target, components, supermodules):
    '''The :class:`ResolvedSelection` of ``step`` (whose model is a spec.Selection) in the
    module-array ``target`` (a resolve.ResolvedArray).'''
    names = step.model.names
    by_name = {r['name']: r for r in target.records}
    for name in names:
        if _parts(name)[0] not in by_name:
            raise WorkflowError(f'step "{step.id}" selects "{name}", which is not an instance '
                                f'of the target {target.model.label()} (its records: '
                                f'{sorted(by_name)}).')
    host_rows = _host_rows(target.model.parameters)
    if len(names) == 1:
        return _single(step, _parts(names[0]), by_name, host_rows, components, supermodules)
    nested = [n for n in names if len(_parts(n)) > 1]
    if nested:
        raise WorkflowError(f'step "{step.id}" selects {nested} inside a super instance along '
                            f'with other instances. Select a nested instance on its own, or '
                            f'the records that hold it.')
    selected = set(names)
    records = []
    for record in target.records:
        if record['name'] in selected:
            record = copy.deepcopy(record)
            for key in ('inp_instances', 'out_instances'):
                record[key] = [i for i in record.get(key, []) if _parts(i)[0] in selected]
            records.append(record)
    rows = [row for path, row in host_rows if not path or path[0] in selected]
    model = SelectedArray(target.model.module_array, target.model.parameters, names=tuple(names),
                          records=records)
    return ResolvedSelection(model=model, vessel=None, rows=rows)


def _single(step, parts, by_name, host_rows, components, supermodules):
    record = by_name[parts[0]]
    where = f'step "{step.id}" (selects "{"/".join(parts)}")'
    # the module at each level of the path, and the parameterisation rows it gives what is below
    levels = []        # [(entry, parameterisation or None)], the outermost first
    entry, parameterisation = _entry(record, components, supermodules, where), \
        record.get(PARAMETERISATION_KEY)
    levels.append((entry, parameterisation))
    for depth, name in enumerate(parts[1:], start=1):
        subs = {s['name']: s for s in entry.get('submodules', [])}
        if name not in subs:
            raise WorkflowError(
                f'{where}: "{"/".join(parts[:depth])}" has no submodule "{name}" (it has '
                f'{sorted(subs) or "none"}).')
        sub = subs[name]
        # a submodule's parameterisation is the one its supermodule's parameters file records
        started = ((entry.get('instance_parameters') or {}).get(name) or {}).get(
            'parameterisation', 'default')
        entry = _entry(sub, components, supermodules, where)
        levels.append((entry, None if started == 'default' else started))
    entry, parameterisation = levels[-1]
    try:
        chosen, _ = parameterisation_rows(entry, parameterisation, where)
    except ValueError as exc:
        raise WorkflowError(str(exc)) from exc
    if chosen is None:
        raise WorkflowError(
            f'{where}: {entry["module_type"]} {entry["module_subtype"]} has no parameterisation '
            f'to calibrate (no "parameterisation" and no default_parameterisation file).')
    model = LibraryModule(entry['module_type'], entry['module_subtype'], chosen)

    # lowest precedence first: each enclosing super instance's parameters file and then its
    # parameterisation, the innermost first (an outer supermodule's values win, as in expansion),
    # then the host
    rows = {}
    for depth, (outer, name) in reversed(list(enumerate(levels[:-1]))):
        try:
            _, outer_rows = parameterisation_rows(outer, name, where)
            outer_rows = instance_parameter_rows(outer, supermodules, where) + outer_rows
        except ValueError as exc:
            raise WorkflowError(str(exc)) from exc
        below = parts[depth + 1:]
        for row in outer_rows:
            if not is_path_name(row['variable_name']):
                continue        # a global
            path, variable = split_path_name(row['variable_name'], where)
            if tuple(path[:len(below)]) == below:
                _put(rows, path[len(below):], variable, row)
    for path, row in host_rows:
        if tuple(path[:len(parts)]) == parts:
            _put(rows, path[len(parts):], split_path_name(row['variable_name'])[1], row)
    return ResolvedSelection(model=model, vessel=flat_name(parts), rows=list(rows.values()))


def _put(rows, sub_path, variable, row):
    '''Add ``row`` named for the step model generated as ``mod``: ``mod/<var>``, or
    ``mod__<sub>/<var>`` below it.'''
    vessel = flat_name((naming.STEP_VESSEL,) + tuple(sub_path))
    name = naming.host_row_name(vessel, variable)
    rows[name] = dict(row, variable_name=name)


def _entry(record, components, supermodules, where):
    key = (record['module_type'], record['module_subtype'])
    entry = supermodules.get(key) or components.get(key)
    if entry is None:
        raise WorkflowError(f'{where}: no module {key[0]} version {key[1]} in the module '
                            f'library.')
    return entry


def _host_rows(path):
    '''``[(vessel path, row)]`` of a host parameters file; a global's path is ().'''
    from libcuflynx.calibration_workflow.generate import read_parameter_rows
    out = []
    for row in read_parameter_rows(path):
        name = (row.get('variable_name') or '').strip()
        if not name:
            continue
        if is_path_name(name):
            vessel_path, _ = split_path_name(name, f'{path}: ')
            out.append((tuple(vessel_path), row))
        else:
            out.append(((), row))
    return out


# --------------------------------------------------------------------------------------------
# a single instance's obs_data and params_for_id, renamed for its model
# --------------------------------------------------------------------------------------------

def step_input_paths(step_dir, step):
    '''Where :func:`write_step_inputs` writes the renamed obs_data and params_for_id.'''
    directory = os.path.join(step_dir, 'inputs')
    return (os.path.join(directory, f'{step.id}_obs_data.json'),
            os.path.join(directory, f'{step.id}_params_for_id.json'))


def write_step_inputs(step_dir, step, params_for_id, vessel_map, target_vessels):
    '''Write the step's obs_data and ``params_for_id`` with every target vessel renamed by
    ``vessel_map`` (a naming.Placed) for the step's model. Returns the two paths.'''
    from libcuflynx.parsers.PrimitiveParsers import ObsAndParamDataParser

    def rename(qname):
        if not isinstance(qname, str) or '/' not in qname:
            return qname
        *path, variable = qname.strip().split('/')
        vessel = flat_name(path)
        if vessel not in target_vessels:
            return qname        # not a vessel of the target (a global, or the environment)
        mapped = vessel_map.from_target(vessel)
        if mapped is None:
            raise WorkflowError(f'step "{step.id}" names "{qname}", which is outside its '
                                f'selection {", ".join(step.model.names)}.')
        return f'{mapped}/{variable}'

    with open(step.obs_data, encoding='utf-8-sig') as f:
        obs = json.load(f)
    for key in ('data_items', 'prediction_items'):
        for item in obs.get(key) or []:
            if isinstance(item.get('operands'), list):
                item['operands'] = [rename(o) for o in item['operands']]
    info = obs.get('protocol_info') or {}
    if isinstance(info.get('params_to_change'), dict):
        info['params_to_change'] = {rename(k): v for k, v in info['params_to_change'].items()}
    for plot in obs.get('prediction_plots') or []:
        for ref in (plot.get('x'), plot.get('series')):
            if isinstance(ref, dict) and 'params_to_change' in ref:
                ref['params_to_change'] = rename(ref['params_to_change'])

    if str(params_for_id).lower().endswith('.json'):
        with open(params_for_id, encoding='utf-8-sig') as f:
            params = json.load(f)
    else:
        params = ObsAndParamDataParser.params_for_id_csv_to_json(params_for_id)

    def rename_all(value):
        if isinstance(value, str):
            return rename(value)
        if isinstance(value, list):
            return [rename_all(v) for v in value]
        if isinstance(value, dict):
            return {k: rename_all(v) for k, v in value.items()}
        return value

    entries = params['params'] if isinstance(params, dict) else params
    for entry in entries:
        for key in ('targets', 'modifies', 'inputs'):
            if key in entry:
                entry[key] = rename_all(entry[key])

    obs_path, params_path = step_input_paths(step_dir, step)
    os.makedirs(os.path.dirname(obs_path), exist_ok=True)
    for path, doc in ((obs_path, obs), (params_path, params)):
        with open(path, 'w') as f:
            json.dump(doc, f, indent=1)
    return obs_path, params_path


# --------------------------------------------------------------------------------------------
# values a step's protocol gives the parameters generation left unset
# --------------------------------------------------------------------------------------------

def protocol_values(obs_data_path):
    '''``{qname: value}``: the value each ``params_to_change`` entry of the obs_data starts
    at (its first experiment's first sub-experiment; a trace's first value). A step model is
    generated with these for parameters nothing else sets, and the protocol drives them
    while it runs.'''
    from libcuflynx.utilities.protocol_shapes import materialise_shapes
    with open(obs_data_path, encoding='utf-8-sig') as f:
        info = copy.deepcopy((json.load(f) or {}).get('protocol_info') or {})
    materialise_shapes(info)
    traces = info.get('protocol_traces') or {}
    values = {}
    for qname, by_experiment in (info.get('params_to_change') or {}).items():
        try:
            first = by_experiment[0][0]
        except (IndexError, TypeError):
            continue
        if isinstance(first, str):
            trace_values = (traces.get(first) or {}).get('values') or []
            if not trace_values:
                continue
            first = trace_values[0]
        values[qname] = float(first)
    return values
