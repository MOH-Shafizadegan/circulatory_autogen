"""Boundary conditions (``params_to_change``) as emulator inputs.

By default an emulator is trained with the protocol held fixed, so every params_to_change
value is a hidden constant of the surrogate -- and the fingerprint refuses the emulator the
moment one of them changes. Listing an entry under ``emulator_settings.emulated_params_to_change``
turns it into an extra emulator input instead: training samples it over the range given there,
and a run that uses the emulator reads its value from that run's own obs_data. One trained
emulator then serves any protocol whose BCs fall inside the trained box.

What becomes an input is the user's choice, never inferred: every BC that is added costs
training samples, and a BC that barely moves the features is pure cost. Two kinds of key:

* a ``params_to_change`` name whose leaves are numbers -- **one** input, shared by every
  (experiment, sub-experiment) it appears in;
* a numeric field of a ``protocol_shapes`` entry: ``<shape>.from`` / ``<shape>.to`` for a
  ramp, ``<shape>.events[<i>].<field>`` (level, start, length, period) for a pacing shape.

A shared input carries one value per evaluation. An obs_data that gives a listed BC different
values in different experiments cannot be represented by it and is refused rather than silently
reduced to one of them.

Each input is a dict ``{label, key, kind, shape, event, field, min, max}``; the list of them is
stored in the bundle metadata as ``bc_inputs``, so a run that uses the emulator needs no
emulation settings of its own to know which BCs to read.
"""
import copy
import re

import numpy as np

#: The fields of a pacing event that make sense to vary continuously. ``multiplier`` is left
#: out on purpose: it is a repeat count (an integer, and 0 means "unbounded").
EVENT_FIELDS = ('level', 'start', 'length', 'period')
RAMP_FIELDS = ('from', 'to')

_EVENT_KEY = re.compile(r'^(?P<shape>.+)\.events\[(?P<event>\d+)\]\.(?P<field>\w+)$')
_RAMP_KEY = re.compile(r'^(?P<shape>.+)\.(?P<field>from|to)$')

#: What a varied value is replaced with in the fingerprint: the emulator is trained over it,
#: so its value in the obs_data at hand is an input, not part of what was trained against.
EMULATED_MARKER = '<emulated>'


class BCInputError(ValueError):
    """An ``emulated_params_to_change`` entry that cannot be an emulator input."""


# ---------------------------------------------------------------------- leaves

def _leaves(value):
    """``(exp_idx, sub_idx, leaf)`` for every leaf of a params_to_change entry."""
    found = []
    if not isinstance(value, (list, tuple)):
        return [(0, 0, value)]
    for exp_idx, row in enumerate(value):
        if isinstance(row, (list, tuple)):
            found.extend((exp_idx, sub_idx, leaf) for sub_idx, leaf in enumerate(row))
        else:
            found.append((exp_idx, None, row))
    return found


def _is_number(leaf):
    return isinstance(leaf, (int, float, np.integer, np.floating)) and not isinstance(leaf, bool)


def _shape_references(protocol_info):
    """Names of the shapes that some params_to_change entry actually uses."""
    return {leaf for value in (protocol_info.get('params_to_change') or {}).values()
            for _, _, leaf in _leaves(value) if isinstance(leaf, str)}


def _length_key(event):
    # A pacing event's length may be written `length` (.mmt) or `duration` (Myokit's API).
    return 'duration' if 'duration' in event and 'length' not in event else 'length'


# ---------------------------------------------------------------------- spec

def resolve_bc_inputs(protocol_info, emulated):
    """Turn ``emulated_params_to_change`` into the list of BC input dicts, in the order given.

    Every refusal here is a config error that would otherwise surface as a confusing shape or
    key error a hundred simulations into training.
    """
    if not emulated:
        return []
    if not isinstance(emulated, dict):
        raise BCInputError(
            f'emulator_settings.emulated_params_to_change must map each BC to '
            f'{{range: [min, max]}}, got {type(emulated).__name__}.')
    params_to_change = protocol_info.get('params_to_change') or {}
    shapes = protocol_info.get('protocol_shapes') or {}
    referenced = _shape_references(protocol_info)

    inputs = []
    for key, entry in emulated.items():
        key = str(key)
        lo, hi = _range(key, entry)
        base = {'label': key, 'key': key, 'shape': None, 'event': None, 'field': None,
                'min': lo, 'max': hi}
        if key in params_to_change:
            leaves = _leaves(params_to_change[key])
            traces = sorted({leaf for _, _, leaf in leaves if isinstance(leaf, str)})
            if traces:
                raise BCInputError(
                    f'emulated_params_to_change["{key}"]: that params_to_change entry uses the '
                    f'trace(s)/shape(s) {traces}, so it has no single number to vary. Vary a '
                    f'field of the shape instead (e.g. "<shape>.events[0].level"), or give the '
                    f'entry numeric values.')
            if not leaves or not all(_is_number(leaf) for _, _, leaf in leaves):
                raise BCInputError(
                    f'emulated_params_to_change["{key}"]: params_to_change["{key}"] has no '
                    f'numeric value to vary.')
            inputs.append(dict(base, kind='value'))
            continue

        match = _EVENT_KEY.match(key)
        if match:
            shape_name, event_idx, field = (match['shape'], int(match['event']), match['field'])
            shape = _referenced_shape(key, shape_name, shapes, referenced)
            if shape.get('type') == 'ramp':
                raise BCInputError(f'emulated_params_to_change["{key}"]: "{shape_name}" is a '
                                   f'ramp; vary "{shape_name}.from" or "{shape_name}.to".')
            if field not in EVENT_FIELDS:
                raise BCInputError(
                    f'emulated_params_to_change["{key}"]: "{field}" cannot be varied; an event '
                    f'field must be one of {", ".join(EVENT_FIELDS)}.')
            events = shape.get('events') or []
            if event_idx >= len(events):
                raise BCInputError(
                    f'emulated_params_to_change["{key}"]: "{shape_name}" has {len(events)} '
                    f'event(s), so there is no event {event_idx}.')
            stored = _length_key(events[event_idx]) if field == 'length' else field
            if stored not in events[event_idx]:
                raise BCInputError(
                    f'emulated_params_to_change["{key}"]: event {event_idx} of "{shape_name}" '
                    f'does not set "{field}", so there is no value to vary.')
            inputs.append(dict(base, kind='event', shape=shape_name, event=event_idx,
                               field=field))
            continue

        match = _RAMP_KEY.match(key)
        if match and match['shape'] in shapes:
            shape_name, field = match['shape'], match['field']
            shape = _referenced_shape(key, shape_name, shapes, referenced)
            if shape.get('type') != 'ramp':
                raise BCInputError(f'emulated_params_to_change["{key}"]: "{shape_name}" is not a '
                                   f'ramp, so it has no "{field}".')
            inputs.append(dict(base, kind='ramp', shape=shape_name, field=field))
            continue

        raise BCInputError(
            f'emulated_params_to_change["{key}"] names neither a params_to_change entry '
            f'({sorted(params_to_change)}) nor a protocol_shapes field '
            f'("<shape>.events[<i>].<field>" or "<shape>.from/to", shapes: {sorted(shapes)}).')

    labels = [item['label'] for item in inputs]
    if len(set(labels)) != len(labels):
        raise BCInputError(f'emulated_params_to_change lists a BC twice: {labels}')
    # Checked here as well as at use: an obs_data that already gives the BC different values
    # per experiment is telling us one shared input is the wrong model of it.
    bc_values(protocol_info, inputs)
    return inputs


def _range(key, entry):
    if not isinstance(entry, dict) or 'range' not in entry:
        raise BCInputError(f'emulated_params_to_change["{key}"] needs a range: '
                           f'{{range: [min, max]}}.')
    bounds = entry['range']
    extra = sorted(set(entry) - {'range'})
    if extra:
        raise BCInputError(f'emulated_params_to_change["{key}"] has unknown setting(s) {extra}; '
                           f'only "range" is supported.')
    try:
        lo, hi = (float(v) for v in bounds)
    except (TypeError, ValueError):
        raise BCInputError(f'emulated_params_to_change["{key}"].range must be [min, max], '
                           f'got {bounds!r}.') from None
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
        raise BCInputError(f'emulated_params_to_change["{key}"].range is [{lo}, {hi}]; it needs '
                           f'finite bounds with min < max.')
    return lo, hi


def _referenced_shape(key, shape_name, shapes, referenced):
    if shape_name not in shapes:
        raise BCInputError(f'emulated_params_to_change["{key}"]: there is no protocol_shapes '
                           f'entry "{shape_name}" (shapes: {sorted(shapes)}).')
    if shape_name not in referenced:
        raise BCInputError(f'emulated_params_to_change["{key}"]: protocol_shapes["{shape_name}"] '
                           f'is not used by any params_to_change entry, so varying it changes '
                           f'nothing.')
    return shapes[shape_name]


# ---------------------------------------------------------------------- values

def bc_values(protocol_info, inputs):
    """The value of every BC input in ``protocol_info``, in input order.

    Refuses a ``value`` BC whose leaves differ between experiments: one shared input cannot
    hold two values at once, and picking one would answer about a protocol nobody ran.
    """
    values = []
    params_to_change = protocol_info.get('params_to_change') or {}
    shapes = protocol_info.get('protocol_shapes') or {}
    for item in inputs:
        kind, key = item['kind'], item['key']
        if kind == 'value':
            if key not in params_to_change:
                raise BCInputError(
                    f'the emulator takes the BC "{key}" as an input, but this obs_data has no '
                    f'params_to_change["{key}"] to read it from.')
            leaves = [float(leaf) if _is_number(leaf) else leaf
                      for _, _, leaf in _leaves(params_to_change[key])]
            if not leaves or not all(isinstance(leaf, float) for leaf in leaves):
                raise BCInputError(f'params_to_change["{key}"] must be numeric to be read as an '
                                   f'emulator input, got {params_to_change[key]!r}.')
            if not np.allclose(leaves, leaves[0], rtol=1e-12, atol=0.0):
                raise BCInputError(
                    f'params_to_change["{key}"] takes different values across experiments '
                    f'({params_to_change[key]!r}), but it is one shared emulator input. '
                    f'Emulating a BC per experiment is not supported yet: give it the same value '
                    f'everywhere, or leave it out of emulated_params_to_change.')
            values.append(leaves[0])
            continue
        shape = shapes.get(item['shape'])
        if shape is None:
            raise BCInputError(f'the emulator takes the BC "{key}" as an input, but this '
                               f'obs_data has no protocol_shapes["{item["shape"]}"].')
        if kind == 'ramp':
            values.append(float(shape[item['field']]))
        else:
            event = (shape.get('events') or [])[item['event']]
            field = _length_key(event) if item['field'] == 'length' else item['field']
            values.append(float(event[field]))
    return np.asarray(values, dtype=float)


def apply_bc_values(protocol_info, inputs, values):
    """Write ``values`` into ``protocol_info`` in place, and re-expand any shape touched.

    In place because the solver helper holds this very dict (``set_protocol_info`` keeps a
    reference), so a copy would change nothing the simulation sees.
    """
    from libcuflynx.utilities.protocol_shapes import materialise_shapes

    touched = set()
    params_to_change = protocol_info.get('params_to_change') or {}
    for item, value in zip(inputs, values):
        value = float(value)
        if item['kind'] == 'value':
            params_to_change[item['key']] = _replace_leaves(params_to_change[item['key']], value)
            continue
        shape = protocol_info['protocol_shapes'][item['shape']]
        if item['kind'] == 'ramp':
            shape[item['field']] = value
        else:
            event = shape['events'][item['event']]
            event[_length_key(event) if item['field'] == 'length' else item['field']] = value
        touched.add(item['shape'])
    if touched:
        traces = protocol_info.get('protocol_traces') or {}
        # The expanded trace of the old shape would otherwise "disagree" with the new one.
        for name in touched:
            traces.pop(name, None)
        materialise_shapes(protocol_info)


def _replace_leaves(value, new):
    if not isinstance(value, (list, tuple)):
        return new
    return [[new for _ in row] if isinstance(row, (list, tuple)) else new for row in value]


class preserved_protocol:
    """Restore ``protocol_info``'s BC-bearing parts on exit, in place.

    Training writes a different BC into the shared protocol for every sample; without this the
    engine is left holding the last sample's protocol, and anything that reads it afterwards
    (plots, validation, the fingerprint) describes a protocol that was never asked for.
    """

    _KEYS = ('params_to_change', 'protocol_shapes', 'protocol_traces')

    def __init__(self, protocol_info):
        self.protocol_info = protocol_info

    def __enter__(self):
        self._saved = {key: copy.deepcopy(self.protocol_info[key])
                       for key in self._KEYS if key in self.protocol_info}
        return self.protocol_info

    def __exit__(self, *exc):
        for key in self._KEYS:
            current = self.protocol_info.get(key)
            if key not in self._saved:
                self.protocol_info.pop(key, None)
            elif isinstance(current, dict):
                current.clear()
                current.update(self._saved[key])
            else:
                self.protocol_info[key] = self._saved[key]
        return False


# ---------------------------------------------------------------------- fingerprint

def masked_protocol(protocol_info, inputs):
    """params_to_change and protocol_shapes with every emulated value replaced by a marker.

    What the fingerprint hashes instead of the raw values: a BC that is an input may take any
    value in its range, while every other BC is still something the emulator was trained at.
    """
    params_to_change = copy.deepcopy(protocol_info.get('params_to_change') or {})
    shapes = copy.deepcopy(protocol_info.get('protocol_shapes') or {})
    for item in inputs or []:
        if item['kind'] == 'value' and item['key'] in params_to_change:
            params_to_change[item['key']] = _replace_leaves(params_to_change[item['key']],
                                                            EMULATED_MARKER)
        elif item.get('shape') in shapes:
            shape = shapes[item['shape']]
            if item['kind'] == 'ramp':
                shape[item['field']] = EMULATED_MARKER
            else:
                events = shape.get('events') or []
                if item['event'] < len(events):
                    event = events[item['event']]
                    field = _length_key(event) if item['field'] == 'length' else item['field']
                    event[field] = EMULATED_MARKER
    return params_to_change, shapes


def fingerprint_entries(inputs):
    """The part of each BC input that defines the trained box (not the obs_data value)."""
    return [{'label': item['label'], 'kind': item['kind'], 'min': float(item['min']),
             'max': float(item['max'])} for item in inputs or []]


# ---------------------------------------------------------------------- use

def configure_bc_inputs(helper, protocol_info):
    """Hand an emulator helper the BC values of the run's own protocol.

    Shared by every consumer of an emulator (param id, SA) so they cannot disagree on where a
    BC input's value comes from. A no-op for an emulator trained without BC inputs.
    """
    inputs = getattr(helper.bundle, 'bc_inputs', None) or []
    if not inputs:
        return
    helper.set_bc_values(bc_values(protocol_info, inputs))
