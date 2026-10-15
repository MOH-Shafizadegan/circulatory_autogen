"""Checking an input file against its shipped JSON Schema, as every reader does on load.

The schema owns a file's structure: its keys, their types, enums and bounds, and which are
required. The readers then check only what a schema cannot see (that a file exists, that a
vessel is in the model, that two steps name the same model). ``validate`` raises
:class:`SchemaValidationError`, a ``ValueError``, listing every problem with where it is and
what the field is for::

    obs_data.json: 2 problems with the obs_data file
      at data_items[3].operation: 'maxx' is not one of ['max', 'mean', ...]
          the feature computed from the operands ...
      at protocol_info.sim_times: ...
"""

import functools
import hashlib
import json
import warnings
from collections import OrderedDict

from libcuflynx.schemas import load_schema

#: The most problems a message lists; the rest are counted.
MAX_LISTED = 10
#: How many documents that passed are remembered, so a file read again (a module library is
#: read once per generated model) is not checked again.
VALID_CACHE_SIZE = 4096
_valid = OrderedDict()


NO_JSONSCHEMA = (
    'jsonschema is not installed, so input files are not checked against their schemas (only '
    'by the readers\' own checks). Install it: pip install jsonschema. This happens in a '
    'CUFLynx app released before libcuflynx required it.')


@functools.lru_cache(maxsize=None)
def jsonschema_available():
    """Whether jsonschema imports; warns once when it does not."""
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        warnings.warn(NO_JSONSCHEMA, UserWarning, stacklevel=3)
        return False
    return True


def _digest(schema_name, document):
    try:
        text = json.dumps(document, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return None
    return schema_name, hashlib.sha1(text.encode('utf-8')).hexdigest()


class SchemaValidationError(ValueError):
    """A document its schema refuses. ``problems`` is ``[(path, message)]``."""

    def __init__(self, message, problems=()):
        super().__init__(message)
        self.problems = list(problems)


def _type_checker(base):
    '''``base`` with numpy arrays as arrays and numpy scalars as numbers: a document built
    in Python (an obs_data dict with array values) is checked as its JSON would be.'''
    import numpy as np

    def is_array(checker, value):
        return isinstance(value, (list, tuple, np.ndarray))

    def is_number(checker, value):
        return (isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, (bool, np.bool_)))

    def is_integer(checker, value):
        if isinstance(value, (bool, np.bool_)):
            return False
        if isinstance(value, (int, np.integer)):
            return True
        return isinstance(value, (float, np.floating)) and float(value).is_integer()

    def is_boolean(checker, value):
        return isinstance(value, (bool, np.bool_))

    return base.redefine_many({'array': is_array, 'number': is_number,
                               'integer': is_integer, 'boolean': is_boolean})


@functools.lru_cache(maxsize=None)
def _validator(schema_name):
    import jsonschema
    schema = _rule_descriptions(load_schema(schema_name))
    base = jsonschema.validators.validator_for(schema)
    cls = jsonschema.validators.extend(base, type_checker=_type_checker(base.TYPE_CHECKER))
    return cls(schema, format_checker=base.FORMAT_CHECKER)


#: keys that name an entry of a list, shown after its index
IDENTIFYING_KEYS = ('data_item_name', 'variable', 'id', 'name')


def _where(path, document):
    """``data_items[2] ('systolic').value``: the path, with each list entry's name."""
    out, node = '', document
    for part in path:
        out += f'[{part}]' if isinstance(part, int) else (f'.{part}' if out else str(part))
        try:
            node = node[part]
        except (KeyError, IndexError, TypeError):
            node = None
        if isinstance(part, int) and isinstance(node, dict):
            label = next((node[k] for k in IDENTIFYING_KEYS
                          if isinstance(node.get(k), str) and node[k]), None)
            if label:
                out += f' ({label!r})'
    return out or '(the top level)'


@functools.lru_cache(maxsize=None)
def _owners(schema_name):
    """``{id(subschema): the schema it sits in}`` for the loaded schema ``schema_name``: how to
    get from a failing rule back to the object definition it constrains."""
    owners = {}

    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                for child in (value if isinstance(value, list) else [value]):
                    if isinstance(child, dict):
                        owners.setdefault(id(child), node)
                        walk(child)
                    elif isinstance(child, list):
                        walk(child)
    walk(_validator(schema_name).schema)
    return owners


def _schemas_up(error, owners):
    """The schemas from the failing one out to the document's, innermost first."""
    seen = set()
    while error is not None:
        schema = error.schema
        while isinstance(schema, dict) and id(schema) not in seen:
            seen.add(id(schema))
            yield schema
            schema = owners.get(id(schema))
        error = error.parent


def _description(error, root, owners):
    """What the failing field is for. A rule with its own description (a conditional such as
    "a series needs obs_dt") gives that; a missing key gives the key's description; anything
    else its field's. The whole document's description says nothing about one field."""
    schemas = [s for s in _schemas_up(error, owners) if s is not root]
    # a rule between the error and the definition it constrains says what the rule is for
    for schema in schemas:
        if 'properties' in schema and schema is not schemas[0]:
            break
        if schema.get('description'):
            return schema['description']
    if error.validator == 'required':
        missing = error.message.split("'")[1] if "'" in error.message else None
        for schema in schemas:
            prop = (schema.get('properties') or {}).get(missing)
            if isinstance(prop, dict) and prop.get('description'):
                return f'{missing}: {prop["description"]}'
    for schema in schemas[1:]:
        if schema.get('description'):
            return schema['description']
    return ''


def _rule_descriptions(schema):
    """Copy each ``allOf`` rule's description into its ``then`` (and each ``not`` rule's into
    the ``not``), where its errors point: a rule like "a series needs obs_dt" then reads as
    itself. Works on a copy of the loaded schema."""
    if isinstance(schema, dict):
        for rule in schema.get('allOf') or []:
            if isinstance(rule, dict) and rule.get('description'):
                for key in ('then', 'else'):
                    if isinstance(rule.get(key), dict):
                        rule[key].setdefault('description', rule['description'])
        for value in schema.values():
            _rule_descriptions(value)
    elif isinstance(schema, list):
        for value in schema:
            _rule_descriptions(value)
    return schema


def _leaves(error):
    """What is wrong, for an ``anyOf``/``oneOf``: every error of the branch the document
    meant (the one whose type it has, with the fewest errors), each followed down."""
    if not error.context:
        return [error]
    branches = {}
    for sub in error.context:
        branches.setdefault(sub.relative_schema_path[0], []).append(sub)

    def wrong_type(errs):
        return any(e.validator == 'type' and list(e.absolute_path) == list(error.absolute_path)
                   for e in errs)

    meant = min(branches.values(), key=lambda errs: (wrong_type(errs), len(errs)))
    return [leaf for sub in meant for leaf in _leaves(sub)]


def _message(error):
    """jsonschema's message, minus the parts that repeat the document or the schema."""
    if error.validator == 'not':
        return 'this combination of keys is not allowed'
    if error.validator == 'pattern':
        return f'{error.instance!r} is not in the form described'
    return error.message


def problems(schema_name, document):
    """``[(path, message, description)]`` of everything schema ``schema_name`` refuses in
    ``document``, in document order."""
    seen, out = set(), []
    validator = _validator(schema_name)
    errors = sorted(validator.iter_errors(document),
                    key=lambda e: [str(p) for p in e.absolute_path])
    for error in errors:
        for leaf in _leaves(error):
            message = _message(leaf)
            key = (tuple(leaf.absolute_path), message)
            if key not in seen:
                seen.add(key)
                out.append((_where(leaf.absolute_path, document), message,
                            _description(leaf, validator.schema, _owners(schema_name))))
    return out


def validate(schema_name, document, source='', what='file'):
    """Raise :class:`SchemaValidationError` if schema ``schema_name`` (e.g.
    ``OBS_DATA_SCHEMA``) refuses ``document``. ``source`` names the file in the message and
    ``what`` the kind of file. Without jsonschema (an app bundled before it was required) this
    warns once and checks nothing."""
    if not jsonschema_available():
        return
    key = _digest(schema_name, document)
    if key is not None and key in _valid:
        _valid.move_to_end(key)
        return
    found = problems(schema_name, document)
    if not found:
        if key is not None:
            _valid[key] = True
            if len(_valid) > VALID_CACHE_SIZE:
                _valid.popitem(last=False)
        return
    head = f'{source}: ' if source else ''
    lines = [f'{head}{len(found)} problem{"s" if len(found) != 1 else ""} with the {what}']
    for path, message, description in found[:MAX_LISTED]:
        lines.append(f'  at {path}: {message}')
        if description:
            lines.append(f'      ({description})')
    if len(found) > MAX_LISTED:
        lines.append(f'  and {len(found) - MAX_LISTED} more')
    raise SchemaValidationError('\n'.join(lines), [(p, m) for p, m, _ in found])
