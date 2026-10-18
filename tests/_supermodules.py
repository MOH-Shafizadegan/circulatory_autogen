"""Writing supermodule versions for tests.

A test states a supermodule as one dict with its instance records inline under
``submodules``. ``split_supermodule`` turns it into what a library holds: the config entry,
with the ``submodule_versions`` its records use, and the records of its module array
(``config_schemas.supermodule_array_path``). ``write_supermodule(s)`` writes both.
"""
import json
import os


def split_supermodule(entry):
    """``(config entry, array records)`` of a supermodule dict with inline ``submodules``."""
    entry = dict(entry)
    records = [dict(r) for r in entry.pop('submodules')]
    versions = dict.fromkeys((r['module_type'], r['module_subtype']) for r in records)
    entry['submodule_versions'] = [{'module_type': t, 'module_subtype': s} for t, s in versions]
    return entry, records


def array_file_name(entry):
    return f'{entry["module_type"]}_{entry["module_subtype"]}_module_array.json'


def write_supermodules(directory, entries, config_name):
    """Write ``entries`` (supermodule dicts with inline ``submodules``, or other config
    entries, kept as they are) to ``<directory>/<config_name>``, each supermodule's records to
    its module array beside it. Returns the config path."""
    os.makedirs(directory, exist_ok=True)
    written = []
    for entry in entries:
        if isinstance(entry.get('submodules'), list):
            entry, records = split_supermodule(entry)
            with open(os.path.join(directory, array_file_name(entry)), 'w') as f:
                json.dump(records, f, indent=1)
        written.append(entry)
    path = os.path.join(str(directory), config_name)
    with open(path, 'w') as f:
        json.dump(written, f, indent=1)
    return path


def write_supermodule(directory, entry, config_name=None):
    """Write one supermodule (``entry`` with inline ``submodules``) to
    ``<directory>/<module_type>_<module_subtype>_modules_config.json`` (or ``config_name``)
    and its module array beside it. Returns the config path."""
    name = config_name or f'{entry["module_type"]}_{entry["module_subtype"]}_modules_config.json'
    return write_supermodules(directory, [entry], name)
