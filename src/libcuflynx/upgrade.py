'''
``cuflynx-upgrade``: convert, once, files written for an earlier libcuflynx.

libcuflynx reads only the current forms of its input files. A reader that meets an older form
refuses the file and names the command here that converts it::

    cuflynx-upgrade module-config <*_modules_config.json>... [--module-library-dir <dir>]
    cuflynx-upgrade module-array <*_module_array|_vessel_array .json|.csv>...
    cuflynx-upgrade obs-data <*_obs_data.json>... --user-inputs <user_inputs.yaml>
    cuflynx-upgrade parameter-names --module-array <array> --module-library-dir <dir> ...

Each converts its files in place (a ``_vessel_array`` file is renamed) and prints what it
changed. Convert a model's module configs and module array before its parameter names: the
names are checked against the converted model (the par and pvn a one-output heart gains
among them). A file already in the current form is left as it is. ``--dry-run`` prints the changes
without writing them.

What each converts:

* ``module-config``: the older keys ``vessel_type``/``BC_type``/``module_file``/``module_type``
  (component) to ``module_type``/``module_subtype``/``component_file``/``component_type``;
  ``default_instance`` to ``default_parameterisation``; a ``multi_port`` of ``"False"`` is
  dropped; a supermodule's ``default_parameters`` CSV becomes its ``default`` parameterisation;
  a supermodule's inline ``submodules`` become its ``submodule_versions`` and its own module array
  file, and a parameterisation a submodule named (other than its version's default) is folded into
  the supermodule's (``--module-library-dir`` says where the submodule versions are).
* ``module-array``: the keys or columns ``vessel_type``/``BC_type``/``inp_vessels``/
  ``out_vessels`` to ``module_type``/``module_subtype``/``inp_instances``/``out_instances``;
  ``<prefix>_vessel_array.*`` is renamed ``<prefix>_module_array.*``; a heart with one output
  gets the pulmonary artery and vein (``par``, ``pvn``) the generator used to add for it.
* ``obs-data``: a bare list of data items, or an obs_data with no ``protocol_info``, gets the
  one-experiment ``protocol_info`` of ``pre_time`` and ``sim_time`` from user_inputs.yaml.
* ``parameter-names``: ``{variable}_{vessel}`` parameter names to ``<vessel>/<variable>``, and
  names joined with a single ``_`` across nested levels to ``__`` (see
  ``scripts/migrate_parameter_names.py``).
'''

import argparse
import json
import os
import re
import shutil
import sys

COMMAND = 'cuflynx-upgrade'

#: older module config key -> key; an older component entry's ``module_type`` is its component
MODULE_CONFIG_RENAMES = (('module_type', 'component_type'), ('module_file', 'component_file'),
                         ('vessel_type', 'module_type'), ('BC_type', 'module_subtype'),
                         ('inp_vessels', 'inp_instances'), ('out_vessels', 'out_instances'),
                         ('default_instance', 'default_parameterisation'))
#: older module array key or column -> key
MODULE_ARRAY_RENAMES = (('vessel_type', 'module_type'), ('BC_type', 'module_subtype'),
                        ('inp_vessels', 'inp_instances'), ('out_vessels', 'out_instances'))
OLD_ARRAY_SUFFIX, ARRAY_SUFFIX = '_vessel_array', '_module_array'


def hint(kind, path, extra=''):
    '''The sentence a reader's error ends with: how to convert ``path``.'''
    return f'Convert it once with: {COMMAND} {kind} {path}{extra}'


def _objects(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from _objects(value)


def _rename_keys(text, renames):
    '''``text`` with each JSON key of ``renames`` renamed, formatting and order kept.'''
    for old, new in renames:
        text = re.sub(r'"%s"(\s*):' % re.escape(old), r'"%s"\1:' % new, text)
    return text


def _write(path, text, dry_run, changes):
    print(f'{path}: ' + '; '.join(changes))
    if not dry_run:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)


# ------------------------------------------------------------------------------ module configs

def upgrade_module_config(path, dry_run=False, module_library_dirs=(), builtin_modules=True):
    '''Convert the module config at ``path``; returns the changes made (empty if none).
    ``module_library_dirs`` (and the built-in modules, unless ``builtin_modules`` is False) are
    where a supermodule's submodule versions are looked for, when a submodule names a
    parameterisation to fold into the supermodule's.'''
    text = open(path, encoding='utf-8-sig').read()
    doc = json.loads(text)
    changes = []
    old = [o for o in _objects(doc) if 'vessel_type' in o or 'BC_type' in o]
    if old:
        current = [o for o in _objects(doc)
                   if ('module_subtype' in o or 'component_type' in o) and o not in old]
        if current:
            raise SystemExit(f'{path} mixes entries in the older keys with entries in the current '
                             f'ones; convert the older entries by hand.')
        text = _rename_keys(text, MODULE_CONFIG_RENAMES)
        changes.append('older keys renamed')
    elif any('default_instance' in o for o in _objects(doc)):
        text = _rename_keys(text, MODULE_CONFIG_RENAMES[-1:])
        changes.append('default_instance renamed default_parameterisation')
    doc = json.loads(text)
    rewritten = len(changes)
    for entry in doc if isinstance(doc, list) else []:
        for key in ('entrance_ports', 'exit_ports', 'general_ports'):
            for port in entry.get(key) or []:
                if isinstance(port.get('multi_port'), str) and port['multi_port'].lower() == 'false':
                    del port['multi_port']
                    changes.append(f'multi_port "False" dropped ({port.get("port_type")})')
        if entry.get('default_parameters'):
            changes.append(_default_parameters_to_parameterisation(path, entry, dry_run))
        if isinstance(entry.get('submodules'), list):
            changes += _submodules_to_versions(path, entry, dry_run, module_library_dirs,
                                               builtin_modules)
    if len(changes) > rewritten:
        text = json.dumps(doc, indent=2, ensure_ascii=False) + '\n'
    if changes:
        _write(path, text, dry_run, changes)
    return changes


def _version_entries(module_library_dirs, builtin_modules):
    '''``{(module_type, module_subtype): (entry, config path)}`` of every config entry in the
    module sources, read as written (so older supermodules are found too).'''
    # listed, not read through ModuleSources, which refuses the older forms being converted
    from libcuflynx.utilities.module_library import (_is_any_json, _is_library_config_json,
                                                     _list_flat, _list_recursive)
    from libcuflynx.utilities.package_resources import builtin_modules_dir
    from libcuflynx.utilities.paths import default_module_config_user_dir
    files = []
    for directory in module_library_dirs:
        files += _list_recursive(directory, _is_library_config_json)
    if builtin_modules:
        files += (_list_flat(default_module_config_user_dir(), _is_any_json)
                  + _list_flat(builtin_modules_dir(), _is_any_json))
    found = {}
    for config in files:
        try:
            entries = json.load(open(config, encoding='utf-8-sig'))
        except (OSError, ValueError):
            continue
        for e in entries if isinstance(entries, list) else []:
            if isinstance(e, dict) and 'module_type' in e and 'module_subtype' in e:
                found.setdefault((e['module_type'], e['module_subtype']), (e, config))
    return found


def _read_rows(path):
    import csv
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), [dict(r) for r in reader]


def _write_rows(path, columns, rows):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows([{c: r.get(c, '') for c in columns} for r in rows])


def _submodule_rows(sub, version, config, parameterisation):
    '''The rows of ``parameterisation`` of the submodule version ``version`` (from ``config``),
    named as a row of the supermodule's parameterisation: ``<sub>/<var>``, a global plainly.'''
    from libcuflynx.utilities.parameterisations import parameterisation_parameters_path
    _, rows = _read_rows(parameterisation_parameters_path(config, parameterisation))
    supermodule = version.get('module_format') == 'supermodule'
    globals_ = {v[0] for v in version.get('variables_and_units') or []
                if len(v) > 3 and v[3] == 'global_constant'}
    out = []
    for row in rows:
        name = row['variable_name'].strip()
        if name.startswith('global/'):
            name = name[len('global/'):]
        elif (supermodule and '/' in name) or (not supermodule and name not in globals_):
            name = f'{sub}/{name}'
        out.append(dict(row, variable_name=name))
    return out


def _submodules_to_versions(config_path, entry, dry_run, module_library_dirs, builtin_modules):
    '''A supermodule entry's inline ``submodules``: its instances to its own module array file,
    ``submodule_versions`` in the entry, and each submodule's own parameterisation (other than
    its version's default) folded into the supermodule's parameterisations.'''
    from libcuflynx.utilities.parameterisations import (available_parameterisations,
                                                        parameterisation_parameters_path)
    records = entry.pop('submodules')
    versions = list(dict.fromkeys((r['module_type'], r['module_subtype']) for r in records))
    entry['submodule_versions'] = [{'module_type': t, 'module_subtype': s} for t, s in versions]
    named = {r['name']: r.pop('parameterisation') for r in records if 'parameterisation' in r}
    array = os.path.join(os.path.dirname(os.path.abspath(config_path)),
                         f'{entry["module_type"]}_{entry["module_subtype"]}_module_array.json')
    if os.path.exists(array):
        raise SystemExit(f'{config_path}: {array} exists already; remove it or convert by hand.')
    changes = [f'submodules moved to {os.path.basename(array)}, with submodule_versions']
    by_name = {r['name']: r for r in records}
    folded = []
    if named:
        sources = _version_entries(module_library_dirs, builtin_modules)
        for sub, parameterisation in named.items():
            key = (by_name[sub]['module_type'], by_name[sub]['module_subtype'])
            if key not in sources:
                raise SystemExit(f'{config_path}: submodule "{sub}" names parameterisation '
                                 f'"{parameterisation}" of {key}, which no module source given '
                                 f'has; pass --module-library-dir for its library.')
            version, config = sources[key]
            if parameterisation == version.get('default_parameterisation',
                                               version.get('default_instance')):
                continue
            folded += _submodule_rows(sub, version, config, parameterisation)
            changes.append(f'"{sub}"\'s parameterisation "{parameterisation}" folded in')
    # written once every named parameterisation is found, so a refusal leaves nothing behind
    if not dry_run:
        with open(array, 'w', encoding='utf-8') as f:
            f.write('[\n' + ',\n'.join(' ' + json.dumps(r) for r in records) + '\n]\n')
    if folded:
        targets = available_parameterisations(config_path)
        if not entry.get('default_parameterisation'):
            entry['default_parameterisation'] = 'default'
        if entry['default_parameterisation'] not in targets:
            targets.append(entry['default_parameterisation'])
        for name in targets:
            path = parameterisation_parameters_path(config_path, name)
            columns, rows = (_read_rows(path) if os.path.isfile(path) else
                             (['variable_name', 'units', 'value', 'data_reference'], []))
            have = {r['variable_name'].strip() for r in rows}
            rows += [r for r in folded if r['variable_name'] not in have]
            if not dry_run:
                _write_rows(path, columns, rows)
    return changes


def _default_parameters_to_parameterisation(config_path, entry, dry_run):
    '''Move a supermodule's default_parameters CSV to ``parameterisations/default/``.'''
    from libcuflynx.utilities.parameterisations import PARAMETERISATIONS_DIR
    if entry.get('default_parameterisation'):
        raise SystemExit(f'{config_path}: supermodule ({entry.get("module_type")}) has both '
                         f'default_parameters and a default_parameterisation; move the '
                         f'default_parameters rows into that parameterisation by hand.')
    config_dir = os.path.dirname(os.path.abspath(config_path))
    source = os.path.join(config_dir, entry.pop('default_parameters'))
    target_dir = os.path.join(config_dir, PARAMETERISATIONS_DIR, 'default')
    target = os.path.join(target_dir, 'default_parameters.csv')
    if os.path.exists(target):
        raise SystemExit(f'{config_path}: {target} exists already; move {source} by hand.')
    entry['default_parameterisation'] = 'default'
    if not dry_run:
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(source, target)
    return f'default_parameters moved to {os.path.relpath(target, config_dir)}'


# ------------------------------------------------------------------------------- module arrays

def _heart_rows(records):
    '''The par/pvn records a heart with one output needs, with the heart connected to them.'''
    from libcuflynx.utilities.config_schemas import is_heart_module_type
    hearts = [r for r in records if is_heart_module_type(r.get('module_type'))]
    if len(hearts) != 1 or len(hearts[0].get('out_instances') or []) != 1:
        return []
    heart = hearts[0]
    heart['out_instances'] = list(heart.get('out_instances') or []) + ['par']
    heart['inp_instances'] = list(heart.get('inp_instances') or []) + ['pvn']
    return [{'name': 'par', 'module_type': 'arterial_simple', 'module_subtype': 'vp',
             'inp_instances': [heart['name']], 'out_instances': ['pvn']},
            {'name': 'pvn', 'module_type': 'arterial_simple', 'module_subtype': 'vp',
             'inp_instances': ['par'], 'out_instances': [heart['name']]}]


def upgrade_module_array(path, dry_run=False):
    '''Convert the module array at ``path`` (and rename a ``_vessel_array`` file); returns the
    changes made.'''
    import pandas as pd
    changes = []
    if path.lower().endswith('.json'):
        text = open(path, encoding='utf-8-sig').read()
        if any(k in o for o in _objects(json.loads(text)) for k, _ in MODULE_ARRAY_RENAMES):
            text = _rename_keys(text, MODULE_ARRAY_RENAMES)
            changes.append('older keys renamed')
        records = json.loads(text)
        for record in records:
            for key in ('inp_instances', 'out_instances'):
                if isinstance(record.get(key), str):
                    record[key] = record[key].split()
        added = _heart_rows(records)
        if added:
            from libcuflynx.utilities.config_schemas import dump_vessel_records
            text = dump_vessel_records(records + added)
            changes.append('par and pvn added for the one-output heart')
    else:
        df = pd.read_csv(path, dtype=str, na_filter=False)
        df = df.rename(columns=lambda c: str(c).strip()).apply(lambda col: col.str.strip())
        renames = dict(MODULE_ARRAY_RENAMES)
        if any(c in renames for c in df.columns):
            df = df.rename(columns=renames)
            order = ['name', 'module_type', 'module_subtype', 'inp_instances', 'out_instances']
            df = df[order + [c for c in df.columns if c not in order]]
            changes.append('older columns renamed')
        records = [{**row, 'inp_instances': row['inp_instances'].split(),
                    'out_instances': row['out_instances'].split()}
                   for row in df.to_dict('records')]
        added = _heart_rows(records)
        if added:
            changes.append('par and pvn added for the one-output heart')
        rows = [{**r, 'inp_instances': ' '.join(r['inp_instances']),
                 'out_instances': ' '.join(r['out_instances'])} for r in records + added]
        text = pd.DataFrame(rows, columns=df.columns).fillna('').to_csv(index=False)
    stem, ext = os.path.splitext(path)
    target = path
    if stem.endswith(OLD_ARRAY_SUFFIX):
        target = stem[:-len(OLD_ARRAY_SUFFIX)] + ARRAY_SUFFIX + ext
        if os.path.exists(target):
            raise SystemExit(f'{path}: {target} exists already; remove one of them.')
        changes.append(f'renamed {os.path.basename(target)}')
    if changes:
        _write(target, text, dry_run, changes)
        if target != path and not dry_run:
            os.remove(path)
    return changes


# ------------------------------------------------------------------------------------ obs_data

def upgrade_obs_data(path, pre_time, sim_time, dry_run=False):
    '''Give the obs_data at ``path`` a protocol_info (one experiment of ``pre_time`` then
    ``sim_time``) if it has none; returns the changes made.'''
    text = open(path, encoding='utf-8-sig').read()
    doc = json.loads(text)
    if isinstance(doc, dict) and 'protocol_info' in doc:
        return []
    protocol_info = {'pre_times': [float(pre_time)], 'sim_times': [[float(sim_time)]],
                     'params_to_change': {}}
    if isinstance(doc, list):
        doc = {'protocol_info': protocol_info, 'data_items': doc}
        change = 'bare list wrapped in {protocol_info, data_items}'
    else:
        doc = {'protocol_info': protocol_info, **doc}
        change = 'protocol_info added'
    _write(path, json.dumps(doc, indent=2, ensure_ascii=False) + '\n', dry_run,
           [f'{change} (pre_time {pre_time}, sim_time {sim_time})'])
    return [change]


def _times(args):
    if args.user_inputs:
        import yaml
        with open(args.user_inputs) as f:
            inputs = yaml.safe_load(f) or {}
        pre, sim = inputs.get('pre_time', args.pre_time), inputs.get('sim_time', args.sim_time)
    else:
        pre, sim = args.pre_time, args.sim_time
    if pre is None or sim is None:
        raise SystemExit('obs-data needs pre_time and sim_time: give --user-inputs '
                         '<user_inputs.yaml> (its pre_time and sim_time), or --pre-time and '
                         '--sim-time.')
    return pre, sim


# ------------------------------------------------------------------------------------------ CLI

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['parameter-names']:
        from libcuflynx.scripts.migrate_parameter_names import main as migrate
        return migrate(argv[1:])
    parser = argparse.ArgumentParser(
        prog=COMMAND, description=__doc__.split('\n\n')[1],
        epilog='Not a pipeline stage: it takes file paths, not user_inputs.yaml (obs-data reads '
               "the user_inputs.yaml given with --user-inputs for its pre_time and sim_time).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, help_text in (('module-config', 'module config files'),
                            ('module-array', 'module (or vessel) array files'),
                            ('obs-data', 'obs_data files')):
        sub = commands.add_parser(name, help=f'convert {help_text}')
        sub.add_argument('paths', nargs='+')
        sub.add_argument('--dry-run', action='store_true', help='print, do not write')
        if name == 'module-config':
            sub.add_argument('--module-library-dir', dest='module_library_dirs', action='append',
                             default=[], metavar='DIR',
                             help="a module library (repeatable): where a supermodule's "
                                  "submodule versions are, to fold a parameterisation a "
                                  "submodule names into the supermodule's")
            sub.add_argument('--no-builtin-modules', action='store_true',
                             help="leave libcuflynx's built-in modules out")
        if name == 'obs-data':
            sub.add_argument('--user-inputs', help="the user_inputs.yaml whose pre_time and "
                                                   "sim_time the files were run with")
            sub.add_argument('--pre-time', type=float)
            sub.add_argument('--sim-time', type=float)
    commands.add_parser('parameter-names', help='convert parameter names (see '
                                                f'{COMMAND} parameter-names --help)')
    args = parser.parse_args(argv)
    changed = 0
    for path in args.paths:
        if args.command == 'module-config':
            changed += bool(upgrade_module_config(path, args.dry_run, args.module_library_dirs,
                                                  not args.no_builtin_modules))
        elif args.command == 'module-array':
            changed += bool(upgrade_module_array(path, args.dry_run))
        else:
            changed += bool(upgrade_obs_data(path, *_times(args), dry_run=args.dry_run))
    print(f'{changed} of {len(args.paths)} file(s) {"would be " if args.dry_run else ""}changed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
