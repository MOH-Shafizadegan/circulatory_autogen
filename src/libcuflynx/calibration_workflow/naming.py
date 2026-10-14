'''
How a parameter calibrated in one workflow step is named in another model.

Every step model is generated alone, as a single module-array record named ``STEP_VESSEL``
(``mod``). A params_for_id row ``(vessel_name, param_name)`` of a step therefore names
``param_name_for_gen(vessel_name, param_name)`` in that step's model: ``rho_M_mod`` for a
plain module, ``g_leak_mod_i_leak_Na`` for a supermodule's submodule, ``R`` for a global.

When that module is the submodule at ``path`` (``i_M``, or ``soma_i_M`` two levels down) of
a supermodule that is itself generated as ``mod``, supermodule expansion
(``utilities/supermodules.py``) renames its vessel ``mod[_x]`` to ``mod_<path>[_x]``, so the
same parameter is ``param_name_for_gen('mod_<path>[_x]', param_name)`` there. In the
supermodule's own parameterisation parameters CSV it is written ``<path>/<var>``
(``i_M/rho_M``, ``soma/i_M/rho_M`` when nested), as ``utilities/parameter_names.py`` reads it.

These are the rules circulatory-autogen-modules' ``calibrate apply`` follows by hand.
'''

from libcuflynx.parsers.PrimitiveParsers import param_name_for_gen

STEP_VESSEL = 'mod'
GLOBAL_VESSEL = 'global'


def is_global(vessel):
    return vessel == GLOBAL_VESSEL


def map_vessel(vessel, path, step_vessel=STEP_VESSEL):
    '''``vessel`` of a model generated as ``step_vessel``, renamed for the model in which
    that module is the submodule at ``path`` ('' for the same model).

    ``('mod', 'i_M') -> 'mod_i_M'``; ``('mod_A', 'soma') -> 'mod_soma_A'``; a global is
    unchanged. A vessel that does not belong to ``step_vessel`` is a ValueError.
    '''
    if is_global(vessel) or not path:
        return vessel
    if vessel == step_vessel:
        return f'{step_vessel}_{path}'
    prefix = step_vessel + '_'
    if vessel.startswith(prefix):
        return f'{step_vessel}_{path}_{vessel[len(prefix):]}'
    raise ValueError(f'vessel "{vessel}" is not part of the step model, whose single record '
                     f'is "{step_vessel}"; params_for_id vessel_name must be "{step_vessel}", '
                     f'"{step_vessel}_<submodule>" or "{GLOBAL_VESSEL}".')


def model_name(vessel, param):
    '''The name of ``param`` of ``vessel`` in a generated (flat) model.'''
    return param_name_for_gen(vessel, param)


def parameterisation_name(vessel, param, submodule_paths=(), step_vessel=STEP_VESSEL):
    '''The name of ``param`` of ``vessel`` in the parameterisation parameters CSV of the module
    generated as ``step_vessel``, whose submodules are ``submodule_paths`` (tuples, as
    ``supermodules.submodule_path_parts`` gives them): ``('mod', 'p') -> 'p'``,
    ``('mod_A', 'p') -> 'A/p'``, ``('mod_soma_i_M', 'p') -> 'soma/i_M/p'``; a global
    unchanged.'''
    if is_global(vessel) or vessel == step_vessel:
        return param
    prefix = step_vessel + '_'
    if vessel.startswith(prefix):
        joined = vessel[len(prefix):]
        for parts in submodule_paths:
            if '_'.join(parts) == joined:
                return '/'.join(tuple(parts) + (param,))
    raise ValueError(f'vessel "{vessel}" is not a submodule of the model generated as '
                     f'"{step_vessel}" (its submodules: '
                     f'{sorted("/".join(p) for p in submodule_paths)}).')


def relative_path(inner, outer):
    '''The submodule path of a module at ``inner`` relative to the module at ``outer``, both
    paths in the same model ('' is that model itself); None when ``inner`` is not inside
    ``outer``.'''
    if not outer:
        return inner
    if inner == outer:
        return ''
    if inner.startswith(outer + '_'):
        return inner[len(outer) + 1:]
    return None


def host_model_name(name):
    '''The model name of a row of a module array's parameters file: ``aortic_root/C`` and
    ``C_aortic_root`` are ``C_aortic_root``; ``global/R`` and ``R`` are ``R``.'''
    from libcuflynx.utilities.parameter_names import is_path_name, split_path_name
    if not is_path_name(name):
        return name
    path, variable = split_path_name(name)
    return f'{variable}_{"_".join(path)}' if path else variable


class VesselMap:
    '''How the vessels of a step's model are named in the workflow's target model, and back.

    ``to_target(vessel)`` names a vessel of the step's model in the target. ``from_target(vessel)``
    names a target vessel in the step's model, or None when the step's model has no such vessel
    (``vessels`` lists the ones it has). A global (``global``) is the same everywhere.'''

    def __init__(self, vessels):
        self.vessels = set(vessels)

    def to_target(self, vessel):
        return vessel if is_global(vessel) else self._to(vessel)

    def from_target(self, vessel):
        if is_global(vessel):
            return vessel
        name = self._from(vessel)
        return name if name in self.vessels else None


class Nested(VesselMap):
    '''A library module generated as ``mod``, inside a library supermodule target (also
    generated as ``mod``) at the submodule ``path`` (``'i_M'``, ``'soma_i_M'``; ``''`` for
    the target module itself).'''

    def __init__(self, path, vessels):
        super().__init__(vessels)
        self.path = path

    def _to(self, vessel):
        return map_vessel(vessel, self.path)

    def _from(self, vessel):
        if not self.path:
            return vessel
        root = f'{STEP_VESSEL}_{self.path}'
        if vessel == root:
            return STEP_VESSEL
        if vessel.startswith(root + '_'):
            return f'{STEP_VESSEL}_{vessel[len(root) + 1:]}'
        return None


class Placed(VesselMap):
    '''A library module generated as ``mod``, that is the record ``vessel`` of a module-array
    target: ``mod`` is ``vessel`` and ``mod_<sub>`` is ``vessel_<sub>``.'''

    def __init__(self, vessel, vessels):
        super().__init__(vessels)
        self.vessel = vessel

    def _to(self, vessel):
        if vessel == STEP_VESSEL:
            return self.vessel
        if vessel.startswith(STEP_VESSEL + '_'):
            return f'{self.vessel}_{vessel[len(STEP_VESSEL) + 1:]}'
        raise ValueError(f'vessel "{vessel}" is not part of the step model, whose single record '
                         f'is "{STEP_VESSEL}".')

    def _from(self, vessel):
        if vessel == self.vessel:
            return STEP_VESSEL
        if vessel.startswith(self.vessel + '_'):
            return f'{STEP_VESSEL}_{vessel[len(self.vessel) + 1:]}'
        return None


class Renamed(VesselMap):
    '''A module array's own vessels: named the same in the target unless ``mapping`` (its
    vessel -> the target's) renames them.'''

    def __init__(self, mapping, vessels):
        super().__init__(vessels)
        self.mapping = dict(mapping)
        self.inverse = {v: k for k, v in self.mapping.items()}

    def _to(self, vessel):
        return self.mapping.get(vessel, vessel)

    def _from(self, vessel):
        if vessel in self.inverse:
            return self.inverse[vessel]
        return None if vessel in self.mapping else vessel


def between(source, receiver, vessel):
    '''``vessel`` of the step whose map is ``source``, named in the step whose map is
    ``receiver`` (through the target's naming); None when the receiving model has no such
    vessel.'''
    return receiver.from_target(source.to_target(vessel))
