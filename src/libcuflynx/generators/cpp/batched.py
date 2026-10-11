"""C for the BE_batched_c solver: libCellML's C for the whole model, plus the kernels of
:mod:`libcuflynx.generators.kernel_ir` and a runtime, built as one shared library with CMake.

Files written to the build folder (templates in ``templates/batched_*.j2``):

=========================  ===========================================================================
``model0d_core.c/.h``      libCellML's C, one variables array (as ``model_type: cpp`` writes it):
                           initial values, computed constants, and the algebraic variables for outputs
``batched_kernels.c/.h``   each kernel's rates and exact Jacobian, over a block's own states and inputs
``batched_runtime.c``      the integrator (adaptive backward Euler, modified Newton), OpenMP over blocks
``CMakeLists.txt``         the library ``batched_model``
``blocks.json``            the blocks: kernel, model state indices, model variable indices of its inputs
=========================  ===========================================================================

The folder is named after a digest of its sources, so a model already built is loaded, not rebuilt.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import tempfile
import time

import numpy as np

from libcuflynx.generators import kernel_ir
from libcuflynx.generators.cpp.generator import core_code, template_environment

TARGET = 'batched_model'

_C_FUNCTIONS = {
    'sqrt': 'sqrt', 'exp': 'exp', 'ln': 'log', 'log10': 'log10', 'abs': 'fabs', 'sign': 'bk_sign',
    'floor': 'floor', 'ceil': 'ceil', 'trunc': 'trunc', 'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
    'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh', 'asin': 'asin', 'acos': 'acos', 'atan': 'atan',
    'asinh': 'asinh', 'acosh': 'acosh', 'atanh': 'atanh', 'pow': 'pow', 'min': 'bk_min', 'max': 'bk_max',
    'rem': 'fmod',
}
_C_INFIX = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>=',
            'eq': '==', 'ne': '!=', 'and': '&&', 'or': '||'}


class BatchedBuildError(RuntimeError):
    pass


def c_literal(value):
    if math.isnan(value):
        return 'NAN'
    if math.isinf(value):
        return 'INFINITY' if value > 0 else '(-INFINITY)'
    text = repr(float(value))
    return text if ('.' in text or 'e' in text) else text + '.0'


def c_expression(e):
    """An expression of kernel_ir as C (states ``y[i]``, inputs ``in[k]``, time ``t``)."""
    op = e[0]
    if op == 'num':
        return c_literal(e[1])
    if op == 'y':
        return f'y[{e[1]}]'
    if op == 'in':
        return f'in[{e[1]}]'
    if op == 't':
        return 't'
    if op == 'tmp':
        return e[1]
    if op in _C_INFIX:
        return f'({c_expression(e[1])} {_C_INFIX[op]} {c_expression(e[2])})'
    if op == 'neg':
        return f'(-{c_expression(e[1])})'
    if op == 'not':
        return f'(!{c_expression(e[1])})'
    if op == 'xor':
        return f'(({c_expression(e[1])} != 0) != ({c_expression(e[2])} != 0))'
    if op == 'select':
        return f'({c_expression(e[1])} ? {c_expression(e[2])} : {c_expression(e[3])})'
    if op in _C_FUNCTIONS:
        return f'{_C_FUNCTIONS[op]}({", ".join(c_expression(x) for x in e[1:])})'
    raise BatchedBuildError(f'no C for operation {op}')


def _reads_time(program):
    def walk(e):
        return e[0] == 't' or (e[0] not in ('num', 'y', 'in', 'tmp') and any(walk(x) for x in e[1:]))
    return any(walk(e) for _, e in program.assignments) or any(walk(e) for e in program.rates)


def kernel_context(kernel, n_blocks):
    """The template context of one kernel: its C statements."""
    n = kernel.n_states
    rates = [f'const double {name} = {c_expression(e)};' for name, e in kernel.rates.assignments]
    rates += [f'dy[{i}] = {c_expression(e)};' for i, e in enumerate(kernel.rates.rates)]
    jacobian = [f'const double {name} = {c_expression(e)};' for name, e in kernel.jacobian.assignments]
    jacobian += [f'J[{r * n + c}] = {c_expression(e)};  /* d rate {r} / d state {c} */'
                 for r, c, e in kernel.jacobian.entries]
    return {'index': kernel.index, 'n_states': n, 'n_inputs': kernel.n_inputs, 'nonzeros': kernel.nonzeros,
            'reads_time': _reads_time(kernel.rates), 'n_blocks': n_blocks, 'rates': rates, 'jacobian': jacobian}


def sources(am, km, model_name='model'):
    """{file name: text} of the library for an analysed model and its kernel model."""
    if am.externalVariableCount():
        raise BatchedBuildError('BE_batched_c does not take models with external variables yet')
    env = template_environment()
    counts = {}
    for b in km.blocks:
        counts[b.kernel] = counts.get(b.kernel, 0) + 1
    ctx = {'model_name': model_name, 'kernels': [kernel_context(k, counts[k.index]) for k in km.kernels],
           'n_blocks': len(km.blocks), 'n_states': km.state_count, 'target': TARGET}
    interface, implementation = core_code(am)
    blocks = [{'kernel': b.kernel, 'states': b.states, 'inputs': b.inputs} for b in km.blocks]
    return {
        'model0d_core.h': interface,
        'model0d_core.c': implementation,
        'batched_kernels.h': env.get_template('batched_kernels.h.j2').render(**ctx),
        'batched_kernels.c': env.get_template('batched_kernels.c.j2').render(**ctx),
        'batched_runtime.c': env.get_template('batched_runtime.c.j2').render(**ctx),
        'CMakeLists.txt': env.get_template('batched_CMakeLists.txt.j2').render(**ctx),
        'blocks.json': json.dumps(blocks),
    }


def build_dir_for(files, parent):
    digest = hashlib.sha1()
    for name in sorted(files):
        digest.update(name.encode())
        digest.update(files[name].encode())
    return os.path.join(parent, f'batched_c_{digest.hexdigest()[:12]}')


def build(files, parent=None, verbose=False):
    """Write ``files`` to a folder named after their digest under ``parent`` (the system temp folder by
    default) and build the library there unless it is already built; return (library path, seconds spent
    building, 0 when it was already built). A lock file serialises concurrent builds of the same sources."""
    import fcntl
    from libcuflynx.coupling.model0d_lib import _library_path, build_library
    parent = parent or os.path.join(tempfile.gettempdir(), 'libcuflynx_batched_c')
    out = build_dir_for(files, parent)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, '.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = _library_path(os.path.join(out, 'build'), TARGET)
        if path is not None:
            return path, 0.0
        t0 = time.perf_counter()
        for name, text in files.items():
            with open(os.path.join(out, name), 'w') as f:
                f.write(text)
        try:
            path = build_library(out, os.path.join(out, 'build'), verbose=verbose, target=TARGET)
        except Exception as exc:
            raise BatchedBuildError(str(exc)) from exc
        return path, time.perf_counter() - t0


class BatchedLibrary:
    """The built library of one model, through ctypes."""

    def __init__(self, path):
        self.path = path
        lib = ctypes.CDLL(path)
        d, i, l, p = ctypes.c_double, ctypes.c_int, ctypes.c_long, ctypes.c_void_p
        dp, ip, lp = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_long)
        sigs = {
            'bk_kernel_count': ([], i), 'bk_kernel_states': ([i], i), 'bk_kernel_inputs': ([i], i),
            'bk_kernel_nonzeros': ([i], i),
            'bk_kernel_rates': ([i, d, dp, dp, dp], None), 'bk_kernel_jacobian': ([i, d, dp, dp, dp], None),
            'bk_create': ([i, ip, ip, ip, ip, ip, i, i], p), 'bk_destroy': ([p], None),
            'bk_set_options': ([p, d, d, d, d, d, i, i, l, i], None), 'bk_invalidate': ([p], None),
            'bk_set_time_origin': ([p, d], None), 'bk_run': ([p, dp, dp, d, i, dp, dp, i], i),
            'bk_block_status': ([p, i, dp], i), 'bk_stats': ([p, lp], None), 'bk_openmp_threads': ([], i),
            'bk_initialise': ([dp, dp, dp], None), 'bk_compute_computed_constants': ([dp], None),
            'bk_outputs': ([i, dp, dp, dp, dp, i], None),
        }
        for name, (argtypes, restype) in sigs.items():
            fn = getattr(lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
        self._lib = lib

    def __getattr__(self, name):
        return getattr(self._lib, name)


def ptr(a, ctype=ctypes.c_double):
    return a.ctypes.data_as(ctypes.POINTER(ctype)) if a is not None else None


STAT_NAMES = ('steps', 'rejected_steps', 'rate_evaluations', 'jacobians', 'finite_difference_jacobians',
              'factorisations')


class BatchedSimulation:
    """The blocks of one model and their integrator state, in a built library."""

    def __init__(self, library, blocks, n_states, n_variables):
        self.lib = library
        self.n_states = n_states
        self.n_variables = n_variables
        kernel = np.array([b['kernel'] for b in blocks], dtype=np.int32)
        state_offsets = np.cumsum([0] + [len(b['states']) for b in blocks]).astype(np.int32)
        input_offsets = np.cumsum([0] + [len(b['inputs']) for b in blocks]).astype(np.int32)
        states = np.array([s for b in blocks for s in b['states']], dtype=np.int32)
        inputs = np.array([k for b in blocks for k in b['inputs']] or [0], dtype=np.int32)
        self._keep = (kernel, state_offsets, input_offsets, states, inputs)
        self.n_blocks = len(blocks)
        self._h = library.bk_create(self.n_blocks, ptr(kernel, ctypes.c_int), ptr(state_offsets, ctypes.c_int),
                                    ptr(states, ctypes.c_int), ptr(input_offsets, ctypes.c_int),
                                    ptr(inputs, ctypes.c_int), n_states, n_variables)

    def set_options(self, rtol=1e-4, atol=1e-12, max_step=0.0, min_step=1e-14, newton_tol=0.01, max_newton=6,
                    max_jacobian_age=50, max_steps=10_000_000, finite_difference_jacobian=False):
        self.lib.bk_set_options(self._h, rtol, atol, max_step or 0.0, min_step, newton_tol, max_newton,
                                max_jacobian_age, int(max_steps), int(bool(finite_difference_jacobian)))

    def invalidate(self):
        self.lib.bk_invalidate(self._h)

    def set_time_origin(self, origin):
        self.lib.bk_set_time_origin(self._h, origin)

    def run(self, states, variables, t0, t_out, out=None, threads=1):
        """Integrate from t0 through t_out; ``states`` (updated to the last output time), ``variables`` and
        ``out`` (len(t_out) x n_states) are float64 numpy arrays. Returns [(block, status, time)] of failures."""
        t_out = np.ascontiguousarray(t_out, dtype=float)
        failed = self.lib.bk_run(self._h, ptr(states), ptr(variables), float(t0), len(t_out), ptr(t_out),
                                 ptr(out), int(threads))
        if not failed:
            return []
        out_failures = []
        t = ctypes.c_double()
        for b in range(self.n_blocks):
            status = self.lib.bk_block_status(self._h, b, ctypes.byref(t))
            if status:
                out_failures.append((b, status, t.value))
        return out_failures

    def stats(self):
        buf = (ctypes.c_long * 6)()
        self.lib.bk_stats(self._h, buf)
        return dict(zip(STAT_NAMES, list(buf)))

    def close(self):
        if self._h:
            self.lib.bk_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
