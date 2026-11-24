"""Реализации трёх ускоренных подходов и исходного варианта (вариант 6).

Все функции ядра возвращают один и тот же набор чисел:
    (pi_est, integral_est, walk_mean, walk_std)
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import numpy as np
from numba import njit, prange

from variant_6 import (
    N_STEPS,
    make_workload as make_workload_py,
    run_baseline as run_baseline_py,
    summarize as summarize_py,
)

SEED = 23
Output = Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
def shape_for_size(size: int) -> Tuple[int, int, int, int]:
    """(n_pi, n_int, n_walks, n_steps) — те же формулы, что в variant_6.py."""
    return size * 1000, size * 500, max(1, size), N_STEPS


def make_workload_numpy(size: int):
    """Те же распределения и seed, что у исходного генератора, но в NumPy."""
    n_pi, n_int, n_walks, n_steps = shape_for_size(size)
    rng = np.random.default_rng(SEED)
    xs = rng.random(n_pi)
    ys = rng.random(n_pi)
    us = rng.random(n_int)
    steps = rng.integers(0, 2, size=n_walks * n_steps, dtype=np.int8) * 2 - 1
    return xs, ys, us, steps, n_walks, n_steps


def as_numpy_workload(workload):
    """Перевод исходных array('d')/array('b') в непрерывные массивы NumPy."""
    xs, ys, us, steps, n_walks, n_steps = workload
    return (
        np.ascontiguousarray(xs, dtype=np.float64),
        np.ascontiguousarray(ys, dtype=np.float64),
        np.ascontiguousarray(us, dtype=np.float64),
        np.ascontiguousarray(steps, dtype=np.int8),
        n_walks,
        n_steps,
    )


# ---------------------------------------------------------------------------
# A. Исходный чистый Python
# ---------------------------------------------------------------------------
def python_kernel(workload) -> Output:
    return summarize_py(run_baseline_py(workload))


def python_e2e(size: int) -> Output:
    return python_kernel(make_workload_py(size))


# ---------------------------------------------------------------------------
# B. NumPy
# ---------------------------------------------------------------------------
def numpy_kernel(xs, ys, us, steps, n_walks: int, n_steps: int) -> Output:
    """Прямая векторизация: три свёртки/редукции над массивами."""
    inside = int(np.count_nonzero(xs * xs + ys * ys <= 1.0))
    pi_est = 4.0 * inside / xs.size

    integral_est = float(np.mean(np.exp(-us * us)))

    finals = steps.reshape(n_walks, n_steps).sum(axis=1)
    mean = float(finals.mean())
    var = float(np.mean((finals.astype(np.float64) - mean) ** 2))
    return pi_est, integral_est, mean, math.sqrt(var)


def numpy_lowmem_kernel(xs, ys, us, steps, n_walks: int, n_steps: int) -> Output:
    """Т же формулы, но с переиспользованием буферов (меньше пиковая память)."""
    buf = np.empty(xs.shape[0], dtype=np.float64)
    np.square(xs, out=buf)
    inside = int(np.count_nonzero(buf + ys * ys <= 1.0))
    pi_est = 4.0 * inside / xs.size

    ubuf = np.square(us)
    np.negative(ubuf, out=ubuf)
    np.exp(ubuf, out=ubuf)
    integral_est = float(ubuf.mean())

    finals = steps.reshape(n_walks, n_steps).sum(axis=1)
    mean = float(finals.mean())
    var = float(np.mean(np.square(finals.astype(np.float64) - mean)))
    return pi_est, integral_est, mean, math.sqrt(var)


def numpy_e2e(size: int) -> Output:
    return numpy_kernel(*make_workload_numpy(size))


# ---------------------------------------------------------------------------
# C. Numba (последовательная и параллельная версии)
# ---------------------------------------------------------------------------
@njit(cache=False, parallel=False)
def nb_kernel_serial(xs, ys, us, steps, n_walks, n_steps):
    inside = 0
    for i in range(xs.shape[0]):
        if xs[i] * xs[i] + ys[i] * ys[i] <= 1.0:
            inside += 1
    pi_est = 4.0 * inside / xs.shape[0]

    s = 0.0
    for i in range(us.shape[0]):
        s += math.exp(-us[i] * us[i])
    integral_est = s / us.shape[0]

    finals = np.empty(n_walks, dtype=np.int64)
    for k in range(n_walks):
        pos = 0
        base = k * n_steps
        for t in range(n_steps):
            pos += steps[base + t]
        finals[k] = pos

    mean = 0.0
    for k in range(n_walks):
        mean += finals[k]
    mean /= n_walks
    var = 0.0
    for k in range(n_walks):
        d = finals[k] - mean
        var += d * d
    var /= n_walks
    return pi_est, integral_est, mean, math.sqrt(var)


@njit(cache=False, parallel=True)
def nb_kernel_parallel(xs, ys, us, steps, n_walks, n_steps):
    inside = 0
    for i in prange(xs.shape[0]):
        if xs[i] * xs[i] + ys[i] * ys[i] <= 1.0:
            inside += 1
    pi_est = 4.0 * inside / xs.shape[0]

    s = 0.0
    for i in prange(us.shape[0]):
        s += math.exp(-us[i] * us[i])
    integral_est = s / us.shape[0]

    finals = np.empty(n_walks, dtype=np.int64)
    for k in prange(n_walks):
        pos = 0
        base = k * n_steps
        for t in range(n_steps):
            pos += steps[base + t]
        finals[k] = pos

    mean = 0.0
    for k in prange(n_walks):
        mean += finals[k]
    mean /= n_walks
    var = 0.0
    for k in prange(n_walks):
        d = finals[k] - mean
        var += d * d
    var /= n_walks
    return pi_est, integral_est, mean, math.sqrt(var)


def numba_serial_kernel(xs, ys, us, steps, n_walks: int, n_steps: int) -> Output:
    return tuple(nb_kernel_serial(xs, ys, us, steps, n_walks, n_steps))


def numba_parallel_kernel(xs, ys, us, steps, n_walks: int, n_steps: int) -> Output:
    return tuple(nb_kernel_parallel(xs, ys, us, steps, n_walks, n_steps))


def numba_serial_e2e(size: int) -> Output:
    return numba_serial_kernel(*make_workload_numpy(size))


def numba_parallel_e2e(size: int) -> Output:
    return numba_parallel_kernel(*make_workload_numpy(size))


# ---------------------------------------------------------------------------
# D. ProcessPoolExecutor
# ---------------------------------------------------------------------------
_POOL: ProcessPoolExecutor | None = None


def init_pool(workers: int | None = None) -> ProcessPoolExecutor:
    global _POOL
    if workers is None:
        workers = min(8, os.cpu_count() or 1)
    _POOL = ProcessPoolExecutor(max_workers=workers)
    return _POOL


def shutdown_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=True)
        _POOL = None


def _bounds(n: int, parts: int) -> List[Tuple[int, int]]:
    return [(i * n // parts, (i + 1) * n // parts) for i in range(parts)]


def _local_stats(xs, ys, us, steps_chunk, pi_b, int_b, walk_b, n_steps):
    """Считает частичные суммы; на выходе только 6 скаляров (дешёвый IPC).

    xs/ys/us — куски исходных массивов, steps_chunk — уже локальный кусок шагов.
    """
    xa, xb = pi_b
    inside = int(np.count_nonzero(xs[xa:xb] * xs[xa:xb] + ys[xa:xb] * ys[xa:xb] <= 1.0))

    ua, ub = int_b
    integ_sum = float(np.exp(-np.square(us[ua:ub])).sum())

    wa, wb = walk_b
    n_local = wb - wa
    if n_local > 0:
        finals = steps_chunk.reshape(n_local, n_steps).sum(axis=1)
        f = finals.astype(np.float64)
        walk_sum, walk_sq = float(f.sum()), float(np.dot(f, f))
    else:
        walk_sum = walk_sq = 0.0
    return inside, integ_sum, int(ub - ua), walk_sum, walk_sq, n_local


def _proc_kernel_task(payload):
    xs, ys, us, steps, n_steps, pi_b, int_b, walk_b = payload
    wa, wb = walk_b
    chunk = steps[wa * n_steps:wb * n_steps]
    return _local_stats(xs, ys, us, chunk, pi_b, int_b, walk_b, n_steps)


def _proc_e2e_task(payload):
    pi_b, int_b, walk_b, n_steps, seed = payload
    rng = np.random.default_rng(seed)
    xa, xb = pi_b
    xs = rng.random(xb - xa)
    ys = rng.random(xb - xa)
    ua, ub = int_b
    us = rng.random(ub - ua)
    wa, wb = walk_b
    steps = rng.integers(0, 2, size=(wb - wa) * n_steps, dtype=np.int8) * 2 - 1
    # массивы здесь уже локальные, поэтому и границы срезов локальные
    local_pi = (0, xb - xa)
    local_int = (0, ub - ua)
    local_walk = (0, wb - wa)
    return _local_stats(xs, ys, us, steps, local_pi, local_int, local_walk, n_steps)


def _combine(partials, n_pi: int, n_int: int, n_walks: int) -> Output:
    inside = sum(p[0] for p in partials)
    integ_sum = sum(p[1] for p in partials)
    walk_sum = sum(p[3] for p in partials)
    walk_sq = sum(p[4] for p in partials)
    mean = walk_sum / n_walks
    var = max(0.0, walk_sq / n_walks - mean * mean)
    return 4.0 * inside / n_pi, integ_sum / n_int, mean, math.sqrt(var)


def _make_bounds(n_pi, n_int, n_walks, parts):
    return _bounds(n_pi, parts), _bounds(n_int, parts), _bounds(n_walks, parts)


def process_kernel(xs, ys, us, steps, n_walks: int, n_steps: int,
                   pool: ProcessPoolExecutor | None = None, parts: int = 8) -> Output:
    if pool is None:
        pool = _POOL
    if pool is None:
        raise RuntimeError("Сначала вызовите init_pool()")
    n_pi, n_int = xs.size, us.size
    pi_b, int_b, walk_b = _make_bounds(n_pi, n_int, n_walks, parts)
    tasks = [
        (xs, ys, us, steps, n_steps, pi_b[i], int_b[i], walk_b[i])
        for i in range(parts)
    ]
    partials = list(pool.map(_proc_kernel_task, tasks))
    return _combine(partials, n_pi, n_int, n_walks)


def process_e2e(size: int, workers: int = 8,
                pool: ProcessPoolExecutor | None = None) -> Output:
    """End-to-end: данные генерируются прямо в рабочих процессах, в main идут только bounds."""
    n_pi, n_int, n_walks, n_steps = shape_for_size(size)
    parts = workers
    pi_b, int_b, walk_b = _make_bounds(n_pi, n_int, n_walks, parts)
    tasks = [(pi_b[i], int_b[i], walk_b[i], n_steps, SEED * 1000 + i) for i in range(parts)]
    if pool is not None:
        partials = list(pool.map(_proc_e2e_task, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as own_pool:
            partials = list(own_pool.map(_proc_e2e_task, tasks))
    return _combine(partials, n_pi, n_int, n_walks)
