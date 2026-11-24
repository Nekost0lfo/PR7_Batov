"""Стенд измерений для варианта 6.

Запуск:  python benchmark.py
Результаты: results/*.csv, results/*.png, results/results.json

Режимы:
  kernel   — только вычисление на уже подготовленных данных;
  e2e      — генерация данных + вычисление (как run_task в исходном файле).
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import numba
import psutil

import implementations as impl

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

TINY = 1
SIZES = [400, 800, 1600, 3200, 6400]
ALL_SIZES = [TINY] + SIZES
REPEATS = 3
WORKERS = min(8, os.cpu_count() or 1)

PROC = psutil.Process()
CHILDREN_LABEL = "A..D: Python / NumPy / Numba-serial / Numba-prange / Processes"


def rss_tree() -> int:
    """RSS основного процесса + всех детей (для ProcessPoolExecutor)."""
    total = PROC.memory_info().rss
    for child in PROC.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total


def peak_rss_delta(fn, interval_ms: int = 1) -> float:
    """Дельта пикового RSS в MiB во время вызова fn (sampling в отдельном потоке)."""
    gc = __import__("gc")
    gc.collect()
    base = rss_tree()
    peak = base
    stop = threading.Event()

    def monitor():
        nonlocal peak
        while not stop.is_set():
            cur = rss_tree()
            if cur > peak:
                peak = cur
            stop.wait(interval_ms / 1000.0)

    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    try:
        fn()
    finally:
        stop.set()
        th.join()
    return max(0.0, (peak - base) / 2**20)


def traced_peak_mib(fn) -> float:
    """Пик traced-памяти (Python + NumPy) в MiB; замер отдельно от времени."""
    gc = __import__("gc")
    gc.collect()
    import tracemalloc
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] / 2**20
    finally:
        tracemalloc.stop()


def time_call(fn, *, repeats: int = REPEATS):
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return {
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "times": times,
        "result": tuple(float(v) for v in result),
    }


def write_csv(name: str, header, rows) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def measure_cold_warm(wnp):
    """Первый вызов njit-функции = cold (компиляция), последующие = warm."""
    xs, ys, us, steps, nw, ns = wnp

    t0 = time.perf_counter()
    impl.nb_kernel_serial(xs, ys, us, steps, nw, ns)
    cold_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    impl.nb_kernel_parallel(xs, ys, us, steps, nw, ns)
    cold_parallel = time.perf_counter() - t0

    warm_serial = time_call(lambda: impl.numba_serial_kernel(*wnp), repeats=5)
    warm_parallel = time_call(lambda: impl.numba_parallel_kernel(*wnp), repeats=5)
    return {
        "cold_serial_s": cold_serial,
        "cold_parallel_s": cold_parallel,
        "cold_both_s": cold_serial + cold_parallel,
        "warm_serial_median_s": warm_serial["median"],
        "warm_parallel_median_s": warm_parallel["median"],
        "cold_over_warm_serial": cold_serial / warm_serial["median"],
        "cold_over_warm_parallel": cold_parallel / warm_parallel["median"],
    }


def run():
    print(f"Python {os.sys.version.split()[0]}, numpy {np.__version__}, "
          f"numba {numba.__version__}, потоков Numba={numba.get_num_threads()}, "
          f"ядер={WORKERS}")

    pool = impl.init_pool(WORKERS)
    results = {"env": {
        "python": os.sys.version.split()[0],
        "numpy": np.__version__,
        "numba": numba.__version__,
        "cpu_count": WORKERS,
        "numba_threads": numba.get_num_threads(),
        "repeats": REPEATS,
    }}

    # Важно: до этого ни одна njit-функция не вызывалась.
    cold = measure_cold_warm(impl.as_numpy_workload(impl.make_workload_py(400)))
    results["cold_warm"] = cold
    print(f"Cold compile: serial={cold['cold_serial_s']:.3f}s, "
          f"parallel={cold['cold_parallel_s']:.3f}s")

    kernel_rows, e2e_rows, mem_rows, speedup_rows = [], [], [], []
    checks = []

    for size in ALL_SIZES:
        wp = impl.make_workload_py(size)
        wnp = impl.as_numpy_workload(wp)
        xs, ys, us, steps, nw, ns = wnp

        kn = {
            "A Python": lambda: impl.python_kernel(wp),
            "B NumPy": lambda: impl.numpy_kernel(xs, ys, us, steps, nw, ns),
            "B NumPy lowmem": lambda: impl.numpy_lowmem_kernel(xs, ys, us, steps, nw, ns),
            "C Numba serial": lambda: impl.numba_serial_kernel(xs, ys, us, steps, nw, ns),
            "C Numba prange": lambda: impl.numba_parallel_kernel(xs, ys, us, steps, nw, ns),
            "D Processes": lambda: impl.process_kernel(xs, ys, us, steps, nw, ns,
                                                       pool=pool, parts=WORKERS),
        }
        en = {
            "A Python": lambda: impl.python_e2e(size),
            "B NumPy": lambda: impl.numpy_e2e(size),
            "C Numba serial": lambda: impl.numba_serial_e2e(size),
            "C Numba prange": lambda: impl.numba_parallel_e2e(size),
            "D Processes": lambda: impl.process_e2e(size, workers=WORKERS),
        }

        kres = {name: time_call(fn) for name, fn in kn.items()}
        eres = {name: time_call(fn) for name, fn in en.items()}
        base_k = kres["A Python"]["median"]
        base_e = eres["A Python"]["median"]

        for name in kn:
            kernel_rows.append([size, name, f"{kres[name]['median']:.6f}",
                                f"{kres[name]['min']:.6f}", f"{kres[name]['max']:.6f}"])
            speedup_rows.append([size, "kernel", name, f"{base_k / kres[name]['median']:.3f}"])
        for name in en:
            e2e_rows.append([size, name, f"{eres[name]['median']:.6f}",
                             f"{eres[name]['min']:.6f}", f"{eres[name]['max']:.6f}"])
            speedup_rows.append([size, "e2e", name, f"{base_e / eres[name]['median']:.3f}"])

        # Память: отдельные прогоны без tracemalloc в таймерах.
        mem = {}
        for name, fn in kn.items():
            mem[(name, "kernel")] = (
                traced_peak_mib(lambda: fn()),
                peak_rss_delta(lambda: fn()),
            )
        for name, fn in en.items():
            mem[(name, "e2e")] = (
                traced_peak_mib(lambda: fn()),
                peak_rss_delta(lambda: fn()),
            )
        for (name, mode), (tr, rss) in mem.items():
            mem_rows.append([size, name, mode, f"{tr:.3f}", f"{rss:.3f}"])

        ref = np.asarray(kres["A Python"]["result"], dtype=np.float64)
        for name in kn:
            got = np.asarray(kres[name]["result"], dtype=np.float64)
            checks.append({
                "size": size, "mode": "kernel", "impl": name,
                "max_abs_diff": float(np.max(np.abs(got - ref))),
                "allclose": bool(np.allclose(got, ref, rtol=1e-12, atol=1e-12)),
            })
        for name in en:
            got = np.asarray(eres[name]["result"], dtype=np.float64)
            checks.append({
                "size": size, "mode": "e2e", "impl": name,
                "max_abs_diff_e2e_different_rng": float(np.max(np.abs(got - ref))),
            })

        print(f"size={size:<5} A={base_k:.4f}s  B={kres['B NumPy']['median']:.4f}s  "
              f"C={kres['C Numba serial']['median']:.4f}s  "
              f"C||={kres['C Numba prange']['median']:.4f}s  "
              f"D={kres['D Processes']['median']:.4f}s")

    # Масштабирование Numba по потокам (ядро, N=6400).
    wnp_big = impl.as_numpy_workload(impl.make_workload_py(SIZES[-1]))
    serial_t = time_call(lambda: impl.numba_serial_kernel(*wnp_big))["median"]
    thread_rows = []
    for threads in (1, 2, 4, 8):
        numba.set_num_threads(threads)
        # прогреть после смены числа потоков
        impl.numba_parallel_kernel(*wnp_big)
        t = time_call(lambda: impl.numba_parallel_kernel(*wnp_big))["median"]
        thread_rows.append([threads, f"{t:.6f}", f"{serial_t:.6f}",
                            f"{serial_t / t:.3f}", f"{serial_t / t / threads:.3f}"])
    numba.set_num_threads(WORKERS)
    results["numba_threads"] = {
        "serial_s": serial_t,
        "rows": thread_rows,
    }

    # Масштабирование процессов (N=6400).
    proc_rows = []
    for parts in (1, 2, 4, 8):
        t_k = time_call(lambda: impl.process_kernel(*wnp_big, pool=pool, parts=parts))["median"]
        t_e = time_call(lambda: impl.process_e2e(SIZES[-1], workers=parts))["median"]
        proc_rows.append([parts, f"{t_k:.6f}", f"{t_e:.6f}", "0.000"])
    # пересчитаем speedup относительно 1 части
    t1 = float(proc_rows[0][1])
    for r in proc_rows:
        r[3] = f"{t1 / float(r[1]):.3f}"
    results["process_threads"] = proc_rows

    # Негативный кейс: TINY, сравнение с лучшим "большим" ускорением.
    tiny_kernel = {r[1]: float(r[2]) for r in kernel_rows if r[0] == TINY}
    tiny_e2e = {r[1]: float(r[2]) for r in e2e_rows if r[0] == TINY}
    big_kernel = {r[1]: float(r[2]) for r in kernel_rows if r[0] == SIZES[-1]}
    results["negative_case"] = {
        "tiny_kernel_s": tiny_kernel,
        "tiny_e2e_s": tiny_e2e,
        "big_kernel_s": big_kernel,
        "numpy_tiny_speedup": tiny_kernel["A Python"] / tiny_kernel["B NumPy"],
        "numba_serial_tiny_speedup": tiny_kernel["A Python"] / tiny_kernel["C Numba serial"],
        "process_tiny_kernel_speedup": tiny_kernel["A Python"] / tiny_kernel["D Processes"],
        "process_tiny_e2e_speedup": tiny_e2e["A Python"] / tiny_e2e["D Processes"],
    }
    results["correctness_spot_check"] = checks

    write_csv("kernel_times.csv",
              ["size", "implementation", "median_s", "min_s", "max_s"], kernel_rows)
    write_csv("e2e_times.csv",
              ["size", "implementation", "median_s", "min_s", "max_s"], e2e_rows)
    write_csv("speedup.csv", ["size", "mode", "implementation", "speedup_vs_python"], speedup_rows)
    write_csv("memory.csv",
              ["size", "implementation", "mode", "traced_peak_mib", "rss_delta_peak_mib"], mem_rows)
    write_csv("cold_warm.csv",
              ["metric", "value_s"],
              [[k, f"{v:.6f}"] for k, v in cold.items()])
    write_csv("numba_threads.csv",
              ["threads", "parallel_kernel_s", "serial_kernel_s", "speedup_vs_serial", "efficiency"],
              thread_rows)
    write_csv("process_threads.csv",
              ["parts", "kernel_s", "e2e_s", "speedup_kernel_vs_1"], proc_rows)

    with (OUT / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    impl.shutdown_pool()
    print("Готово: results/*.csv, results/*.json")

    make_plots(results, kernel_rows, e2e_rows, speedup_rows, mem_rows, cold, thread_rows)


def make_plots(results, kernel_rows, e2e_rows, speedup_rows, mem_rows, cold, thread_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "A Python": "#222222",
        "B NumPy": "#1f77b4",
        "B NumPy lowmem": "#17becf",
        "C Numba serial": "#d62728",
        "C Numba prange": "#ff7f0e",
        "D Processes": "#2ca02c",
    }
    names = ["A Python", "B NumPy", "C Numba serial", "C Numba prange", "D Processes"]

    def grouped(rows):
        d = {}
        for size, impl_name, med, *_ in rows:
            d.setdefault(impl_name, []).append((int(size), float(med)))
        return {k: sorted(v) for k, v in d.items()}

    kd, ed = grouped(kernel_rows), grouped(e2e_rows)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in kd:
        pts = kd[name]
        ax.plot(*zip(*pts), marker="o", label=name, color=colors[name])
    ax.set(xscale="log", yscale="log", xlabel="Workload size N/1700 (log)",
           ylabel="Kernel time, s (log)", title="Kernel time vs N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "time_kernel.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in names:
        pts = ed[name]
        ax.plot(*zip(*pts), marker="s", label=name, color=colors[name])
    ax.set(xscale="log", yscale="log", xlabel="Workload size N/1700 (log)",
           ylabel="End-to-end time, s (log)", title="End-to-end time vs N")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "time_e2e.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for mode, ax in zip(("kernel", "e2e"), axes):
        for name in names:
            pts = sorted((int(s), float(sp)) for s, m, n, sp in speedup_rows
                         if m == mode and n == name)
            ax.plot(*zip(*pts), marker="o", label=name, color=colors[name])
        ax.set(xscale="log", yscale="log", title=f"Speedup vs Python — {mode}",
               xlabel="Workload size N/1700 (log)")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "speedup.png", dpi=150); plt.close(fig)

    md = {}
    for size, name, mode, tr, rss in mem_rows:
        md.setdefault((name, mode), []).append((int(size), float(tr)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for mode, ax in zip(("kernel", "e2e"), axes):
        for name in names:
            pts = sorted(md.get((name, mode), []))
            if pts:
                ax.plot(*zip(*pts), marker="o", label=name, color=colors[name])
        ax.set(xscale="log", yscale="log", title=f"Peak traced memory — {mode}",
               xlabel="Workload size N/1700 (log)", ylabel="MiB (log)")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "memory.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = ["Numba serial", "Numba prange"]
    x = np.arange(2)
    cold_v = [cold["cold_serial_s"], cold["cold_parallel_s"]]
    warm_v = [cold["warm_serial_median_s"], cold["warm_parallel_median_s"]]
    ax.bar(x - 0.18, cold_v, 0.36, label="cold (compile + run)", color="#d62728")
    ax.bar(x + 0.18, warm_v, 0.36, label="warm (median)", color="#2ca02c")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("s (log)"); ax.set_title("Numba: cold vs warm")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "cold_warm.png", dpi=150); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    t = [int(r[0]) for r in thread_rows]
    tm = [float(r[1]) for r in thread_rows]
    eff = [float(r[4]) for r in thread_rows]
    ax1.plot(t, tm, marker="o", color="#ff7f0e", label="prange time (s)")
    ax1.set_xlabel("Numba threads"); ax1.set_ylabel("kernel time, s", color="#ff7f0e")
    ax2 = ax1.twinx()
    ax2.plot(t, eff, marker="s", color="#1f77b4", label="efficiency")
    ax2.set_ylabel("efficiency = speedup(1T)/T", color="#1f77b4")
    ax1.set_title(f"prange scaling at N/1700=6400 (serial={results['numba_threads']['serial_s']:.4f}s)")
    fig.tight_layout(); fig.savefig(OUT / "numba_threads.png", dpi=150); plt.close(fig)

    # Проверка линейности: время, отнесённое к N, почти не растёт.
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in names:
        pts = [(n, t / n) for n, t in kd[name] if n > 1]
        ax.plot(*zip(*pts), marker="o", label=name, color=colors[name])
    ax.set(xscale="log", yscale="log", xlabel="Workload size N/1700 (log)",
           ylabel="kernel time / N, s (log)", title="Linearity check: t(N)/N")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "linearity.png", dpi=150); plt.close(fig)

    # Аппроксимация показателя степени p в t ~ N^p по двум крайним рабочим точкам.
    fits = {}
    for name, pts in kd.items():
        pts = [(n, t) for n, t in pts if n >= SIZES[0]]
        n1, t1 = pts[0]; n2, t2 = pts[-1]
        fits[name] = math.log(t2 / t1) / math.log(n2 / n1)
    results["loglog_slope"] = fits
    with (OUT / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Графики сохранены в results/*.png; показатели степени p:",
          {k: round(v, 2) for k, v in fits.items()})


if __name__ == "__main__":
    run()
