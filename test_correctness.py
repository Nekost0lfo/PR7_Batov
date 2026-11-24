"""Тесты корректности и устойчивости для варианта 6."""

from __future__ import annotations

import math
import statistics
import unittest

import numpy as np

import implementations as impl
from variant_6 import make_workload, run_baseline, summarize


class TestCorrectness(unittest.TestCase):
    def assert_outputs_close(self, got, ref, rtol=1e-10, atol=1e-10):
        self.assertEqual(len(got), 4)
        for g, r in zip(got, ref):
            self.assertTrue(
                math.isclose(float(g), float(r), rel_tol=rtol, abs_tol=atol),
                msg=f"{got} != {ref}",
            )

    def test_kernel_matches_baseline_tiny_and_sizes(self):
        for size in [1, 400, 1600, 6400]:
            wp = make_workload(size)
            ref = summarize(run_baseline(wp))
            wnp = impl.as_numpy_workload(wp)
            xs, ys, us, steps, nw, ns = wnp

            for name, fn in [
                ("numpy", lambda: impl.numpy_kernel(xs, ys, us, steps, nw, ns)),
                ("numpy_lowmem", lambda: impl.numpy_lowmem_kernel(xs, ys, us, steps, nw, ns)),
                ("numba_serial", lambda: impl.numba_serial_kernel(xs, ys, us, steps, nw, ns)),
                ("numba_prange", lambda: impl.numba_parallel_kernel(xs, ys, us, steps, nw, ns)),
            ]:
                got = fn()
                self.assert_outputs_close(got, ref, rtol=1e-12, atol=1e-12)

            impl.init_pool(4)
            try:
                gotp = impl.process_kernel(xs, ys, us, steps, nw, ns, pool=impl._POOL, parts=4)
            finally:
                impl.shutdown_pool()
            self.assert_outputs_close(gotp, ref, rtol=1e-12, atol=1e-12)

    def test_repeatability(self):
        # три запуска подряд дают одинаковый результат
        for fn in [
            lambda: impl.numpy_e2e(800),
            lambda: impl.numba_serial_e2e(800),
            lambda: impl.numba_parallel_e2e(800),
        ]:
            outs = [fn() for _ in range(3)]
            self.assertTrue(all(o == outs[0] for o in outs[1:]))

    def test_boundaries(self):
        # минимальный допустимый размер для этой реализации: size>=1, n_walks=max(1,size)=1
        out = impl.numpy_e2e(1)
        pi_est, integ, mean, std = out
        self.assertTrue(0.0 <= pi_est <= 4.0)
        self.assertTrue(math.isfinite(integ) and integ > 0.0)
        self.assertTrue(math.isfinite(mean) and math.isfinite(std))
        self.assertGreaterEqual(std, 0.0)

        # типы/непрерывность: через as_numpy_workload должны получаться float64/int8 contiguous
        wp = make_workload(2)
        xs, ys, us, steps, nw, ns = impl.as_numpy_workload(wp)
        self.assertEqual(xs.dtype, np.float64)
        self.assertEqual(ys.dtype, np.float64)
        self.assertEqual(us.dtype, np.float64)
        self.assertEqual(steps.dtype, np.int8)
        self.assertTrue(xs.flags["C_CONTIGUOUS"])
        self.assertTrue(steps.flags["C_CONTIGUOUS"])

    def test_prange_ordering_independence(self):
        # prange меняет порядок редукций; суммирование по walk-разделам зависит от разбиения,
        # но прогоны дают те же результаты по тем же данным (детерминированность входа).
        wnp = impl.as_numpy_workload(make_workload(6400))
        ref = impl.numba_serial_kernel(*wnp)
        for threads in [1, 4, 8]:
            import numba

            numba.set_num_threads(threads)
            got = impl.numba_parallel_kernel(*wnp)
            # порядок суммирования изменился, но данные те же и нет гонок — значения совпадают
            self.assert_outputs_close(got, ref, rtol=1e-12, atol=1e-12)

        # Примечание: если бы был 'steps[k] += ...' из разных потоков — была бы гонка.
        # В нашем алгоритме каждый поток пишет только в finals[k] для своего k (split по walks),
        # и все редукции над finals — читатели, поэтому гонок нет.


if __name__ == "__main__":
    unittest.main()
