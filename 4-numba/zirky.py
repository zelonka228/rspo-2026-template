"""
РСПО, практична 6, завдання із зірочкою.

  python zirky.py kompiliatsiya  ⭐1. Коли компіляція окупається (100×60 і повний розмір)
  python zirky.py fastmath       ⭐2. @njit(fastmath=True): час і розбіжні точки
  python zirky.py matrytsi       ⭐3. Множення матриць 400×400: NumPy проти @njit
  python zirky.py porih          ⭐4. З якого розміру prange виграє у @njit
"""

import os
import sys
import time

import numba
import numpy as np
from numba import njit

import numba_versiya as nv
from numba_versiya import m, mandelbrot_numba, mandelbrot_prange


def min_chasu(fn, povtoriv, *args):
    """Найкращий із `povtoriv` викликів, c."""
    kr = float("inf")
    for _ in range(povtoriv):
        t0 = time.perf_counter()
        fn(*args)
        kr = min(kr, time.perf_counter() - t0)
    return kr


def kompiliatsiya():
    MAX_ITER = m.MAX_ITER
    t_komp = nv.kompiliatsiya(mandelbrot_numba, MAX_ITER)
    print("компіляція @njit (перший виклик на 2×2): %.3f c" % t_komp)
    print()
    print("| Розмір | Чистий Python, c | @njit, c | Виграш за виклик, c | Викликів, щоб окупити компіляцію |")
    print("|---|---|---|---|---|")
    for w, h in ((100, 60), (m.SHYRYNA, m.VYSOTA)):
        args = (w, h, MAX_ITER, m.X_MIN, m.X_MAX, m.Y_MIN, m.Y_MAX)
        t_py = min_chasu(m.obchyslyty, 1 if w > 100 else 3, w, h, MAX_ITER)
        t_nj = min_chasu(mandelbrot_numba, 20, *args)
        vygrash = t_py - t_nj
        print("| %d×%d | %.4f | %.5f | %.4f | %.1f |" % (w, h, t_py, t_nj, vygrash, t_komp / vygrash))


def fastmath():
    MAX_ITER = m.MAX_ITER
    args = (m.SHYRYNA, m.VYSOTA, MAX_ITER, m.X_MIN, m.X_MAX, m.Y_MIN, m.Y_MAX)
    mandelbrot_fm = njit(fastmath=True)(mandelbrot_numba.py_func)
    mandelbrot_fm(2, 2, MAX_ITER, m.X_MIN, m.X_MAX, m.Y_MIN, m.Y_MAX)
    mandelbrot_numba(2, 2, MAX_ITER, m.X_MIN, m.X_MAX, m.Y_MIN, m.Y_MAX)
    r0 = mandelbrot_numba(*args)
    r1 = mandelbrot_fm(*args)
    t0 = min_chasu(mandelbrot_numba, 5, *args)
    t1 = min_chasu(mandelbrot_fm, 5, *args)
    print("@njit:                 %.4f c" % t0)
    print("@njit(fastmath=True):  %.4f c  (×%.2f)" % (t1, t0 / t1))
    print("розбіжних точок відносно @njit: %d із %d" % ((r0 != r1).sum(), r0.size))


@njit
def matmul_tsykl(a, b):
    n, k = a.shape
    k2, p = b.shape
    c = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            s = 0.0
            for q in range(k):
                s += a[i, q] * b[q, j]
            c[i, j] = s
    return c


def matrytsi():
    rng = np.random.default_rng(1)
    a = rng.random((400, 400))
    b = rng.random((400, 400))
    matmul_tsykl(a[:2, :2], b[:2, :2])            # компіляція окремо
    t_np = min_chasu(lambda: a @ b, 20)
    t_nb = min_chasu(matmul_tsykl, 5, a, b)
    ok = np.allclose(a @ b, matmul_tsykl(a, b))
    print("NumPy a @ b:                    %.5f c" % t_np)
    print("@njit, потрійний цикл:          %.5f c" % t_nb)
    print("NumPy швидший у %.0f разів; результати збігаються (allclose): %s" % (t_nb / t_np, ok))


def porih():
    MAX_ITER = m.MAX_ITER
    # компіляція обох функцій окремо
    for fn in (mandelbrot_numba, mandelbrot_prange):
        nv.kompiliatsiya(fn, MAX_ITER)
    print("потоків Numba:", numba.get_num_threads())
    print()
    print("| Розмір | Повторів | @njit | prange | prange / @njit |")
    print("|---|---|---|---|---|")
    for w, h, povt in ((2, 2, 2000), (4, 3, 2000), (10, 6, 1000), (30, 20, 300),
                       (300, 200, 30), (1200, 800, 5)):
        args = (w, h, MAX_ITER, m.X_MIN, m.X_MAX, m.Y_MIN, m.Y_MAX)
        t1 = min_chasu(mandelbrot_numba, povt, *args)
        t2 = min_chasu(mandelbrot_prange, povt, *args)
        print("| %d×%d | %d | %.2e c | %.2e c | ×%.2f |" % (w, h, povt, t1, t2, t1 / t2))


if __name__ == "__main__":
    rezhym = sys.argv[1] if len(sys.argv) > 1 else ""
    funktsii = {"kompiliatsiya": kompiliatsiya, "fastmath": fastmath,
                "matrytsi": matrytsi, "porih": porih}
    if rezhym in funktsii:
        funktsii[rezhym]()
    else:
        print(__doc__)
