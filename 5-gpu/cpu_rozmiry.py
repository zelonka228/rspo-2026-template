"""
РСПО, практична 7. Процесорні часи на тих самих розмірах, що й GPU (300×200, 1200×800, 3000×2000).

Потрібні для порівняння з Colab: Numba @njit і @njit(parallel=True) на моєму ПК
(Ryzen 7 2700, 8 фізичних / 16 логічних ядер), мінімум із 3 прогонів, без компіляції.

ЗАПУСК:  python cpu_rozmiry.py        (≈ 1 хв)
"""

import os
import sys
import time

KOREN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KOREN, "4-numba"))
import numba_versiya as nv  # noqa: E402
from numba_versiya import m, mandelbrot_numba, mandelbrot_prange  # noqa: E402

PROGONIV = 3
MAX_ITER = m.MAX_ITER

nv.kompiliatsiya(mandelbrot_numba, MAX_ITER)
nv.kompiliatsiya(mandelbrot_prange, MAX_ITER)

print("Ryzen 7 2700, max_iter=%d, мінімум із %d прогонів, c" % (MAX_ITER, PROGONIV))
print()
print("| Розмір | @njit | @njit(parallel=True) |")
print("|---|---|---|")
for w, h in ((300, 200), (1200, 800), (3000, 2000)):
    args = (w, h, MAX_ITER, m.X_MIN, m.X_MAX, m.Y_MIN, m.Y_MAX)
    rez = []
    for fn in (mandelbrot_numba, mandelbrot_prange):
        kr = float("inf")
        for _ in range(PROGONIV):
            t0 = time.perf_counter()
            fn(*args)
            kr = min(kr, time.perf_counter() - t0)
        rez.append(kr)
    print("| %d×%d | %.4f | %.4f |" % (w, h, *rez))
