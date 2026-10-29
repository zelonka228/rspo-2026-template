"""
РСПО, практична 6. Усі реалізації в одному сеансі.

Розкид між сеансами на цій машині більший, ніж усередині сеансу (zamiry.md,
практична 02), тому для спільної таблиці всі шість реалізацій міряються
одним запуском, одна за одною, з однаковими умовами:

  * лише обчислення, без запису файлу (запис ≈ 0.02 c і не залежить від реалізації);
  * у замір процесів і потоків входить створення пулу;
  * 3 прогони, у таблиці мінімум;
  * збіг із еталоном перевіряється в кожному прогоні.

ЗАПУСК:  python porivnyannya.py        (≈ 1.5 хв)
"""

import functools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

KOREN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KOREN, "1-poslidovno"))
import mandelbrot as m  # noqa: E402

# numba і numpy імпортуються лише в головному процесі (див. блок запуску): новий процес
# пулу заново імпортує цей файл, і важкі імпорти роздули б вартість запуску пулу.

PROGONIV = 3
W, H, MAX_ITER = m.SHYRYNA, m.VYSOTA, m.MAX_ITER
ARGS = (W, H, MAX_ITER, m.X_MIN, m.X_MAX, m.Y_MIN, m.Y_MAX)


def potokamy(n):
    robota = functools.partial(m.obchyslyty_riadok, shyryna=W, vysota=H, max_iter=MAX_ITER)
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(robota, range(H)))


def protsesamy(n):
    robota = functools.partial(m.obchyslyty_riadok, shyryna=W, vysota=H, max_iter=MAX_ITER)
    with ProcessPoolExecutor(max_workers=n) as ex:
        return list(ex.map(robota, range(H), chunksize=1))


def vidpovidaye(r, base):
    return np.array_equal(np.asarray(r), base)


def zamir(nazva, fn, base, *a):
    chasy = []
    for _ in range(PROGONIV):
        t0 = time.perf_counter()
        r = fn(*a)
        chasy.append(time.perf_counter() - t0)
        assert vidpovidaye(r, base), nazva + ": результат розійшовся з еталоном"
    print("  %-26s %s | мінімум %.4f c" % (nazva, " / ".join("%.4f" % c for c in chasy), min(chasy)))
    return chasy


if __name__ == "__main__":
    import numpy as np
    import numba_versiya as nv
    from numba_versiya import mandelbrot_numba, mandelbrot_prange
    from numpy_versiya import mandelbrot_numpy

    n = os.cpu_count()
    print("Усі реалізації в одному сеансі: %d×%d, max_iter=%d, %d прогони, лише обчислення"
          % (W, H, MAX_ITER, PROGONIV))
    print()

    # компіляцію виношу з заміру (вона міряється окремо в numba_versiya.py)
    nv.kompiliatsiya(mandelbrot_numba, MAX_ITER)
    nv.kompiliatsiya(mandelbrot_prange, MAX_ITER)

    t0 = time.perf_counter()
    base_list = m.obchyslyty(W, H, MAX_ITER)
    base = np.array(base_list)
    t_pershyi = time.perf_counter() - t0

    rezultaty = {}
    rezultaty["послідовно"] = zamir("послідовно", m.obchyslyty, base, W, H, MAX_ITER)
    rezultaty["потоків %d" % n] = zamir("потоків %d" % n, potokamy, base, n)
    rezultaty["процесів %d" % n] = zamir("процесів %d" % n, protsesamy, base, n)
    rezultaty["NumPy"] = zamir("NumPy", mandelbrot_numpy, base, W, H, MAX_ITER)
    rezultaty["Numba @njit"] = zamir("Numba @njit", mandelbrot_numba, base, *ARGS)
    rezultaty["Numba prange"] = zamir("Numba prange", mandelbrot_prange, base, *ARGS)

    posl = min(rezultaty["послідовно"])
    print()
    print("| Реалізація | Прогін 1 | Прогін 2 | Прогін 3 | Мінімум | Прискорення |")
    print("|---|---|---|---|---|---|")
    for nazva, chasy in rezultaty.items():
        print("| %s | %s | %.4f c | ×%.2f |" % (
            nazva, " | ".join("%.4f c" % c for c in chasy), min(chasy), posl / min(chasy)))

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "zamiry-porivnyannya.json"),
              "w", encoding="utf-8") as f:
        json.dump(rezultaty, f, ensure_ascii=False, indent=2)
