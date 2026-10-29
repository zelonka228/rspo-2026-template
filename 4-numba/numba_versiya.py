"""
РСПО, практична 6. Множина Мандельброта на Numba: @njit і @njit(parallel=True).

Цикл той самий, що в 1-poslidovno/mandelbrot.py (комплексне z, перевірка abs(z) > 2,
ті самі формули координат), змінюється лише обгортка: функція одразу повертає
масив, а не список рядків. Тому результат має збігатися з еталоном точно:

  assert np.array_equal(r, base)

ЩО МІРЯЄТЬСЯ (time.perf_counter):

  компіляція — перший виклик на крихітному зображенні 2×2 з тими самими типами
               аргументів. Numba компілює функцію один раз для набору типів, а на
               2×2 саме виконання займає мікросекунди, тож час ≈ час компіляції;
  виконання  — три прогони на повному розмірі вже скомпільованою функцією.

Еталон (чистий Python) рахується тут же, в тому самому сеансі, один раз: розкид між
сеансами більший, ніж усередині сеансу (див. zamiry.md, практична 02).

ЗАПУСК:  python numba_versiya.py          (≈ 20 с: еталон ≈ 7–11 с, решта швидко)
         python numba_versiya.py --shyryna 150 --vysota 100   (швидка перевірка)
"""

import argparse
import json
import os
import platform
import sys
import time

import numba
import numpy as np
from numba import njit, prange

KOREN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KOREN, "1-poslidovno"))
import mandelbrot as m  # noqa: E402

PROGONIV = 3
TUT = os.path.dirname(os.path.abspath(__file__))
ZAMIRY = os.path.join(TUT, "zamiry-numba.json")


@njit
def mandelbrot_numba(w, h, max_iter, x_min, x_max, y_min, y_max):
    out = np.empty((h, w), dtype=np.int32)
    krok = (x_max - x_min) / w
    for y in range(h):
        im = y_min + (y_max - y_min) * y / h
        for x in range(w):
            c = complex(x_min + krok * x, im)
            z = 0j
            n = max_iter
            for k in range(max_iter):
                z = z * z + c
                if abs(z) > 2.0:
                    n = k
                    break
            out[y, x] = n
    return out


@njit(parallel=True)
def mandelbrot_prange(w, h, max_iter, x_min, x_max, y_min, y_max):
    out = np.empty((h, w), dtype=np.int32)
    krok = (x_max - x_min) / w
    for y in prange(h):                      # було range
        im = y_min + (y_max - y_min) * y / h
        for x in range(w):
            c = complex(x_min + krok * x, im)
            z = 0j
            n = max_iter
            for k in range(max_iter):
                z = z * z + c
                if abs(z) > 2.0:
                    n = k
                    break
            out[y, x] = n
    return out


def zamir(fn, prohoniv, *args):
    """Повертає (мінімум, список часів, результат останнього прогону)."""
    chasy = []
    for _ in range(prohoniv):
        t0 = time.perf_counter()
        r = fn(*args)
        chasy.append(time.perf_counter() - t0)
    return min(chasy), chasy, r


def kompiliatsiya(fn, max_iter):
    """Час першого виклику на 2×2 з тими самими типами аргументів."""
    t0 = time.perf_counter()
    fn(2, 2, max_iter, m.X_MIN, m.X_MAX, m.Y_MIN, m.Y_MAX)
    return time.perf_counter() - t0


def pasport(shyryna, vysota, max_iter):
    print("=" * 78)
    print("УМОВИ ЗАПУСКУ")
    print("=" * 78)
    print("  Python        :", sys.version.split()[0], platform.python_implementation())
    print("  numpy, numba  :", np.__version__, numba.__version__)
    print("  процесор      :", platform.processor())
    print("  логічних ядер :", os.cpu_count(), "| потоків Numba:", numba.get_num_threads())
    print("  ОС            :", platform.platform())
    print("  розмір        : %d×%d, max_iter=%d, ділянка [%g, %g]×[%g, %g]"
          % (shyryna, vysota, max_iter, m.X_MIN, m.X_MAX, m.Y_MIN, m.Y_MAX))
    print("  прогонів      :", PROGONIV, "(еталон — 1 прогін, він довгий)")
    print("  у замір входить: лише обчислення, без запису файлу")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shyryna", type=int, default=m.SHYRYNA)
    ap.add_argument("--vysota", type=int, default=m.VYSOTA)
    ap.add_argument("--max-iter", type=int, default=m.MAX_ITER)
    a = ap.parse_args()
    W, H, MAX_ITER = a.shyryna, a.vysota, a.max_iter
    ARGS = (W, H, MAX_ITER, m.X_MIN, m.X_MAX, m.Y_MIN, m.Y_MAX)

    pasport(W, H, MAX_ITER)

    t_base, chasy_base, base = zamir(m.obchyslyty, 1, W, H, MAX_ITER)
    base = np.array(base)
    print("чистий Python (еталон): %.3f c" % t_base)
    print()

    zapys = {"posl": t_base, "shyryna": W, "vysota": H, "max_iter": MAX_ITER,
             "cpu_count": os.cpu_count(), "numba_potokiv": numba.get_num_threads()}

    for nazva, fn in (("@njit", mandelbrot_numba), ("@njit(parallel=True) + prange", mandelbrot_prange)):
        t_komp = kompiliatsiya(fn, MAX_ITER)
        t, chasy, r = zamir(fn, PROGONIV, *ARGS)
        assert np.array_equal(r, base), "результат розійшовся з еталоном"
        print("%s" % nazva)
        print("  компіляція (перший виклик на 2×2) : %.3f c" % t_komp)
        print("  виконання, 3 прогони              :", " / ".join("%.4f c" % c for c in chasy),
              "| мінімум %.4f c" % t)
        print("  прискорення відносно еталона      : ×%.0f" % (t_base / t))
        print("  збіг з еталоном                   : точний, 0 розбіжних точок із %d" % base.size)
        print()
        zapys[nazva] = {"kompiliatsiya": t_komp, "chasy": chasy, "min": t}

    t1 = zapys["@njit"]["min"]
    t2 = zapys["@njit(parallel=True) + prange"]["min"]
    print("prange відносно @njit: ×%.2f (потоків Numba %d)" % (t1 / t2, numba.get_num_threads()))
    zapys["prange_vidnosno_njit"] = t1 / t2

    povnyi = (W, H, MAX_ITER) == (m.SHYRYNA, m.VYSOTA, m.MAX_ITER)
    if povnyi:
        with open(ZAMIRY, "w", encoding="utf-8") as f:
            json.dump(zapys, f, ensure_ascii=False, indent=2)
        print("Збережено:", ZAMIRY)
    else:
        print("Розмір не той, що в заміру, — %s не чіпаю." % os.path.basename(ZAMIRY))
