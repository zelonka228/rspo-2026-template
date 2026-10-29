"""
РСПО, практична 6. Мандельброт, векторизована версія на NumPy.

Усі пікселі одночасно: масиви zx, zy, cx, cy розміру H×W, цикл лише по ітераціях.
Точки, що вже вилетіли за |z| > 2, маскою `aktyvni` виключаються з оновлення.

Дві речі, які тут треба показати числами, а не проігнорувати:

  1. Сітка. Якщо взяти np.linspace(a, b, n) за замовчуванням, останній піксель
     потрапляє точно в b, а в еталоні крок (b - a) / n і b не досягається.
     Сітки різні, і розбіжних точок тисячі. З endpoint=False сітка збігається.
  2. Навіть із тією самою сіткою формула інша: |z|² > 4 замість abs(z) > 2 (еталон
     рахує hypot) і інший порядок дій. На межі фрактала орбіта хаотична, тож
     різниця в останніх бітах іноді виростає до іншої кількості ітерацій.
     Скільки таких точок — рахується тут.

ЗАПУСК:  python numpy_versiya.py
"""

import argparse
import os
import platform
import sys
import time

import numpy as np

KOREN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KOREN, "1-poslidovno"))
import mandelbrot as m  # noqa: E402

PROGONIV = 3


def mandelbrot_numpy(w, h, max_iter, endpoint=False):
    cx, cy = np.meshgrid(np.linspace(m.X_MIN, m.X_MAX, w, endpoint=endpoint),
                         np.linspace(m.Y_MIN, m.Y_MAX, h, endpoint=endpoint))
    zx = np.zeros((h, w))
    zy = np.zeros((h, w))
    out = np.full((h, w), max_iter, dtype=np.int32)
    aktyvni = np.ones((h, w), dtype=bool)          # точки, що ще не вилетіли
    for n in range(max_iter):
        zx_n = zx * zx - zy * zy + cx
        zy_n = 2.0 * zx * zy + cy
        zx = np.where(aktyvni, zx_n, zx)
        zy = np.where(aktyvni, zy_n, zy)
        vyletily = aktyvni & (zx * zx + zy * zy > 4.0)
        out[vyletily] = n                           # як в еталоні: номер кроку, на якому вилетіла
        aktyvni &= ~vyletily
        if not aktyvni.any():
            break
    return out


def zamir(fn, prohoniv, *args, **kw):
    chasy = []
    for _ in range(prohoniv):
        t0 = time.perf_counter()
        r = fn(*args, **kw)
        chasy.append(time.perf_counter() - t0)
    return min(chasy), chasy, r


def rozbizhnosti(r, base):
    return int((r != base).sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shyryna", type=int, default=m.SHYRYNA)
    ap.add_argument("--vysota", type=int, default=m.VYSOTA)
    ap.add_argument("--max-iter", type=int, default=m.MAX_ITER)
    a = ap.parse_args()
    W, H, MAX_ITER = a.shyryna, a.vysota, a.max_iter

    print("=" * 78)
    print("УМОВИ ЗАПУСКУ")
    print("=" * 78)
    print("  Python, numpy :", sys.version.split()[0], np.__version__)
    print("  процесор      :", platform.processor())
    print("  логічних ядер :", os.cpu_count())
    print("  розмір        : %d×%d, max_iter=%d" % (W, H, MAX_ITER))
    print("  прогонів      :", PROGONIV)
    print()

    t0 = time.perf_counter()
    base = np.array(m.obchyslyty(W, H, MAX_ITER))
    t_base = time.perf_counter() - t0
    print("чистий Python (еталон): %.3f c" % t_base)

    # 1. сітка за замовчуванням: endpoint=True
    r_bad = mandelbrot_numpy(W, H, MAX_ITER, endpoint=True)
    d_bad = rozbizhnosti(r_bad, base)
    print()
    print("linspace БЕЗ endpoint=False:  розбіжних точок %d із %d (%.2f %%)"
          % (d_bad, base.size, 100 * d_bad / base.size))

    # 2. та сама сітка, що в еталоні
    t, chasy, r = zamir(mandelbrot_numpy, PROGONIV, W, H, MAX_ITER, endpoint=False)
    d = rozbizhnosti(r, base)
    print("linspace З endpoint=False:    розбіжних точок %d із %d (%.4f %%)"
          % (d, base.size, 100 * d / base.size))
    print()
    print("NumPy, 3 прогони:", " / ".join("%.3f c" % c for c in chasy), "| мінімум %.3f c" % t)
    print("прискорення відносно еталона: ×%.2f" % (t_base / t))

    if d:
        ys, xs = np.nonzero(r != base)
        print()
        print("розбіжні точки (рядок, стовпець, еталон, NumPy), перші 12:")
        for y, x in list(zip(ys, xs))[:12]:
            print("  (%3d, %3d)  еталон %3d   NumPy %3d" % (y, x, base[y, x], r[y, x]))
        print("значення еталона в розбіжних точках: від %d до %d (max_iter=%d)"
              % (base[ys, xs].min(), base[ys, xs].max(), MAX_ITER))
        print("з них в еталоні вилетіли (не в множині): %d, у множині: %d"
              % ((base[ys, xs] < MAX_ITER).sum(), (base[ys, xs] == MAX_ITER).sum()))
