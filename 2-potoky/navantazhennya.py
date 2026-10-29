"""
РСПО, практична 3, крок 2. Скільки ядер реально працювало.

process_time() рахує процесорний час усіх потоків програми разом.
Поділивши його на звичайний (настінний) час, отримуємо середню кількість
ядер, які були зайняті:

  1.0  — працювало одне ядро, скільки б потоків не було;
  N    — N ядер працювало постійно;
  ≈ 0  — потоки майже весь час чекали і процесор не вантажили.

Міряємо обчислення (Мандельброт) і очікування (sleep) однією й тією самою
схемою пулу потоків.

На Windows process_time() бере дані з GetProcessTimes(), а там процесорний час
нараховується за тиками таймера (≈ 15.6 мс) і занижується: на послідовному
прогоні, де працює рівно одне ядро, виходить 0.8–0.9 замість 1.0. Тому поруч
додано другий вимір — лічильник тактів процесу (QueryProcessCycleTime),
який тиками не округлюється. Такти переводяться в «ядра» діленням на такти за
секунду одного повністю зайнятого ядра; цю величину беремо з послідовного
прогону того ж сеансу.

ЗАПУСК:  python navantazhennya.py        (≈ 2 хв)
"""

import ctypes
import functools
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

KOREN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KOREN, "1-poslidovno"))
import mandelbrot as m  # noqa: E402

PROGONIV = 3


def poslidovno():
    return m.obchyslyty(m.SHYRYNA, m.VYSOTA, m.MAX_ITER)


def potokamy(n):
    robota = functools.partial(m.obchyslyty_riadok, shyryna=m.SHYRYNA,
                               vysota=m.VYSOTA, max_iter=m.MAX_ITER)
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(robota, range(m.VYSOTA)))


def zapyt(nomer):
    time.sleep(0.5)
    return nomer


def ochikuvannya_potokamy(n):
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(zapyt, range(32)))


def taktyprocesu():
    """Такти процесора, витрачені всіма потоками процесу (тільки Windows)."""
    if sys.platform != "win32":
        return 0
    yadro = ctypes.windll.kernel32
    yadro.GetCurrentProcess.restype = ctypes.c_void_p
    taktu = ctypes.c_ulonglong()
    yadro.QueryProcessCycleTime(ctypes.c_void_p(yadro.GetCurrentProcess()),
                                ctypes.byref(taktu))
    return taktu.value


def navantazhennya(fn, *a):
    """Повертає (настінний час, процесорний час, такти)."""
    c0, w0, k0 = time.process_time(), time.perf_counter(), taktyprocesu()
    fn(*a)
    return (time.perf_counter() - w0, time.process_time() - c0,
            taktyprocesu() - k0)


def seriya(nazva, fn, *a, takty_na_yadro=None):
    """Три прогони; друкує відношення за process_time і (якщо відомо) за тактами."""
    vidnoshennya = []
    for i in range(1, PROGONIV + 1):
        t, cpu, takty = navantazhennya(fn, *a)
        riadok = ("  %-26s прогін %d:  настінний %7.3f c   процесорний %7.3f c   "
                  "process_time/perf_counter %.2f" % (nazva, i, t, cpu, cpu / t))
        if takty_na_yadro:
            riadok += "   ядер за тактами %.2f" % (takty / t / takty_na_yadro)
        print(riadok)
        vidnoshennya.append(takty / t)
    return statistics.median(vidnoshennya)


if __name__ == "__main__":
    print("=" * 90)
    print("УМОВИ ЗАПУСКУ")
    print("=" * 90)
    print("  логічних ядер :", os.cpu_count())
    print("  обчислення    : Мандельброт %d×%d, max_iter=%d, у замір входить пул, без запису PGM"
          % (m.SHYRYNA, m.VYSOTA, m.MAX_ITER))
    print("  очікування    : 32 × time.sleep(0.5)")
    print("  прогонів      :", PROGONIV)
    print()
    print("=" * 90)
    print("ВІДНОШЕННЯ process_time / perf_counter")
    print("=" * 90)
    # послідовний прогін: працює рівно одне ядро, звідси такти за секунду одного ядра
    na_yadro = seriya("обчислення, послідовно", poslidovno)
    print("  такти за секунду одного зайнятого ядра: %.2f ГГц (медіана послідовних прогонів)"
          % (na_yadro / 1e9))
    for n in (2, 8, os.cpu_count()):
        seriya("обчислення, потоків %d" % n, potokamy, n, takty_na_yadro=na_yadro)
    for n in (8, os.cpu_count()):
        seriya("очікування, потоків %d" % n, ochikuvannya_potokamy, n,
               takty_na_yadro=na_yadro)
