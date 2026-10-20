"""
РСПО, практична 4. Множина Мандельброта через ProcessPoolExecutor.

Кожен рядок — окреме завдання. Заміри для 1, 2, 4, 8, cpu_count() процесів,
по 3 прогони, еталон (послідовна версія) рахується в тому самому сеансі.

Кожен прогін розкладено на частини:
  запуск  — створення пулу, старт усіх процесів і shutdown;
  робота  — ex.map по всіх рядках (обчислення + пересилання результатів);
  запис   — PGM-файл (послідовна частина).

ЗАПУСК:  python protsesy.py              (≈ 2 хв)
         python protsesy.py --chunksize  (⭐ chunksize = 1, 8, 50, 200)
"""

import argparse
import functools
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import psutil

KOREN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KOREN, "1-poslidovno"))
import mandelbrot as m  # noqa: E402

PROGONIV = 3
TUT = os.path.dirname(os.path.abspath(__file__))
FAIL = os.path.join(TUT, "mandelbrot.pgm")
ZAMIRY = os.path.join(TUT, "zamiry.json")


def kilkosti():
    spysok = [1, 2, 4, 8]
    if os.cpu_count() not in spysok:
        spysok.append(os.cpu_count())
    return spysok


def poslidovno():
    t0 = time.perf_counter()
    rezultat = m.obchyslyty(m.SHYRYNA, m.VYSOTA, m.MAX_ITER)
    t1 = time.perf_counter()
    m.zapysaty_pgm(rezultat, m.MAX_ITER, FAIL)
    t2 = time.perf_counter()
    return rezultat, {"zapusk": 0.0, "robota": t1 - t0, "zapys": t2 - t1, "razom": t2 - t0}


def procesamy(n, chunksize=1):
    robota = functools.partial(m.obchyslyty_riadok, shyryna=m.SHYRYNA,
                               vysota=m.VYSOTA, max_iter=m.MAX_ITER)
    t0 = time.perf_counter()
    ex = ProcessPoolExecutor(n)
    # процеси стартують лише при першому submit, тож запускаємо їх явно
    list(ex.map(abs, range(n)))
    t1 = time.perf_counter()
    rezultat = list(ex.map(robota, range(m.VYSOTA), chunksize=chunksize))
    t2 = time.perf_counter()
    ex.shutdown(wait=True)
    t3 = time.perf_counter()
    m.zapysaty_pgm(rezultat, m.MAX_ITER, FAIL)
    t4 = time.perf_counter()
    return rezultat, {"zapusk": (t1 - t0) + (t3 - t2), "robota": t2 - t1,
                      "zapys": t4 - t3, "razom": t4 - t0}


def vartist_pulu(n):
    """Два способи: як у завданні (pass) і з реальним стартом процесів."""
    t0 = time.perf_counter()
    with ProcessPoolExecutor(n) as ex:
        pass
    t1 = time.perf_counter()
    with ProcessPoolExecutor(n) as ex:
        list(ex.map(abs, range(n)))
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def osnovnyi():
    print("процесор:       ", platform.processor())
    print("ядер фізичних:  ", psutil.cpu_count(logical=False))
    print("ядер логічних:  ", os.cpu_count())
    print("Python:         ", sys.version.split()[0])
    print("задача:          Мандельброт %d×%d, max_iter=%d, %d прогони"
          % (m.SHYRYNA, m.VYSOTA, m.MAX_ITER, PROGONIV))
    print()

    base = None
    tochky = []
    for n in [0] + kilkosti():          # 0 — послідовна версія без пулу
        zamiry = []
        for _ in range(PROGONIV):
            rezultat, z = poslidovno() if n == 0 else procesamy(n)
            if base is None:
                base = rezultat
            assert rezultat == base, "результат розійшовся з послідовним"
            zamiry.append(z)
        tochky.append({"procesiv": n, "zamiry": zamiry})
        print("  %-11s %s" % ("послідовно" if n == 0 else "процесів %d" % n,
                              "  ".join("%.3f c" % z["razom"] for z in zamiry)))
    print("assert: усі прогони збіглися з послідовною версією")
    print()

    posl = min(z["razom"] for z in tochky[0]["zamiry"])
    print("| Процесів | Прогін 1 | Прогін 2 | Прогін 3 | Мінімум | Прискорення | Ефективність |")
    print("|---|---|---|---|---|---|---|")
    for t in tochky[1:]:
        n, chasy = t["procesiv"], [z["razom"] for z in t["zamiry"]]
        s = posl / min(chasy)
        print("| %d | %s | %.3f c | ×%.2f | %.0f %% |"
              % (n, " | ".join("%.3f c" % c for c in chasy), min(chasy), s, 100 * s / n))
    print()

    print("Розклад найшвидшого прогону:")
    print("| Процесів | Запуск, c | Робота, c | Запис, c | Прискорення роботи |")
    print("|---|---|---|---|---|")
    posl_robota = min(z["robota"] for z in tochky[0]["zamiry"])
    for t in tochky:
        z = min(t["zamiry"], key=lambda z: z["razom"])
        print("| %s | %.3f | %.3f | %.3f | ×%.2f |"
              % (t["procesiv"] or "посл.", z["zapusk"], z["robota"], z["zapys"],
                 posl_robota / z["robota"]))
    print()

    print("Вартість пулу (без роботи):")
    pul = {}
    for n in (2, 4, 8, os.cpu_count()):
        pas, start = vartist_pulu(n)
        pul[n] = {"pass": pas, "start": start}
        print("  %2d процесів:  with ... pass  %.3f c   зі стартом процесів  %.3f c"
              % (n, pas, start))
    print()

    zapys = min(z["zapys"] for z in tochky[0]["zamiry"])
    s = zapys / posl
    n = os.cpu_count()
    print("Амдал: послідовна частка %.2f %%, стеля на %d ядрах ×%.1f, на 8 ×%.1f"
          % (100 * s, n, 1 / (s + (1 - s) / n), 1 / (s + (1 - s) / 8)))

    with open(ZAMIRY, "w", encoding="utf-8") as f:
        json.dump({"posl_min": posl, "fizychnykh": psutil.cpu_count(logical=False),
                   "logichnykh": os.cpu_count(), "tochky": tochky, "pul": pul},
                  f, ensure_ascii=False, indent=2)


def chunksize():
    """⭐ Той самий Мандельброт на cpu_count() процесах з різним chunksize."""
    n = os.cpu_count()
    base, _ = poslidovno()
    print("chunksize, %d процесів, %d рядків, мінімум з %d прогонів:" % (n, m.VYSOTA, PROGONIV))
    print("| chunksize | Шматків | Мінімум, c |")
    print("|---|---|---|")
    for ch in (1, 8, 50, 200):
        chasy = []
        for _ in range(PROGONIV):
            rezultat, z = procesamy(n, chunksize=ch)
            assert rezultat == base
            chasy.append(z["razom"])
        print("| %d | %d | %.3f |" % (ch, -(-m.VYSOTA // ch), min(chasy)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunksize", action="store_true")
    if ap.parse_args().chunksize:
        chunksize()
    else:
        osnovnyi()
