"""
РСПО, практична 3. Множина Мандельброта — потокова версія (ThreadPoolExecutor).

Задача та сама, що в 1-poslidovno: рядки незалежні, тож кожен рядок —
окрема одиниця роботи для пулу потоків. Міняється тільки те, ХТО рахує
рядки: раніше цикл, тепер `pool.map(obchyslyty_riadok, range(vysota))`.

ЩО МІРЯЄТЬСЯ (кожен прогін окремо, time.perf_counter):

  пул         — створення ThreadPoolExecutor і його shutdown;
  обчислення  — pool.map по всіх рядках (частина, яку роздали на потоки);
  запис       — байти PGM + файл (послідовна частина, не ділиться);
  разом       — пул + обчислення + запис.

Еталон (base) рахується цим самим скриптом і в тому самому сеансі —
у zamiry.md з практичної 02 записано, що розкид МІЖ сеансами більший,
ніж усередині сеансу, тому порівнювати чужі серії не можна.

  assert rezultat == base — у кожному прогоні кожної кількості потоків.

ЗАПУСК:  python potoky.py                       (≈ 3 хв)
         python potoky.py --shyryna 150 --vysota 100   (швидка перевірка)
"""

import argparse
import functools
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor

KOREN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KOREN, "1-poslidovno"))
import mandelbrot as m  # noqa: E402  (шлях треба додати до імпорту)

PROGONIV = 3
TUT = os.path.dirname(os.path.abspath(__file__))
FAIL = os.path.join(TUT, "mandelbrot.pgm")
ZAMIRY = os.path.join(TUT, "zamiry-obchyslennya.json")


def kilkosti_potokiv():
    """2, 4, 8 і cpu_count() — без повторів, якщо cpu_count() = 8."""
    spysok = [2, 4, 8]
    if os.cpu_count() not in spysok:
        spysok.append(os.cpu_count())
    return spysok


def obchyslyty_potokamy(n_potokiv, shyryna, vysota, max_iter):
    """Ті самі рядки, але через пул потоків. Повертає (рядки, час_пулу, час_обчислення)."""
    robota = functools.partial(m.obchyslyty_riadok, shyryna=shyryna,
                               vysota=vysota, max_iter=max_iter)
    t0 = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=n_potokiv)
    t1 = time.perf_counter()
    # map зберігає порядок рядків, тож результат порівнюваний з еталоном
    rezultat = list(pool.map(robota, range(vysota)))
    t2 = time.perf_counter()
    pool.shutdown(wait=True)
    t3 = time.perf_counter()
    return rezultat, (t1 - t0) + (t3 - t2), t2 - t1


def progin(n_potokiv, shyryna, vysota, max_iter):
    """Один прогін: пул -> обчислення -> запис. n_potokiv=1 — послідовний еталон."""
    if n_potokiv == 1:
        t0 = time.perf_counter()
        rezultat = m.obchyslyty(shyryna, vysota, max_iter)
        pul, obch = 0.0, time.perf_counter() - t0
    else:
        rezultat, pul, obch = obchyslyty_potokamy(n_potokiv, shyryna, vysota, max_iter)

    t0 = time.perf_counter()
    m.zapysaty_pgm(rezultat, max_iter, FAIL)
    zapys = time.perf_counter() - t0
    return rezultat, pul, obch, zapys, pul + obch + zapys


def seriya(n_potokiv, base, shyryna, vysota, max_iter):
    """PROGONIV прогонів однієї кількості потоків. Повертає список часів."""
    nazva = "послідовно  " if n_potokiv == 1 else "потоків %-4d" % n_potokiv
    zamiry = []
    for i in range(1, PROGONIV + 1):
        rezultat, pul, obch, zapys, razom = progin(n_potokiv, shyryna, vysota, max_iter)
        if base is None:
            base = rezultat
        assert rezultat == base, "результат розійшовся з послідовним"
        print("  %s прогін %d:  пул %7.4f c   обчислення %7.3f c   "
              "запис %6.4f c   разом %7.3f c" % (nazva, i, pul, obch, zapys, razom))
        zamiry.append({"pul": pul, "obchyslennya": obch, "zapys": zapys, "razom": razom})
    return base, zamiry


def pasport(shyryna, vysota, max_iter):
    print("=" * 78)
    print("УМОВИ ЗАПУСКУ")
    print("=" * 78)
    print("  Python        :", sys.version.split()[0], platform.python_implementation())
    print("  процесор      :", platform.processor())
    print("  логічних ядер :", os.cpu_count())
    print("  ОС            :", platform.platform())
    print("  розмір        : %d×%d, max_iter=%d, ділянка [%g, %g]×[%g, %g]"
          % (shyryna, vysota, max_iter, m.X_MIN, m.X_MAX, m.Y_MIN, m.Y_MAX))
    print("  одиниця роботи: один рядок зображення, усього %d рядків" % vysota)
    print("  потоків       :", ", ".join(map(str, kilkosti_potokiv())),
          "(+ послідовний еталон у цьому ж сеансі)")
    print("  прогонів      :", PROGONIV)
    print("  у замір входять: створення пулу + обчислення + запис PGM")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shyryna", type=int, default=m.SHYRYNA)
    ap.add_argument("--vysota", type=int, default=m.VYSOTA)
    ap.add_argument("--max-iter", type=int, default=m.MAX_ITER)
    a = ap.parse_args()

    pasport(a.shyryna, a.vysota, a.max_iter)
    print("=" * 78)
    print("ПРОГОНИ")
    print("=" * 78)

    base, posl = seriya(1, None, a.shyryna, a.vysota, a.max_iter)
    posl_min = min(z["razom"] for z in posl)

    vsi = [{"potoky": 1, "zamiry": posl}]
    for n in kilkosti_potokiv():
        _, zamiry = seriya(n, base, a.shyryna, a.vysota, a.max_iter)
        vsi.append({"potoky": n, "zamiry": zamiry})

    print()
    print("=" * 78)
    print("ПІДСУМОК (для zamiry.md)")
    print("=" * 78)
    print("  еталон у цьому сеансі : %.3f c (мінімум з %d прогонів)" % (posl_min, PROGONIV))
    print("  збіг із еталоном      : так, усі %d прогони (assert rezultat == base)"
          % sum(len(v["zamiry"]) for v in vsi))
    print()
    print("| Реалізація | Прогін 1 | Прогін 2 | Прогін 3 | Мінімум | Прискорення |")
    print("|---|---|---|---|---|---|")
    for v in vsi:
        nazva = "послідовно" if v["potoky"] == 1 else "потоків %d" % v["potoky"]
        chasy = [z["razom"] for z in v["zamiry"]]
        v["min"] = min(chasy)
        v["pryskorennya"] = posl_min / v["min"]
        print("| %s | %s | %.3f c | ×%.2f |"
              % (nazva, " | ".join("%.3f c" % t for t in chasy), v["min"], v["pryskorennya"]))

    print()
    print("| Реалізація | Пул, c | Обчислення, c | Запис, c |")
    print("|---|---|---|---|")
    for v in vsi:
        nazva = "послідовно" if v["potoky"] == 1 else "потоків %d" % v["potoky"]
        i = min(range(len(v["zamiry"])), key=lambda k: v["zamiry"][k]["razom"])
        z = v["zamiry"][i]
        print("| %s | %.4f | %.3f | %.4f |" % (nazva, z["pul"], z["obchyslennya"], z["zapys"]))

    # швидка перевірка іншого розміру не повинна затирати справжній замір
    povnyi = (a.shyryna, a.vysota, a.max_iter) == (m.SHYRYNA, m.VYSOTA, m.MAX_ITER)
    if not povnyi:
        print()
        print("Розмір не той, що в заміру, — %s не чіпаю." % os.path.basename(ZAMIRY))
        raise SystemExit

    with open(ZAMIRY, "w", encoding="utf-8") as f:
        json.dump({"zadacha": "obchyslennya", "posl_min": posl_min,
                   "shyryna": a.shyryna, "vysota": a.vysota, "max_iter": a.max_iter,
                   "cpu_count": os.cpu_count(), "progoniv": PROGONIV, "tochky": vsi},
                  f, ensure_ascii=False, indent=2)
    print()
    print("Збережено:", ZAMIRY)
