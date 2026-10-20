"""
РСПО, практична 3. Та сама схема з потоками, але задача — ОЧІКУВАННЯ.

32 виклики `time.sleep(0.5)` замість 32 порцій обчислень. Код навколо
той самий: `ThreadPoolExecutor(max_workers=N).map(...)`, ті самі кількості
потоків, ті самі три прогони. Єдина різниця — усередині кожного завдання
процесор не рахує, а чекає.

Це друга з «двох версій, які суперечать одна одній»: у potoky.py потоки
нічого не дають, тут вони дають майже ×N. Порівняння — у vysnovky.md.

  ЗАВДАНЬ = 32, КОЖНЕ = 0.5 c  ->  послідовно рівно 16 c.
  Ідеал для N потоків: 16 / N (доки N <= 32).

  base — очікувана відповідь: кожне завдання повернуло свій номер,
  усі 32 на місці й у тому ж порядку. assert rezultat == base.

ЗАПУСК:  python ochikuvannya.py         (≈ 1 хв)
"""

import argparse
import functools
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ZAVDAN = 32
ZATRYMKA = 0.5
PROGONIV = 3
TUT = os.path.dirname(os.path.abspath(__file__))
ZAMIRY = os.path.join(TUT, "zamiry-ochikuvannya.json")


def kilkosti_potokiv():
    spysok = [2, 4, 8]
    if os.cpu_count() not in spysok:
        spysok.append(os.cpu_count())
    return spysok


def odne_zavdannya(nomer, zatrymka):
    """Одиниця роботи: чекаємо й повертаємо свій номер. GIL на час sleep відпускається."""
    time.sleep(zatrymka)
    return nomer


def progin(n_potokiv, zavdan, zatrymka):
    """Один прогін. n_potokiv=1 — послідовний еталон (звичайний цикл)."""
    robota = functools.partial(odne_zavdannya, zatrymka=zatrymka)
    if n_potokiv == 1:
        t0 = time.perf_counter()
        rezultat = [robota(i) for i in range(zavdan)]
        return rezultat, 0.0, time.perf_counter() - t0

    t0 = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=n_potokiv)
    t1 = time.perf_counter()
    rezultat = list(pool.map(robota, range(zavdan)))
    t2 = time.perf_counter()
    pool.shutdown(wait=True)
    t3 = time.perf_counter()
    return rezultat, (t1 - t0) + (t3 - t2), t2 - t1


def seriya(n_potokiv, base, zavdan, zatrymka):
    nazva = "послідовно  " if n_potokiv == 1 else "потоків %-4d" % n_potokiv
    zamiry = []
    for i in range(1, PROGONIV + 1):
        rezultat, pul, ochik = progin(n_potokiv, zavdan, zatrymka)
        assert rezultat == base, "завдання виконались не всі або не в тому порядку"
        razom = pul + ochik
        print("  %s прогін %d:  пул %7.4f c   очікування %7.3f c   разом %7.3f c"
              % (nazva, i, pul, ochik, razom))
        zamiry.append({"pul": pul, "ochikuvannya": ochik, "razom": razom})
    return zamiry


def pasport(zavdan, zatrymka):
    print("=" * 78)
    print("УМОВИ ЗАПУСКУ")
    print("=" * 78)
    print("  Python        :", sys.version.split()[0], platform.python_implementation())
    print("  процесор      :", platform.processor())
    print("  логічних ядер :", os.cpu_count())
    print("  ОС            :", platform.platform())
    print("  задача        : %d × time.sleep(%.1f) — чисте очікування, 0 %% CPU"
          % (zavdan, zatrymka))
    print("  одиниця роботи: одне очікування, усього %d завдань" % zavdan)
    print("  потоків       :", ", ".join(map(str, kilkosti_potokiv())),
          "(+ послідовний еталон у цьому ж сеансі)")
    print("  прогонів      :", PROGONIV)
    print("  у замір входять: створення пулу + очікування")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zavdan", type=int, default=ZAVDAN)
    ap.add_argument("--zatrymka", type=float, default=ZATRYMKA)
    a = ap.parse_args()

    pasport(a.zavdan, a.zatrymka)
    print("=" * 78)
    print("ПРОГОНИ")
    print("=" * 78)

    base = list(range(a.zavdan))
    posl = seriya(1, base, a.zavdan, a.zatrymka)
    posl_min = min(z["razom"] for z in posl)

    vsi = [{"potoky": 1, "zamiry": posl}]
    for n in kilkosti_potokiv():
        vsi.append({"potoky": n, "zamiry": seriya(n, base, a.zavdan, a.zatrymka)})

    print()
    print("=" * 78)
    print("ПІДСУМОК (для zamiry.md)")
    print("=" * 78)
    print("  теоретично послідовно : %.1f c (%d × %.1f)"
          % (a.zavdan * a.zatrymka, a.zavdan, a.zatrymka))
    print("  еталон у цьому сеансі : %.3f c (мінімум з %d прогонів)" % (posl_min, PROGONIV))
    print("  збіг із еталоном      : так, усі %d прогони (assert rezultat == base)"
          % sum(len(v["zamiry"]) for v in vsi))
    print()
    print("| Реалізація | Прогін 1 | Прогін 2 | Прогін 3 | Мінімум | Прискорення | Ідеал |")
    print("|---|---|---|---|---|---|---|")
    for v in vsi:
        nazva = "послідовно" if v["potoky"] == 1 else "потоків %d" % v["potoky"]
        chasy = [z["razom"] for z in v["zamiry"]]
        v["min"] = min(chasy)
        v["pryskorennya"] = posl_min / v["min"]
        # завдань 32, потоків N: кожен потік отримує ceil(32/N) хвиль по 0.5 c
        khvyl = -(-a.zavdan // v["potoky"])
        v["ideal"] = (a.zavdan * a.zatrymka) / (khvyl * a.zatrymka)
        print("| %s | %s | %.3f c | ×%.2f | ×%.2f |"
              % (nazva, " | ".join("%.3f c" % t for t in chasy), v["min"],
                 v["pryskorennya"], v["ideal"]))

    povnyi = (a.zavdan, a.zatrymka) == (ZAVDAN, ZATRYMKA)
    if not povnyi:
        print()
        print("Параметри не ті, що в заміру, — %s не чіпаю." % os.path.basename(ZAMIRY))
        raise SystemExit

    with open(ZAMIRY, "w", encoding="utf-8") as f:
        json.dump({"zadacha": "ochikuvannya", "posl_min": posl_min,
                   "zavdan": a.zavdan, "zatrymka": a.zatrymka,
                   "cpu_count": os.cpu_count(), "progoniv": PROGONIV, "tochky": vsi},
                  f, ensure_ascii=False, indent=2)
    print()
    print("Збережено:", ZAMIRY)
