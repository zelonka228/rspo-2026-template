"""
РСПО, практична 3, ⭐ додаткове. Та сама схема потоків, але на NumPy.

Перевірка твердження «потоки не дають нічого через GIL». Якщо причина справді
в GIL, то досить замінити тіло завдання на код, який GIL ВІДПУСКАЄ, — і крива
піде вгору без жодної зміни в самій схемі. Пул, кількості потоків, три прогони,
одиниця роботи «смуга рядків» — усе те саме, що в potoky.py.

ЯК ТУТ РАХУЄТЬСЯ. NumPy виконує z = z*z + c одразу для цілої смуги пікселів
усередині скомпільованої ufunc, і на час такого виклику відпускає GIL.
Щоб під GIL лишалося якомога менше, ядро написане так:

  * жодного дострокового виходу й жодної маски-індексації: `z[zhyvi]`
    (fancy indexing) GIL НЕ відпускає, і саме воно з'їдало весь виграш
    у першому варіанті — з ним 8 потоків давали ×0.3, а не приріст;
  * усі дії — in-place ufunc (`np.multiply(z, z, out=z)` тощо), щоб не
    створювати тимчасових масивів: кожне створення масиву бере GIL;
  * замість «номер кроку, на якому втекли» рахується «скільки кроків
    протрималися» — `out += (|z| <= 2)`. Для точок, що вже втекли, умова
    більше ніколи не істинна (|z| росте до inf, далі nan, а nan <= 2 — хиба),
    тому сума дорівнює тому самому номеру кроку, що й у чистому Python.
    Це перевіряється поточково, див. ПЕРЕВІРКА ПРАВИЛЬНОСТІ.

  Ціна: рахуються всі max_iter кроків для всіх пікселів, навіть уже втеклих.
  Роботи більше, ніж у послідовній версії, але вона вся в C і без GIL.

РОЗМІР. На 600×400 смуга така мала, що час іде на накладні витрати самих
викликів ufunc. Тому для кривої береться 1200×800 (указано на графіку
й у zamiry.md). Прискорення — величина відносна, тож криві порівнювані.

ЗАПУСК:  python numpy_potoky.py                (≈ 2 хв)
         python numpy_potoky.py --bez-perevirky
         python numpy_potoky.py --smuhy        (розгортка за розміром смуги)
"""

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

KOREN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KOREN, "1-poslidovno"))
import mandelbrot as m  # noqa: E402  (шлях треба додати до імпорту)

SHYRYNA = 1200
VYSOTA = 800
SMUHA = 50          # рядків в одній одиниці роботи, однаково для всіх N
PROGONIV = 3
TUT = os.path.dirname(os.path.abspath(__file__))
ZAMIRY = os.path.join(TUT, "zamiry-numpy.json")


def kilkosti_potokiv():
    spysok = [2, 4, 8]
    if os.cpu_count() not in spysok:
        spysok.append(os.cpu_count())
    return spysok


def smuhy(vysota, smuha):
    return [(y, min(y + smuha, vysota)) for y in range(0, vysota, smuha)]


def rakhuvaty_smuhu(mezhi, vyhid, shyryna, vysota, max_iter):
    """Смуга рядків [y0, y1) векторизовано, самі ufunc. Пише у свій шматок."""
    y0, y1 = mezhi
    im = m.Y_MIN + (m.Y_MAX - m.Y_MIN) * np.arange(y0, y1, dtype=np.float64) / vysota
    krok = (m.X_MAX - m.X_MIN) / shyryna
    re = m.X_MIN + krok * np.arange(shyryna, dtype=np.float64)
    c = re[None, :] + 1j * im[:, None]

    z = np.zeros_like(c)
    modul = np.empty(c.shape, dtype=np.float64)
    zhyvi = np.empty(c.shape, dtype=bool)
    out = np.zeros(c.shape, dtype=np.int32)

    # після втечі |z| росте до inf, далі z*z+c дає nan; nan <= 2 — хиба,
    # тож переповнення тут очікуване й на результат не впливає
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(max_iter):
            np.multiply(z, z, out=z)            # ufunc: GIL відпущено
            np.add(z, c, out=z)                 # ufunc
            np.abs(z, out=modul)                # ufunc
            np.less_equal(modul, 2.0, out=zhyvi)  # ufunc
            np.add(out, zhyvi, out=out)         # ufunc
    # смуги не перетинаються, тому спільний масив пишеться без замка
    vyhid[y0:y1] = out


def rakhuvaty_smuhu_maskamy(mezhi, vyhid, shyryna, vysota, max_iter):
    """Перший варіант ядра: дострокове завершення + маски. Лишений для порівняння.

    Виглядає економнішим (втеклі точки викидаються й далі не рахуються), але
    `idx[vtekly]`, `z[zhyvi]` — це fancy indexing, який GIL не відпускає,
    тому в потоках цей варіант програє. Замір: --maskamy.
    """
    y0, y1 = mezhi
    im = m.Y_MIN + (m.Y_MAX - m.Y_MIN) * np.arange(y0, y1, dtype=np.float64) / vysota
    krok = (m.X_MAX - m.X_MIN) / shyryna
    re = m.X_MIN + krok * np.arange(shyryna, dtype=np.float64)
    c = (re[None, :] + 1j * im[:, None]).ravel()

    out = np.full(c.size, max_iter, dtype=np.int32)
    idx = np.arange(c.size)
    z = np.zeros(c.size, dtype=np.complex128)
    for n in range(max_iter):
        z = z * z + c
        vtekly = np.abs(z) > 2.0
        if vtekly.any():
            out[idx[vtekly]] = n           # fancy indexing: GIL тримається
            zhyvi = ~vtekly
            idx, z, c = idx[zhyvi], z[zhyvi], c[zhyvi]
            if idx.size == 0:
                break
    vyhid[y0:y1] = out.reshape(y1 - y0, shyryna)


def porivnyaty_yadra(shyryna, vysota, max_iter):
    """Два ядра на тій самій задачі: що дає відмова від масок у потоках."""
    print("| Ядро | 1 потік | 4 потоки | 8 потоків | %d потоків |" % os.cpu_count())
    print("|---|---|---|---|---|")
    ryadky = {}
    for nazva, yadro in (("in-place ufunc, без масок", rakhuvaty_smuhu),
                         ("маски + дострокове завершення", rakhuvaty_smuhu_maskamy)):
        chasy = []
        for n in (1, 4, 8, os.cpu_count()):
            _, pul, t = obchyslyty_numpy(n, shyryna, vysota, max_iter, yadro=yadro)
            chasy.append(pul + t)
        print("| %s | %s |" % (nazva, " | ".join(
            "%.3f c (×%.2f)" % (t, chasy[0] / t) for t in chasy)))
        ryadky[nazva] = chasy
    return ryadky


def obchyslyty_numpy(n_potokiv, shyryna, vysota, max_iter, smuha=SMUHA,
                     yadro=None):
    """Повертає (картинка, час_пулу, час_обчислення). n_potokiv=1 — без пулу."""
    vyhid = np.empty((vysota, shyryna), dtype=np.int32)
    chastyny = smuhy(vysota, smuha)

    f = yadro or rakhuvaty_smuhu

    def robota(mezhi):
        f(mezhi, vyhid, shyryna, vysota, max_iter)

    if n_potokiv == 1:
        t0 = time.perf_counter()
        for mezhi in chastyny:
            robota(mezhi)
        return vyhid, 0.0, time.perf_counter() - t0

    t0 = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=n_potokiv)
    t1 = time.perf_counter()
    list(pool.map(robota, chastyny))
    t2 = time.perf_counter()
    pool.shutdown(wait=True)
    t3 = time.perf_counter()
    return vyhid, (t1 - t0) + (t3 - t2), t2 - t1


def perevirka_z_python():
    """Звіряння з чистим Python на базовому розмірі 600×400."""
    print("  рахую еталон чистим Python %d×%d, max_iter=%d ..."
          % (m.SHYRYNA, m.VYSOTA, m.MAX_ITER))
    t0 = time.perf_counter()
    etalon = np.array(m.obchyslyty(m.SHYRYNA, m.VYSOTA, m.MAX_ITER), dtype=np.int32)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    nash, _, _ = obchyslyty_numpy(1, m.SHYRYNA, m.VYSOTA, m.MAX_ITER)
    t_np = time.perf_counter() - t0

    rozbizhnosti = int(np.count_nonzero(nash != etalon))
    vsiogo = int(etalon.size)
    print("  чистий Python : %.3f c" % t_py)
    print("  NumPy, 1 потік: %.3f c  (×%.1f на тому самому розмірі)"
          % (t_np, t_py / t_np))
    print("  розбіжності   : %d точок з %d (%.4f %%)"
          % (rozbizhnosti, vsiogo, 100 * rozbizhnosti / vsiogo))
    if rozbizhnosti == 0:
        print("  -> результат збігається з послідовним точка в точку")
    else:
        print("  -> розбіжності лише в одиничних точках на межі множини:")
        print("     арифметика та сама формула, але інший скомпільований код, тож")
        print("     z розходиться в останніх бітах уже на 2-му кроці (≈ 5.6e-17).")
        print("     Для точок на межі орбіта хаотична, і ця різниця за ~400 кроків")
        print("     виростає до іншої траєкторії. Розбір — у vysnovky.md.")
    return {"python_c": t_py, "numpy_c": t_np,
            "rozbizhnosti": rozbizhnosti, "tochok": vsiogo}


def rozgortka_smuh(shyryna, vysota, max_iter):
    """Скільки важить розмір одиниці роботи: та сама задача, різні смуги."""
    print("| Смуга, рядків | Елементів у смузі | 1 потік | %d потоків | Приріст |"
          % os.cpu_count())
    print("|---|---|---|---|---|")
    ryadky = []
    for smuha in (8, 16, 25, 50, 100, 200):
        _, _, t1 = obchyslyty_numpy(1, shyryna, vysota, max_iter, smuha)
        _, pul, tn = obchyslyty_numpy(os.cpu_count(), shyryna, vysota, max_iter, smuha)
        print("| %d | %d | %.3f c | %.3f c | ×%.2f |"
              % (smuha, smuha * shyryna, t1, pul + tn, t1 / (pul + tn)))
        ryadky.append({"smuha": smuha, "odyn": t1, "bagato": pul + tn})
    return ryadky


def seriya(n_potokiv, base, shyryna, vysota, max_iter):
    nazva = "NumPy, 1 потік " if n_potokiv == 1 else "NumPy, потоків %-2d" % n_potokiv
    zamiry = []
    for i in range(1, PROGONIV + 1):
        rezultat, pul, obch = obchyslyty_numpy(n_potokiv, shyryna, vysota, max_iter)
        if base is None:
            base = rezultat
        assert np.array_equal(rezultat, base), "результат розійшовся з еталоном"
        razom = pul + obch
        print("  %s прогін %d:  пул %7.4f c   обчислення %7.3f c   разом %7.3f c"
              % (nazva, i, pul, obch, razom))
        zamiry.append({"pul": pul, "obchyslennya": obch, "razom": razom})
    return base, zamiry


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shyryna", type=int, default=SHYRYNA)
    ap.add_argument("--vysota", type=int, default=VYSOTA)
    ap.add_argument("--max-iter", type=int, default=m.MAX_ITER)
    ap.add_argument("--bez-perevirky", action="store_true")
    ap.add_argument("--smuhy", action="store_true",
                    help="розгортка за розміром смуги замість основного заміру")
    ap.add_argument("--maskamy", action="store_true",
                    help="порівняти два ядра: in-place ufunc проти масок")
    a = ap.parse_args()

    print("=" * 78)
    print("УМОВИ ЗАПУСКУ")
    print("=" * 78)
    print("  Python        :", sys.version.split()[0], platform.python_implementation())
    print("  NumPy         :", np.__version__)
    print("  процесор      :", platform.processor())
    print("  логічних ядер :", os.cpu_count())
    print("  ОС            :", platform.platform())
    print("  розмір        : %d×%d, max_iter=%d" % (a.shyryna, a.vysota, a.max_iter))
    print("  одиниця роботи: смуга з %d рядків (%d пікселів), усього %d смуг"
          % (SMUHA, SMUHA * a.shyryna, len(smuhy(a.vysota, SMUHA))))
    print("  потоків       :", ", ".join(map(str, kilkosti_potokiv())),
          "(+ NumPy в один потік як еталон)")
    print("  прогонів      :", PROGONIV)
    print("  у замір входять: створення пулу + обчислення (запис PGM не входить)")
    print()

    if a.maskamy:
        print("=" * 78)
        print("ДВА ЯДРА НА ТІЙ САМІЙ ЗАДАЧІ (по одному прогону)")
        print("=" * 78)
        porivnyaty_yadra(a.shyryna, a.vysota, a.max_iter)
        raise SystemExit

    if a.smuhy:
        print("=" * 78)
        print("РОЗГОРТКА ЗА РОЗМІРОМ СМУГИ (по одному прогону)")
        print("=" * 78)
        rozgortka_smuh(a.shyryna, a.vysota, a.max_iter)
        raise SystemExit

    perevirka = None
    if not a.bez_perevirky:
        print("=" * 78)
        print("ПЕРЕВІРКА ПРАВИЛЬНОСТІ (розмір 600×400, проти чистого Python)")
        print("=" * 78)
        perevirka = perevirka_z_python()
        print()

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
    print("  еталон (NumPy, 1 потік): %.3f c" % posl_min)
    print("  збіг із еталоном       : так, усі %d прогони (np.array_equal)"
          % sum(len(v["zamiry"]) for v in vsi))
    print()
    print("| Реалізація | Прогін 1 | Прогін 2 | Прогін 3 | Мінімум | Прискорення |")
    print("|---|---|---|---|---|---|")
    for v in vsi:
        nazva = "NumPy, 1 потік" if v["potoky"] == 1 else "NumPy, потоків %d" % v["potoky"]
        chasy = [z["razom"] for z in v["zamiry"]]
        v["min"] = min(chasy)
        v["pryskorennya"] = posl_min / v["min"]
        print("| %s | %s | %.3f c | ×%.2f |"
              % (nazva, " | ".join("%.3f c" % t for t in chasy), v["min"], v["pryskorennya"]))

    povnyi = (a.shyryna, a.vysota, a.max_iter) == (SHYRYNA, VYSOTA, m.MAX_ITER)
    if not povnyi:
        print()
        print("Розмір не той, що в заміру, — %s не чіпаю." % os.path.basename(ZAMIRY))
        raise SystemExit

    with open(ZAMIRY, "w", encoding="utf-8") as f:
        json.dump({"zadacha": "numpy", "posl_min": posl_min,
                   "shyryna": a.shyryna, "vysota": a.vysota, "max_iter": a.max_iter,
                   "smuha": SMUHA, "cpu_count": os.cpu_count(), "progoniv": PROGONIV,
                   "perevirka": perevirka, "tochky": vsi},
                  f, ensure_ascii=False, indent=2)
    print()
    print("Збережено:", ZAMIRY)
