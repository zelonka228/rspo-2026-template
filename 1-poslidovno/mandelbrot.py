"""
РСПО, практична 2. Множина Мандельброта — послідовна версія.

Базова реалізація, відносно якої міряється все інше. Тільки стандартна
бібліотека, жодного numpy.

ЩО МІРЯЄТЬСЯ (кожен прогін окремо, time.perf_counter):

  обчислення — кількість ітерацій для кожного пікселя
               (частина, яку можна роздати на ядра);
  запис      — зібрати всі точки в одне зображення (байти PGM)
               і записати файл (послідовна частина, не ділиться).

  Частка запису від загального часу — це виміряна послідовна частка s
  для закону Амдала.

ЗАПУСК:  python mandelbrot.py
         python mandelbrot.py --shyryna 600 --vysota 400   (швидка перевірка)
"""

import argparse
import os
import platform
import sys
import time

# підібрано так, щоб один прогін тривав 5–15 с на Ryzen 7 2700
SHYRYNA = 600
VYSOTA = 400
MAX_ITER = 500

# ділянка комплексної площини: 3 × 2, як і співвідношення сторін картинки
X_MIN, X_MAX = -2.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0

PROGONIV = 3
FAIL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mandelbrot.pgm")


def iteratsii(c, max_iter):
    """Скільки кроків z -> z*z + c до втечі за |z| > 2 (або max_iter)."""
    z = 0j
    for n in range(max_iter):
        z = z * z + c
        if abs(z) > 2.0:
            return n
    return max_iter


def obchyslyty_riadok(y, shyryna, vysota, max_iter):
    """Один рядок зображення. Рядки незалежні — одиниця роботи для паралелі."""
    im = Y_MIN + (Y_MAX - Y_MIN) * y / vysota
    krok = (X_MAX - X_MIN) / shyryna
    return [iteratsii(complex(X_MIN + krok * x, im), max_iter)
            for x in range(shyryna)]


def obchyslyty(shyryna, vysota, max_iter):
    """Уся картинка: список рядків, у кожному — кількості ітерацій."""
    return [obchyslyty_riadok(y, shyryna, vysota, max_iter)
            for y in range(vysota)]


def zapysaty_pgm(rezultat, max_iter, shlyakh):
    """Послідовна частина: ітерації -> відтінки сірого -> файл PGM (P5)."""
    vysota = len(rezultat)
    shyryna = len(rezultat[0])
    # точки множини (max_iter) чорні, решта — світліші, чим швидше втекли
    tablytsia = bytes(255 - 255 * n // max_iter for n in range(max_iter + 1))
    dani = bytearray()
    for riadok in rezultat:
        dani += bytes(map(tablytsia.__getitem__, riadok))
    with open(shlyakh, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (shyryna, vysota))
        f.write(dani)


def pasport(shyryna, vysota, max_iter):
    print("=" * 68)
    print("УМОВИ ЗАПУСКУ")
    print("=" * 68)
    print("  Python        :", sys.version.split()[0], platform.python_implementation())
    print("  процесор      :", platform.processor())
    print("  логічних ядер :", os.cpu_count())
    print("  ОС            :", platform.platform())
    print("  розмір        : %d×%d, max_iter=%d, ділянка [%g, %g]×[%g, %g]"
          % (shyryna, vysota, max_iter, X_MIN, X_MAX, Y_MIN, Y_MAX))
    print("  прогонів      :", PROGONIV)
    print()


def prognaty(nomer, shyryna, vysota, max_iter):
    t0 = time.perf_counter()
    rezultat = obchyslyty(shyryna, vysota, max_iter)
    t1 = time.perf_counter()
    zapysaty_pgm(rezultat, max_iter, FAIL)
    t2 = time.perf_counter()

    obch, zapys, razom = t1 - t0, t2 - t1, t2 - t0
    print("  прогін %d:  обчислення %7.3f c   запис %6.4f c   разом %7.3f c   "
          "частка запису %.3f %%" % (nomer, obch, zapys, razom, 100 * zapys / razom))
    return rezultat, obch, zapys, razom


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shyryna", type=int, default=SHYRYNA)
    ap.add_argument("--vysota", type=int, default=VYSOTA)
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    a = ap.parse_args()

    pasport(a.shyryna, a.vysota, a.max_iter)
    print("=" * 68)
    print("ПРОГОНИ")
    print("=" * 68)

    base = None
    zamiry = []
    for i in range(1, PROGONIV + 1):
        rezultat, *chasy = prognaty(i, a.shyryna, a.vysota, a.max_iter)
        if base is None:
            base = rezultat
        assert rezultat == base, "прогони дали різний результат"
        zamiry.append(chasy)

    obch = [z[0] for z in zamiry]
    zapys = [z[1] for z in zamiry]
    razom = [z[2] for z in zamiry]
    i_min = razom.index(min(razom))

    print()
    print("=" * 68)
    print("ПІДСУМОК (для zamiry.md)")
    print("=" * 68)
    print("  результат        :", FAIL, "(%d байт)" % os.path.getsize(FAIL))
    print("  точок у множині  : %d з %d"
          % (sum(r.count(a.max_iter) for r in base), a.shyryna * a.vysota))
    print("  ітерацій усього  : %d" % sum(map(sum, base)))
    print()
    print("| Що | Прогін 1 | Прогін 2 | Прогін 3 | Мінімум |")
    print("|---|---|---|---|---|")
    for nazva, ryad in (("обчислення, c", obch), ("запис, c", zapys),
                        ("разом, c", razom)):
        print("| %s | %s | %.3f |" % (nazva, " | ".join("%.3f" % t for t in ryad),
                                      min(ryad)))
    print("| частка запису, %% | %s | %.3f |"
          % (" | ".join("%.3f" % (100 * z / r) for z, r in zip(zapys, razom)),
             100 * zapys[i_min] / razom[i_min]))
