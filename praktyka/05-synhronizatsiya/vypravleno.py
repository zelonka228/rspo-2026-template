"""
РСПО, практична 5. Виправлена версія zlamane.py.

ЩО БУЛО НЕ ТАК

  Модель складу: кілька процесів списують товар зі спільного залишку.
  Обидва інваріанти (залишок >= 0 і відвантажено + залишок = запас) ламались,
  бо у функції obrobyty_zamovlennya два різні місця, де між діями
  міг втрутитись інший процес:

    1. Читання-зміна-запис: `x.value = x.value - k` це три дії
       (прочитати, відняти, записати). Якщо інший процес записав своє
       значення між читанням і записом, його списання зникає.
       Це ламає інваріант 2 (баланс).

    2. Перевірка-дія: `if zalyshok.value >= kilkist` і списання окремі дії.
       Поки процес списує, інший встигає забрати товар, і перевірка
       застаріває. Це ламає інваріант 1 (від'ємний залишок).

ЩО ЗРОБЛЕНО

  Один мультипроцесорний замок mp.Lock() і `with zamok:` навколо
  мінімальної ділянки: перевірка + списання + збільшення відвантаженого
  (або збільшення відмов). Генерація випадкової кількості лишається
  поза замком, тож процеси не чекають одне одного без потреби.

  Щоб видно було, що паралельність лишилась, кожен процес рахує свої
  відвантаження локально (moi) і друкує їх наприкінці. Якщо замок
  закрити навколо всього циклу, один процес забере все, а решта нічого.

ЗАПУСК:  python vypravleno.py
         python vypravleno.py --protsesiv 8 --zamovlen 100000 --progoniv 10
"""

import argparse
import multiprocessing as mp
import os
import platform
import random
import sys
import time

ZAPAS_POCHATKOVYI = 30_000
PROTSESIV = 4
ZAMOVLEN_NA_PROTSES = 20_000
MAX_U_ZAMOVLENNI = 3
PROGONIV = 10


def obrobyty_zamovlennya(zalyshok, vidvantazheno, vidmov, nasinnya, zamok, zamovlen):
    """Один робітник складу. Обробляє свою пачку замовлень."""
    rnd = random.Random(nasinnya)
    moi = 0

    for _ in range(zamovlen):
        kilkist = rnd.randint(1, MAX_U_ZAMOVLENNI)

        # Критична секція: перевірка і зміна всіх трьох лічильників
        # мають виконатись як одна неподільна дія.
        with zamok:
            if zalyshok.value >= kilkist:
                zalyshok.value = zalyshok.value - kilkist
                vidvantazheno.value = vidvantazheno.value + kilkist
                moi += kilkist
            else:
                vidmov.value = vidmov.value + 1

    print("    процес", nasinnya, "відвантажив", moi)


def pasport(protsesiv, zamovlen, progoniv):
    print("=" * 68)
    print("УМОВИ ЗАПУСКУ")
    print("=" * 68)
    print("  Python        :", sys.version.split()[0])
    print("  процесор      :", platform.processor())
    print("  логічних ядер :", os.cpu_count())
    print("  ОС            :", platform.platform())
    print("  навантаження  : %d процесів x %d замовлень, прогонів %d"
          % (protsesiv, zamovlen, progoniv))
    print()


def prognaty(nomer, protsesiv, zamovlen):
    zalyshok = mp.Value("i", ZAPAS_POCHATKOVYI)
    vidvantazheno = mp.Value("i", 0)
    vidmov = mp.Value("i", 0)
    zamok = mp.Lock()

    procesy = [
        mp.Process(target=obrobyty_zamovlennya,
                   args=(zalyshok, vidvantazheno, vidmov, 1000 + i, zamok, zamovlen))
        for i in range(protsesiv)
    ]

    t0 = time.perf_counter()
    for p in procesy:
        p.start()
    for p in procesy:
        p.join()
    tryvalist = time.perf_counter() - t0

    z = zalyshok.value
    v = vidvantazheno.value
    vm = vidmov.value

    # ── перевірка інваріантів ──────────────────────────────────
    porushennya = []
    if z < 0:
        porushennya.append("залишок від'ємний: %d" % z)
    if v + z != ZAPAS_POCHATKOVYI:
        porushennya.append("баланс не сходиться: %d + %d = %d, а має бути %d"
                           % (v, z, v + z, ZAPAS_POCHATKOVYI))

    stan = "OK" if not porushennya else "ПОРУШЕНО"
    print("  прогін %d:  залишок %8d   відвантажено %8d   відмов %6d   "
          "%6.2f c   %s" % (nomer, z, v, vm, tryvalist, stan))
    for p in porushennya:
        print("             -> " + p)

    return not porushennya, tryvalist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protsesiv", type=int, default=PROTSESIV)
    ap.add_argument("--zamovlen", type=int, default=ZAMOVLEN_NA_PROTSES)
    ap.add_argument("--progoniv", type=int, default=PROGONIV)
    a = ap.parse_args()

    pasport(a.protsesiv, a.zamovlen, a.progoniv)
    print("=" * 68)
    print("ПРОГОНИ.  Початковий запас:", ZAPAS_POCHATKOVYI)
    print("=" * 68)

    chysto = 0
    chasy = []
    for i in range(1, a.progoniv + 1):
        ok, t = prognaty(i, a.protsesiv, a.zamovlen)
        chysto += ok
        chasy.append(t)

    print()
    print("=" * 68)
    print("  Без порушень: %d із %d" % (chysto, a.progoniv))
    print("  Час прогону: мінімум %.2f c, середнє %.2f c" % (min(chasy), sum(chasy) / len(chasy)))
    print("=" * 68)
