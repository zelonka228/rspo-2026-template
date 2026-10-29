"""
РСПО, практична 5, ⭐4. Взаємне блокування двох складів і його усунення.

Два склади, у кожного свій замок. Процес А переказує товар зі складу 1 на
склад 2: бере замок 1, а потім замок 2. Процес Б робить навпаки: бере замок 2,
а потім замок 1. Якщо вони одночасно взяли по одному замку, кожен чекає
замок, який тримає інший, і програма стоїть вічно.

  python vzaiemne_blokuvannya.py zavysaye   А бере замки в порядку 1->2, Б у порядку 2->1
  python vzaiemne_blokuvannya.py vypravleno обидва беруть замки в порядку 1->2

У режимі `zavysaye` через TAJMAUT секунд головний процес виводить стеки
процесів (faulthandler) і завершує їх, тож видно, на якому рядку стоїть
кожен. У режимі `vypravleno` порядок захоплення однаковий, і програма
закінчується, а сума товару на двох складах лишається 2000.
"""

import faulthandler
import multiprocessing as mp
import sys
import time

PEREKAZIV = 2000
TAJMAUT = 5


def pereklad(zamok_z, zamok_do, sklad_z, sklad_do, kilkist, perekaziv, imya):
    # Через TAJMAUT секунд кожен процес-робітник друкує стек своїх потоків.
    faulthandler.dump_traceback_later(TAJMAUT, exit=False)
    for _ in range(perekaziv):
        with zamok_z:                      # беремо перший замок
            time.sleep(0)                  # віддаємо процесор, щоб інший встиг взяти свій
            with zamok_do:                 # беремо другий замок
                sklad_z.value -= kilkist
                sklad_do.value += kilkist
    faulthandler.cancel_dump_traceback_later()
    print("  %s закінчив" % imya, flush=True)


def prognaty(rezhym):
    zamok1, zamok2 = mp.Lock(), mp.Lock()
    sklad1, sklad2 = mp.Value("i", 1000, lock=False), mp.Value("i", 1000, lock=False)

    if rezhym == "zavysaye":
        # А: 1 -> 2, Б: 2 -> 1  (замки в протилежному порядку)
        a_zamky, b_zamky = (zamok1, zamok2), (zamok2, zamok1)
    else:
        # обидва беруть спершу замок 1, потім замок 2
        a_zamky, b_zamky = (zamok1, zamok2), (zamok1, zamok2)

    a = mp.Process(target=pereklad, args=(*a_zamky, sklad1, sklad2, 1, PEREKAZIV, "А"))
    b = mp.Process(target=pereklad, args=(*b_zamky, sklad2, sklad1, 1, PEREKAZIV, "Б"))

    t0 = time.perf_counter()
    a.start()
    b.start()
    a.join(TAJMAUT + 2)
    b.join(TAJMAUT + 2)
    t = time.perf_counter() - t0

    zavyslo = a.is_alive() or b.is_alive()
    if zavyslo:
        print("  ЗАВИСЛО: після %.1f c процеси досі живі" % t)
        for p in (a, b):
            if p.is_alive():
                p.terminate()
    else:
        print("  завершено за %.2f c, сума на складах %d (має бути 2000)"
              % (t, sklad1.value + sklad2.value))
    return zavyslo


if __name__ == "__main__":
    rezhym = sys.argv[1] if len(sys.argv) > 1 else ""
    if rezhym not in ("zavysaye", "vypravleno"):
        print(__doc__)
        raise SystemExit
    print("Режим:", rezhym)
    prognaty(rezhym)
