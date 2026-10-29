"""
РСПО, практична 5, завдання із зірочкою.

  python zirky.py protsesy   ⭐1. Оригінальний zlamane.py при 2, 4 і 8 процесах.
  python zirky.py potoky     Та сама логіка на потоках, зі стандартним
                             інтервалом перемикання й зі setswitchinterval(1e-6).

Другий режим пояснює, чому гонку не видно одразу: з потоками вона
проявляється рідко й не на кожній машині. Оригінальний zlamane.py не змінюється,
його функції беруться імпортом.
"""

import multiprocessing as mp
import random
import sys
import threading
import time

import zlamane as z


def prognin_protsesiv(n):
    """Один прогін zlamane.obrobyty_zamovlennya на n процесах."""
    zalyshok = mp.Value("i", z.ZAPAS_POCHATKOVYI)
    vidvantazheno = mp.Value("i", 0)
    vidmov = mp.Value("i", 0)
    procesy = [mp.Process(target=z.obrobyty_zamovlennya,
                          args=(zalyshok, vidvantazheno, vidmov, 1000 + i))
               for i in range(n)]
    t0 = time.perf_counter()
    for p in procesy:
        p.start()
    for p in procesy:
        p.join()
    return (zalyshok.value, vidvantazheno.value, time.perf_counter() - t0)


def protsesy():
    print("Оригінальний zlamane.py, %d замовлень на процес, 10 прогонів на конфігурацію"
          % z.ZAMOVLEN_NA_PROTSES)
    print()
    print("| Процесів | Прогонів з порушенням | Від'ємний залишок | Баланс не сходиться "
          "| Мін. залишок | Середній час, c |")
    print("|---|---|---|---|---|---|")
    for n in (2, 4, 8):
        porusheno = vid_emnyi = balans = 0
        zalyshky, chasy = [], []
        for _ in range(10):
            zal, vidv, t = prognin_protsesiv(n)
            ploho_1 = zal < 0
            ploho_2 = vidv + zal != z.ZAPAS_POCHATKOVYI
            vid_emnyi += ploho_1
            balans += ploho_2
            porusheno += ploho_1 or ploho_2
            zalyshky.append(zal)
            chasy.append(t)
        print("| %d | %d із 10 | %d із 10 | %d із 10 | %d | %.2f |"
              % (n, porusheno, vid_emnyi, balans, min(zalyshky), sum(chasy) / len(chasy)))


class Sklad:
    """Те саме, що mp.Value, але без жодних замків."""
    def __init__(self, v):
        self.value = v


def robitnyk_potokiv(zalyshok, vidvantazheno, vidmov, nasinnya, zamovlen):
    rnd = random.Random(nasinnya)
    for _ in range(zamovlen):
        kilkist = rnd.randint(1, z.MAX_U_ZAMOVLENNI)
        if zalyshok.value >= kilkist:
            zalyshok.value = zalyshok.value - kilkist
            vidvantazheno.value = vidvantazheno.value + kilkist
        else:
            vidmov.value = vidmov.value + 1


def potoky():
    zamovlen = z.ZAMOVLEN_NA_PROTSES
    print("Та сама логіка на 4 потоках, %d замовлень на потік, 10 прогонів" % zamovlen)
    print()
    print("| sys.setswitchinterval | Прогонів з порушенням | Від'ємний залишок | "
          "Баланс не сходиться | Середній час, c |")
    print("|---|---|---|---|---|")
    for interval in (sys.getswitchinterval(), 1e-6):
        sys.setswitchinterval(interval)
        porusheno = vid_emnyi = balans = 0
        chasy = []
        for _ in range(10):
            zal, vidv, vm = Sklad(z.ZAPAS_POCHATKOVYI), Sklad(0), Sklad(0)
            potoky_ = [threading.Thread(target=robitnyk_potokiv,
                                        args=(zal, vidv, vm, 1000 + i, zamovlen))
                       for i in range(4)]
            t0 = time.perf_counter()
            for p in potoky_:
                p.start()
            for p in potoky_:
                p.join()
            chasy.append(time.perf_counter() - t0)
            ploho_1 = zal.value < 0
            ploho_2 = vidv.value + zal.value != z.ZAPAS_POCHATKOVYI
            vid_emnyi += ploho_1
            balans += ploho_2
            porusheno += ploho_1 or ploho_2
        print("| %g | %d із 10 | %d із 10 | %d із 10 | %.2f |"
              % (interval, porusheno, vid_emnyi, balans, sum(chasy) / len(chasy)))


if __name__ == "__main__":
    rezhym = sys.argv[1] if len(sys.argv) > 1 else ""
    if rezhym == "protsesy":
        protsesy()
    elif rezhym == "potoky":
        potoky()
    else:
        print(__doc__)
