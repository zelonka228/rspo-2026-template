"""
РСПО, практична 2. Профіль послідовної версії через cProfile.

Один прогін того самого розміру, що й у mandelbrot.py (обчислення + запис).
Звіт — у profil.txt поруч, відсортований за власним часом функції (tottime).

⚠️ Під cProfile програма йде в рази повільніше: профайлер обробляє
кожен виклик функції, а iteratsii() викликається для кожного пікселя.
Абсолютні числа звідси не беремо — тільки розподіл часу між функціями.

ЗАПУСК:  python profil.py
"""

import cProfile
import io
import os
import pstats

import mandelbrot as m

ZVIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profil.txt")


def odyn_progin():
    rezultat = m.obchyslyty(m.SHYRYNA, m.VYSOTA, m.MAX_ITER)
    m.zapysaty_pgm(rezultat, m.MAX_ITER, m.FAIL)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.runcall(odyn_progin)

    buf = io.StringIO()
    st = pstats.Stats(profiler, stream=buf)
    st.strip_dirs()
    buf.write("Розмір: %d×%d, max_iter=%d\n\n" % (m.SHYRYNA, m.VYSOTA, m.MAX_ITER))
    buf.write("=== за власним часом (tottime) ===\n")
    st.sort_stats("tottime").print_stats(12)
    buf.write("=== за накопиченим часом (cumtime) ===\n")
    st.sort_stats("cumulative").print_stats(12)

    with open(ZVIT, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(buf.getvalue())
    print("Збережено:", ZVIT)
