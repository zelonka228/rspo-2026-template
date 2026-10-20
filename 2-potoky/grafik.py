"""
РСПО, практична 3. Один графік, дві криві: обчислення й очікування.

Бере готові заміри з zamiry-obchyslennya.json і zamiry-ochikuvannya.json
(їх пишуть potoky.py та ochikuvannya.py) і малює прискорення відносно
послідовного еталона ТОГО САМОГО сеансу.

  X — кількість потоків: 1 (послідовний еталон), 2, 4, 8, cpu_count();
  Y — прискорення = час_еталона / час_з_N_потоками, за мінімумом з 3 прогонів.

Сіра пунктирна лінія ×N — не замір, а орієнтир «ідеальна масштабованість».
Тонка лінія ×1 — рівень «потоки не дали нічого».

ЗАПУСК:  python grafik.py            -> grafik.png (дві криві)
         python grafik.py --numpy    -> grafik-numpy.png (⭐ те саме на NumPy)
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TUT = os.path.dirname(os.path.abspath(__file__))


def chytaty(fail):
    shlyakh = os.path.join(TUT, fail)
    if not os.path.exists(shlyakh):
        raise SystemExit("Немає %s — спочатку запустіть відповідний замір." % fail)
    with open(shlyakh, encoding="utf-8") as f:
        return json.load(f)


def kryva(dani):
    """(кількості потоків, прискорення) з json-файлу заміру."""
    x = [t["potoky"] for t in dani["tochky"]]
    y = [dani["posl_min"] / min(z["razom"] for z in t["zamiry"]) for t in dani["tochky"]]
    return x, y


def malyuvaty(kryvi, zagolovok, pidpys, fail):
    x_vsi = sorted({n for x, _, _ in kryvi for n in x})
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(x_vsi, x_vsi, color="0.6", linestyle="--", linewidth=1,
            label="ідеал ×N (орієнтир, не замір)")
    ax.axhline(1.0, color="0.75", linewidth=1)

    styli = [dict(color="#c0392b", marker="o"),
             dict(color="#1f6fb2", marker="s"),
             dict(color="#2e8b57", marker="^")]
    for (x, y, nazva), st in zip(kryvi, styli):
        ax.plot(x, y, marker=st["marker"], color=st["color"], linewidth=2,
                markersize=7, label=nazva)
        for xi, yi in zip(x, y):
            ax.annotate("×%.2f" % yi, (xi, yi), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, color=st["color"])

    ax.set_xscale("log", base=2)
    ax.set_xticks(x_vsi)
    ax.set_xticklabels([str(n) for n in x_vsi])
    ax.set_xlabel("кількість потоків (1 — послідовний еталон)")
    ax.set_ylabel("прискорення відносно послідовної версії, ×")
    ax.set_title(zagolovok)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left")
    fig.text(0.5, 0.01, pidpys, ha="center", fontsize=8, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    shlyakh = os.path.join(TUT, fail)
    fig.savefig(shlyakh, dpi=130)
    print("Збережено:", shlyakh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--numpy", action="store_true",
                    help="⭐ друга картинка: NumPy-версія проти чистого Python")
    a = ap.parse_args()

    obch = chytaty("zamiry-obchyslennya.json")
    ochik = chytaty("zamiry-ochikuvannya.json")

    if not a.numpy:
        x1, y1 = kryva(obch)
        x2, y2 = kryva(ochik)
        pidpys = ("Мандельброт %d×%d, max_iter=%d (мінімум з %d прогонів) проти "
                  "%d × time.sleep(%.1f);  %d логічних ядер, Python 3.12 CPython"
                  % (obch["shyryna"], obch["vysota"], obch["max_iter"], obch["progoniv"],
                     ochik["zavdan"], ochik["zatrymka"], obch["cpu_count"]))
        malyuvaty([(x1, y1, "обчислення (Мандельброт, чистий Python)"),
                   (x2, y2, "очікування (32 × time.sleep(0.5))")],
                  "Потоки в CPython: та сама схема, дві різні задачі",
                  pidpys, "grafik.png")
    else:
        np_dani = chytaty("zamiry-numpy.json")
        x1, y1 = kryva(obch)
        x3, y3 = kryva(np_dani)
        pidpys = ("NumPy %d×%d, max_iter=%d проти чистого Python %d×%d; "
                  "мінімум з %d прогонів, %d логічних ядер"
                  % (np_dani["shyryna"], np_dani["vysota"], np_dani["max_iter"],
                     obch["shyryna"], obch["vysota"], np_dani["progoniv"],
                     np_dani["cpu_count"]))
        malyuvaty([(x1, y1, "чистий Python (GIL тримається весь час)"),
                   (x3, y3, "NumPy (GIL відпускається на час ufunc)")],
                  "Додатково: та сама задача на NumPy — потоки починають працювати",
                  pidpys, "grafik-numpy.png")
