"""Прискорення з лінією ідеалу + ефективність. Бере zamiry.json з protsesy.py."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TUT = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(TUT, "zamiry.json"), encoding="utf-8") as f:
    d = json.load(f)

tochky = d["tochky"][1:]
n = [t["procesiv"] for t in tochky]
s = [d["posl_min"] / min(z["razom"] for z in t["zamiry"]) for t in tochky]
e = [100 * si / ni for si, ni in zip(s, n)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))

a1.plot(n, n, "--", color="0.6", label="ідеал ×N")
a1.plot(n, s, "o-", color="#c0392b", linewidth=2, label="виміряно")
for x, y in zip(n, s):
    a1.annotate("×%.2f" % y, (x, y), xytext=(5, -12), textcoords="offset points", fontsize=8)
a1.axvline(d["fizychnykh"], color="0.8", linewidth=1)
a1.text(d["fizychnykh"], 1, " фізичних ядер", fontsize=8, color="0.4")
a1.set_title("Прискорення")
a1.set_xlabel("процесів")
a1.set_ylabel("×")

a2.plot(n, e, "s-", color="#1f6fb2", linewidth=2)
for x, y in zip(n, e):
    a2.annotate("%.0f %%" % y, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
a2.axhline(100, linestyle="--", color="0.6")
a2.set_title("Ефективність S/N")
a2.set_xlabel("процесів")
a2.set_ylabel("%")
a2.set_ylim(0, 110)

for a in (a1, a2):
    a.set_xscale("log", base=2)
    a.set_xticks(n)
    a.set_xticklabels([str(x) for x in n])
    a.grid(alpha=0.3)
a1.legend()

fig.suptitle("Мандельброт 600×400, ProcessPoolExecutor, мінімум з 3 прогонів")
fig.tight_layout()
fig.savefig(os.path.join(TUT, "grafik.png"), dpi=120)
print("Збережено grafik.png")
