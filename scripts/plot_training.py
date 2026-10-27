"""Figure des courbes d'entraînement (val mAP50-95 et LR par époque) pour docs/RESULTS.md.

Deux panneaux côte à côte (une seule échelle par panneau, pas de double axe) :
- gauche : mAP50-95 sur le val par époque, meilleure époque marquée ;
- droite : learning rate du groupe principal de poids (échelle log).

Groupe « principal » : avec MuSGD (optimizer=auto, R1), Ultralytics crée d'abord un groupe
« tête de classification » à LR ×3 (`lr/pg0`), puis les autres poids (`lr/pg1`) ; avec SGD (R2)
tous les groupes ont le même LR (`lr/pg0`). Le script choisit la colonne adaptée d'après
`args.yaml` (optimizer) et le log Ultralytics (`MuSGD` si optimizer=auto).

Usage (depuis la racine du dépôt) :
    .venv/bin/python scripts/plot_training.py \
        --runs r1_y11n_640:"R1 · 640 (COCO)" r2_y11n_1280_ft:"R2 · 1280 (init. R1)" \
        --out docs/assets/training_curves.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Palette de référence (skill dataviz, mode clair) : séries catégorielles dans l'ordre fixe.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, TEXT2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def load(run: str) -> tuple[pd.DataFrame, str]:
    """results.csv d'un run + nom de la colonne de LR principal."""
    d = ROOT / "runs" / "train" / run
    df = pd.read_csv(d / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    args = (d / "args.yaml").read_text() if (d / "args.yaml").exists() else ""
    log = ROOT / "runs" / "logs" / f"{run}.log"
    musgd = "optimizer: MuSGD" in args or ("optimizer: auto" in args and log.exists()
                                             and "MuSGD(" in log.read_text(errors="ignore"))
    return df, ("lr/pg1" if musgd else "lr/pg0")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run[:libellé] (dossiers de runs/train)")
    ap.add_argument("--out", default="docs/assets/training_curves.png")
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": TEXT2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
                         "axes.titlecolor": TEXT, "axes.titlesize": 11, "axes.titleweight": "bold"})
    fig, (ax_m, ax_lr) = plt.subplots(1, 2, figsize=(11, 3.9), dpi=110, facecolor=SURFACE)
    for ax in (ax_m, ax_lr):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=1, linestyle="-")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("époque (dans le run)")

    for i, spec in enumerate(args.runs):
        run, _, label = spec.partition(":")
        label = label or run
        df, lr_col = load(run)
        c = SERIES[i]
        ep, m = df["epoch"], df["metrics/mAP50-95(B)"]
        b = int(m.idxmax())  # critère de sélection Ultralytics 8.4 (fitness = mAP50-95)
        ax_m.plot(ep, m, color=c, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                  label=f"{label} — meilleure ép. {int(ep[b])} : {m[b]:.3f}")
        ax_m.plot(ep[b], m[b], "o", markersize=8, color=c, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax_lr.plot(ep, df[lr_col], color=c, linewidth=2, solid_capstyle="round", label=label)
        ax_lr.annotate(label, (ep.iloc[-1], df[lr_col].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                       color=TEXT2, fontsize=9, va="center")

    ax_m.set_title("mAP50-95 sur le val", loc="left")
    ax_m.set_ylabel("mAP50-95")
    ax_m.legend(frameon=False, loc="lower right", labelcolor=TEXT2, title="● = meilleure époque (poids retenus)",
                title_fontsize=9)
    ax_lr.set_title("Learning rate (poids principaux)", loc="left")
    ax_lr.set_yscale("log")
    ax_lr.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax_lr.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_lr.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_lr.set_ylabel("LR (échelle log)")
    ax_lr.margins(x=0.18)
    fig.tight_layout()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"→ {out.relative_to(ROOT)} ({out.stat().st_size // 1024} Ko)")


if __name__ == "__main__":
    main()
