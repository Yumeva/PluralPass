from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from figures.style import PALETTE, save_publication_figure, set_nature_style

PITCH = "#06C99B"
PITCH_LINE = "white"
TEAM = "#2F6FB0"
OPP = "#E84A5F"
NEUTRAL = "#9AA4B2"
TEAL = "#4C9F9A"
ORANGE = "#D55E00"
GOLD = "#D9A441"
DARK = "#202124"
LIGHT = "#F6F7F3"


def panel_label(ax, label: str, x: float = -0.04, y: float = 1.02) -> None:
    ax.text(x, y, label, transform=ax.transAxes, weight="bold", fontsize=9, va="bottom")


def arrow(ax, start, end, color=DARK, lw=1.2, ms=11, style="-|>", rad=0.0, alpha=1.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=ms,
            lw=lw,
            color=color,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def rounded(ax, xy, w, h, text="", fc="white", ec="#CCD3DD", lw=0.9, r=0.02, fontsize=7, weight="normal", color=DARK):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    if text:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight, color=color)
    return patch


def draw_pitch(ax, x: float, y: float, w: float, h: float, alpha: float = 1.0) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor=PITCH, edgecolor=PITCH_LINE, lw=0.75, alpha=alpha))
    ax.add_patch(Rectangle((x + 0.05 * w, y + 0.07 * h), 0.9 * w, 0.86 * h, fill=False, edgecolor=PITCH_LINE, lw=0.55, alpha=0.85))
    ax.plot([x + 0.5 * w, x + 0.5 * w], [y + 0.07 * h, y + 0.93 * h], color=PITCH_LINE, lw=0.45, alpha=0.75)
    ax.add_patch(Circle((x + 0.5 * w, y + 0.5 * h), 0.09 * h, fill=False, edgecolor=PITCH_LINE, lw=0.45, alpha=0.75))
    ax.add_patch(Rectangle((x + 0.05 * w, y + 0.34 * h), 0.15 * w, 0.32 * h, fill=False, edgecolor=PITCH_LINE, lw=0.55, alpha=0.85))
    ax.add_patch(Rectangle((x + 0.80 * w, y + 0.34 * h), 0.15 * w, 0.32 * h, fill=False, edgecolor=PITCH_LINE, lw=0.55, alpha=0.85))
    ax.add_patch(Rectangle((x + 0.025 * w, y + 0.43 * h), 0.025 * w, 0.14 * h, facecolor="#194DAD", edgecolor=PITCH_LINE, lw=0.4))
    ax.add_patch(Rectangle((x + 0.95 * w, y + 0.43 * h), 0.025 * w, 0.14 * h, facecolor="#194DAD", edgecolor=PITCH_LINE, lw=0.4))


def draw_player(ax, x, y, color, label=None, r=0.018, edge="white", z=5, weight="bold", txt_color="white") -> None:
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor=edge, lw=0.55, zorder=z))
    if label is not None:
        ax.text(x, y, label, ha="center", va="center", fontsize=5.5, color=txt_color, weight=weight, zorder=z + 1)


def draw_opportunity_state(ax, x, y, w, h, show_labels=True) -> dict[str, tuple[float, float]]:
    draw_pitch(ax, x, y, w, h)
    visible = np.array(
        [
            [x + 0.16 * w, y + 0.08 * h],
            [x + 0.98 * w, y + 0.15 * h],
            [x + 0.94 * w, y + 0.87 * h],
            [x + 0.26 * w, y + 0.92 * h],
            [x + 0.06 * w, y + 0.53 * h],
        ]
    )
    ax.add_patch(Polygon(visible, closed=True, facecolor="#E7F2EF", edgecolor=TEAL, lw=1.0, alpha=0.88, zorder=2))
    ax.text(x + 0.065 * w, y + 0.82 * h, "visible 360 area", color=TEAL, fontsize=6.4, zorder=8)
    ax.text(x + 0.04 * w, y + 0.08 * h, "unobserved\nspace", color="#51606F", fontsize=5.7, zorder=8)

    positions = {
        "passer": (x + 0.50 * w, y + 0.57 * h),
        "A": (x + 0.69 * w, y + 0.39 * h),
        "B": (x + 0.80 * w, y + 0.53 * h),
        "C": (x + 0.64 * w, y + 0.70 * h),
        "D": (x + 0.40 * w, y + 0.41 * h),
        "E": (x + 0.82 * w, y + 0.76 * h),
        "F": (x + 0.57 * w, y + 0.29 * h),
        "g1": (x + 0.58 * w, y + 0.47 * h),
        "g2": (x + 0.74 * w, y + 0.62 * h),
        "g3": (x + 0.34 * w, y + 0.64 * h),
        "g4": (x + 0.46 * w, y + 0.77 * h),
        "g5": (x + 0.37 * w, y + 0.29 * h),
    }
    for key in ["g1", "g2", "g3", "g4", "g5"]:
        draw_player(ax, *positions[key], OPP, r=0.016)
    for lab in ["A", "B", "C", "D", "E", "F"]:
        draw_player(ax, *positions[lab], TEAM, label=lab if show_labels else None, r=0.017)
    draw_player(ax, *positions["passer"], "#111111", label="P", r=0.023, txt_color="white")
    for lab, rad in [("A", 0.04), ("B", -0.02), ("C", -0.06), ("D", 0.10)]:
        arrow(ax, positions["passer"], positions[lab], color=ORANGE, lw=1.0, ms=8, rad=rad, alpha=0.95)
    return positions


def draw_graph(ax, cx, cy, scale=1.0, labels=True) -> dict[str, tuple[float, float]]:
    nodes = {
        "P": (cx - 0.17 * scale, cy - 0.02 * scale),
        "A": (cx + 0.02 * scale, cy + 0.16 * scale),
        "B": (cx + 0.22 * scale, cy + 0.07 * scale),
        "C": (cx + 0.14 * scale, cy - 0.18 * scale),
        "D": (cx - 0.12 * scale, cy - 0.22 * scale),
        "O1": (cx - 0.30 * scale, cy + 0.12 * scale),
        "O2": (cx + 0.31 * scale, cy - 0.14 * scale),
    }
    keys = list(nodes)
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            x1, y1 = nodes[k1]
            x2, y2 = nodes[k2]
            ax.plot([x1, x2], [y1, y2], color="#CCD2DA", lw=0.55, zorder=1)
    for k1, k2 in [("P", "A"), ("P", "B"), ("P", "C"), ("P", "D")]:
        x1, y1 = nodes[k1]
        x2, y2 = nodes[k2]
        ax.plot([x1, x2], [y1, y2], color=ORANGE, lw=1.0, zorder=2, alpha=0.9)
    for key, (xx, yy) in nodes.items():
        if key == "P":
            draw_player(ax, xx, yy, "#111111", label="P" if labels else None, r=0.025 * scale)
        elif key.startswith("O"):
            draw_player(ax, xx, yy, OPP, r=0.022 * scale)
        else:
            draw_player(ax, xx, yy, TEAM, label=key if labels else None, r=0.023 * scale)
    return nodes


def make_figure(output: str = "artifacts/figures/figure1_pluralpass_overview") -> None:
    set_nature_style()
    fig = plt.figure(figsize=(7.2, 5.35))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Contract:
    # Core conclusion: PluralPass converts a partially observed open-play pass into a masked opportunity graph,
    # then returns calibrated, plural recommendations instead of a single unqualified "best pass".
    # Archetype: schematic-led composite; output: Nature Communications full-width Fig. 1.

    # Panel a: representation and message passing.
    panel_label(ax, "a", x=0.02, y=0.95)
    ax.text(0.055, 0.952, "From a 360 freeze frame to a partially observed opportunity graph", fontsize=9, weight="bold", va="bottom")
    state_pos = draw_opportunity_state(ax, 0.055, 0.655, 0.245, 0.225)
    ax.text(0.177, 0.625, "visible teammates become candidate receivers;\noff-screen players are not assumed available", ha="center", va="top", fontsize=6.2, color="#4E5B66")

    graph_nodes = draw_graph(ax, 0.515, 0.765, scale=0.35)
    ax.text(0.515, 0.625, "role-aware nodes and relational edges\nencode pressure, passing lanes and context", ha="center", va="top", fontsize=6.2, color="#4E5B66")
    arrow(ax, (0.315, 0.765), (0.405, 0.765), lw=1.6, ms=14)
    ax.text(0.36, 0.792, "construct graph", ha="center", fontsize=7.4, style="italic")

    # Player-to-node dashed correspondences.
    for lab, node_key in [("passer", "P"), ("A", "A"), ("B", "B"), ("C", "C")]:
        x1, y1 = state_pos[lab]
        x2, y2 = graph_nodes[node_key]
        ax.plot([x1, x2], [y1, y2], color=TEAM if lab != "passer" else "#333333", lw=0.45, ls=(0, (2, 3)), alpha=0.55)

    # Message passing / representation update.
    ax.text(0.705, 0.792, "masked message passing", ha="center", fontsize=7.4, style="italic")
    arrow(ax, (0.620, 0.765), (0.700, 0.765), lw=1.6, ms=14)
    rounded(ax, (0.760, 0.688), 0.160, 0.154, fc="white", ec="#C7D0DB", lw=1.0, r=0.018)
    ax.text(0.840, 0.828, "candidate-aware\nembeddings", ha="center", va="top", fontsize=6.2, weight="bold", color=DARK)
    for i, (label, color, y0) in enumerate([("passer", "#111111", 0.765), ("receiver", TEAM, 0.727), ("opponent", OPP, 0.689)]):
        ax.add_patch(Circle((0.785, y0), 0.012, facecolor=color, edgecolor="white", lw=0.45))
        for j in range(4):
            ax.add_patch(Rectangle((0.808 + j * 0.016, y0 - 0.010), 0.009, 0.020 + 0.006 * ((i + j) % 2), facecolor="#9DB4D5", edgecolor="none", alpha=0.75))
        ax.text(0.872, y0, label, ha="left", va="center", fontsize=4.8, color="#4E5B66")
    ax.text(0.840, 0.625, "updated representations keep\ncandidate and visibility masks", ha="center", va="top", fontsize=6.2, color="#4E5B66")

    # Separator.
    ax.plot([0.045, 0.955], [0.585, 0.585], color="#D8DCE2", lw=0.8)

    # Panel b: full assistant pipeline.
    panel_label(ax, "b", x=0.02, y=0.548)
    ax.text(0.055, 0.55, "PluralPass returns calibrated options rather than a single unqualified pass", fontsize=9, weight="bold", va="bottom")

    # Left: the same state.
    draw_opportunity_state(ax, 0.055, 0.245, 0.205, 0.210)
    ax.text(0.157, 0.218, "open-play decision state", ha="center", fontsize=6.6, weight="bold")

    # Restricted symmetry / ensemble.
    arrow(ax, (0.270, 0.355), (0.332, 0.355), lw=1.4, ms=13)
    rounded(ax, (0.335, 0.412), 0.13, 0.052, "touchline\nreflection", fc="#EDF6F5", ec=TEAL, fontsize=6.2, weight="bold", color="#244C4A")
    rounded(ax, (0.335, 0.332), 0.13, 0.052, "original\nstate", fc="#F7F8FA", ec="#C6CBD4", fontsize=6.2, weight="bold")
    rounded(ax, (0.335, 0.252), 0.13, 0.052, "visible-area\nmask", fc="#EDF6F5", ec=TEAL, fontsize=6.2, weight="bold", color="#244C4A")
    ax.text(0.399, 0.205, "same attacking direction;\npartial observation explicit", ha="center", fontsize=5.8, color="#4E5B66")
    arrow(ax, (0.470, 0.355), (0.535, 0.355), lw=1.4, ms=13)

    # Ensemble block.
    rounded(ax, (0.545, 0.265), 0.105, 0.205, fc="#FFFFFF", ec="#9DB4D5", lw=1.0, r=0.018)
    ax.text(0.598, 0.445, "PluralPass\nensemble", ha="center", va="top", fontsize=7.3, weight="bold", color=PALETTE["navy"])
    for i in range(5):
        x0 = 0.565 + i * 0.012
        ax.add_patch(Rectangle((x0, 0.305 + i * 0.008), 0.046, 0.090, facecolor="#EAF0F7", edgecolor="#6E8EBB", lw=0.55, alpha=0.95))
        ax.plot([x0 + 0.010, x0 + 0.036], [0.365 + i * 0.008, 0.322 + i * 0.008], color="#6E8EBB", lw=0.45)
        ax.plot([x0 + 0.010, x0 + 0.036], [0.322 + i * 0.008, 0.365 + i * 0.008], color="#6E8EBB", lw=0.45)
    ax.text(0.598, 0.282, "epistemic\nuncertainty", ha="center", va="center", fontsize=5.1, color="#4E5B66")

    # Prediction heads.
    arrow(ax, (0.655, 0.355), (0.700, 0.355), lw=1.3, ms=12)
    heads = [
        (0.705, 0.418, "Receiver\nchoice", "#E8EEF6", PALETTE["blue"]),
        (0.705, 0.335, "Completion\nrisk", "#E6F2F0", TEAL),
        (0.705, 0.252, "Value\ndelta", "#F8F1DE", GOLD),
    ]
    for x0, y0, txt, fc, ec in heads:
        rounded(ax, (x0, y0), 0.105, 0.055, txt, fc=fc, ec=ec, fontsize=6.4, weight="bold")

    # Recommendation set.
    arrow(ax, (0.815, 0.355), (0.857, 0.355), lw=1.3, ms=12)
    rounded(ax, (0.850, 0.225), 0.135, 0.261, fc="#FFFFFF", ec=ORANGE, lw=1.0, r=0.018)
    ax.text(0.918, 0.474, "calibrated\noption set", ha="center", va="top", fontsize=6.6, weight="bold", color=ORANGE)
    ax.text(0.918, 0.414, "candidate   success   value", ha="center", va="center", fontsize=5.0, color="#66707E")
    option_rows = [
        ("A", "0.82", "+0.09", True),
        ("B", "0.75", "+0.12", True),
        ("C", "0.54", "+0.15", True),
        ("D", "0.69", "-0.01", False),
    ]
    for i, (lab, risk, value, in_set) in enumerate(option_rows):
        yy = 0.376 - i * 0.032
        fill = "#FFF3EA" if in_set else "#F1F3F5"
        ec = ORANGE if in_set else "#C7CDD6"
        rounded(ax, (0.868, yy), 0.100, 0.024, fc=fill, ec=ec, r=0.006)
        ax.text(0.884, yy + 0.012, lab, ha="center", va="center", fontsize=6.0, weight="bold", color=ORANGE if in_set else "#66707E")
        ax.text(0.921, yy + 0.012, risk, ha="center", va="center", fontsize=5.4)
        ax.text(0.954, yy + 0.012, value, ha="center", va="center", fontsize=5.4)
    ax.text(0.918, 0.247, "abstain if evidence is weak", ha="center", va="center", fontsize=5.2, color="#4E5B66")

    # Coach validation ribbon.
    rounded(ax, (0.335, 0.070), 0.185, 0.085, fc="#F5F7FA", ec="#C7D0DB", r=0.016)
    ax.text(0.427, 0.128, "external validation", ha="center", va="center", fontsize=7.0, weight="bold", color=PALETTE["navy"])
    ax.text(0.427, 0.096, "leave-one competition-season tests", ha="center", va="center", fontsize=5.8, color="#4E5B66")
    rounded(ax, (0.545, 0.070), 0.185, 0.085, fc="#F5F7FA", ec="#C7D0DB", r=0.016)
    ax.text(0.637, 0.128, "uncertainty check", ha="center", va="center", fontsize=7.0, weight="bold", color=PALETTE["navy"])
    ax.text(0.637, 0.096, "model entropy vs coach disagreement", ha="center", va="center", fontsize=5.8, color="#4E5B66")
    rounded(ax, (0.755, 0.070), 0.185, 0.085, fc="#F5F7FA", ec="#C7D0DB", r=0.016)
    ax.text(0.847, 0.128, "human-AI study", ha="center", va="center", fontsize=7.0, weight="bold", color=PALETTE["navy"])
    ax.text(0.847, 0.096, "sets compared with point advice", ha="center", va="center", fontsize=5.8, color="#4E5B66")
    for sx in [0.918, 0.815, 0.598]:
        arrow(ax, (sx, 0.255), (sx - 0.03 if sx > 0.75 else sx + 0.03, 0.160), color="#8793A1", lw=0.8, ms=8, rad=0.15, alpha=0.75)

    save_publication_figure(fig, output)
    save_publication_figure(fig, "outputs/PluralPass_Figure1")
    # Preserve the historical Nature-style 600 dpi TIFF filename used by the manuscript package.
    fig.savefig("outputs/PluralPass_Figure1_600dpi.tiff", dpi=600, bbox_inches="tight")
    fig.savefig("outputs/PluralPass_Figure1_preview.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
