from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


GAUSSIAN_INFO_PATTERN = re.compile(
    r"window=\s*(\d+)x\d+\s*\|.*?gaussian_info=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

# Short legend names. Use full names in caption/table if needed.
SCHEDULES: dict[str, list[int]] = {
    "C2F Energy (Ours)": [78, 186, 268, 318, 362, 374, 382, 384],
    "F2C Energy": [18, 20, 22, 24, 26, 34, 76, 384],
    "Uniform Energy": [24, 34, 50, 78, 122, 186, 266, 384],
    "Uniform Width": [48, 96, 144, 192, 240, 288, 336, 384],
    "just play": [54, 140, 240, 306, 352, 370, 380, 384],
}

# Mostly grayscale, distinguished by marker + line style.
STYLE: dict[str, dict] = {
    "C2F Energy (Ours)": dict(
        color="0.00", linestyle="-", marker="o", linewidth=1.50,
        markersize=3.6, markerfacecolor="0.00", markeredgewidth=0.65, zorder=5,
    ),
    "F2C Energy": dict(
        color="0.38", linestyle="--", marker="s", linewidth=1.10,
        markersize=3.2, markerfacecolor="white", markeredgewidth=0.65, zorder=4,
    ),
    "Uniform Energy": dict(
        color="0.18", linestyle="-.", marker="^", linewidth=1.10,
        markersize=3.3, markerfacecolor="white", markeredgewidth=0.65, zorder=3,
    ),
    "Uniform Width": dict(
        color="0.58", linestyle=":", marker="D", linewidth=1.15,
        markersize=3.0, markerfacecolor="white", markeredgewidth=0.65, zorder=2,
    ),
    "just play": dict(
        color="0.76", linestyle=(0, (2.0, 2.2)), marker="x", linewidth=1.00,
        markersize=3.7, markerfacecolor="none", markeredgewidth=0.75, zorder=1,
    ),
}


def set_neurips_plot_style() -> None:
    """Compact, vector-friendly style for NeurIPS-width figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        "font.size": 7.0,
        "axes.labelsize": 7.2,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "legend.fontsize": 6.3,

        "axes.linewidth": 0.60,
        "xtick.major.width": 0.50,
        "ytick.major.width": 0.50,
        "xtick.major.size": 2.3,
        "ytick.major.size": 2.3,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "savefig.dpi": 600,
        "figure.dpi": 150,
    })


def parse_gaussian_info(path: Path) -> dict[int, float]:
    text = path.read_text(encoding="utf-8")
    pairs = GAUSSIAN_INFO_PATTERN.findall(text)
    if not pairs:
        raise ValueError(f"No gaussian_info entries found in {path}")
    return {int(w): float(v) for w, v in pairs}


def build_cumulative_energy(
    info_by_window: dict[int, float],
    acs_size: int,
    full_size: int,
) -> dict[int, float]:
    """
    Build the same raw-information curve used by the scheduler:
    sort even windows >= ACS, enforce monotonicity, then normalize
    E(ACS)=0 and E(full)=1.
    """
    windows = np.array(
        [w for w in sorted(info_by_window) if acs_size <= w <= full_size and w % 2 == 0],
        dtype=np.int64,
    )

    if len(windows) < 2:
        raise ValueError("Need at least two valid even windows between ACS and full size.")
    if windows[0] != acs_size:
        raise ValueError(f"ACS window {acs_size} is missing from the gaussian_info file.")
    if windows[-1] != full_size:
        raise ValueError(f"Full window {full_size} is missing from the gaussian_info file.")

    infos = np.array([info_by_window[int(w)] for w in windows], dtype=np.float64)
    infos = np.maximum.accumulate(infos)

    increments = np.clip(np.diff(infos), a_min=0.0, a_max=None)
    cumulative = np.insert(np.cumsum(increments), 0, 0.0)

    if cumulative[-1] <= 0:
        raise ValueError("Information curve has zero total increment.")

    cumulative = cumulative / cumulative[-1]
    return {int(w): float(e) for w, e in zip(windows, cumulative)}


def build_ambiguity_proxy(
    widths: dict[str, np.ndarray],
    acs_size: int,
    mode: str = "square",
    scale: float = 1e3,
) -> dict[str, np.ndarray]:
    """
    Ambiguity proxy for the null-space size.

    mode = "square":
        proxy_t = w_t^2 - acs_size^2
        Interpretable as the number of newly admitted unknown pixels
        in a square-window approximation.

    mode = "line":
        proxy_t = w_t - acs_size
        Simpler 1D line-wise approximation.

    Returned values are divided by `scale` for cleaner plotting.
    """
    if mode not in {"square", "line"}:
        raise ValueError("ambiguity mode must be either 'square' or 'line'.")

    ambiguity: dict[str, np.ndarray] = {}

    for name, w in widths.items():
        w = np.asarray(w, dtype=np.float64)
        if mode == "square":
            a = np.maximum(w**2 - float(acs_size)**2, 0.0)
        else:
            a = np.maximum(w - float(acs_size), 0.0)

        ambiguity[name] = a / scale

    return ambiguity


def collect_schedule_data(
    energy_by_window: dict[int, float],
    acs_size: int,
    ambiguity_mode: str = "square",
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    stages = np.arange(1, 9)

    widths: dict[str, np.ndarray] = {}
    cumulative_energy: dict[str, np.ndarray] = {}
    stage_energy: dict[str, np.ndarray] = {}

    for name, schedule in SCHEDULES.items():
        w = np.asarray(schedule, dtype=np.int64)

        missing = [int(x) for x in w if int(x) not in energy_by_window]
        if missing:
            raise ValueError(f"Schedule '{name}' uses windows missing from the information curve: {missing}")

        e_cum = np.asarray([energy_by_window[int(x)] for x in w], dtype=np.float64)
        e_stage = np.diff(np.r_[0.0, e_cum])

        widths[name] = w
        cumulative_energy[name] = e_cum
        stage_energy[name] = e_stage

    ambiguity = build_ambiguity_proxy(
        widths=widths,
        acs_size=acs_size,
        mode=ambiguity_mode,
        scale=1e3,
    )

    return stages, widths, cumulative_energy, stage_energy, ambiguity


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", pad=1.4)
    ax.grid(axis="y", color="0.92", linewidth=0.45)
    ax.grid(axis="x", visible=False)


def plot_lines(ax: plt.Axes, stages: np.ndarray, y_by_name: dict[str, np.ndarray]) -> None:
    for name, y in y_by_name.items():
        ax.plot(stages, y, label=name, **STYLE[name])


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015, 0.965, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=7.2,
        fontweight="bold",
    )


def save_all(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.012)
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.012)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.012, dpi=600)


def make_three_panel_figure(
    stages: np.ndarray,
    widths: dict[str, np.ndarray],
    cumulative_energy: dict[str, np.ndarray],
    stage_energy: dict[str, np.ndarray],
    ambiguity: dict[str, np.ndarray],
    out_dir: Path,
    energy_mode: str,
    ambiguity_mode: str,
) -> None:
    if energy_mode not in {"stage", "cumulative"}:
        raise ValueError("energy_mode must be either 'stage' or 'cumulative'.")

    # Full-width NeurIPS figure (5.5 in wide), low-height minimalist layout.
    fig, axes = plt.subplots(1, 3, figsize=(5.50, 1.95), sharex=True)
    ax0, ax1, ax2 = axes

    # Panel (a): window / mask trajectory.
    plot_lines(ax0, stages, widths)
    clean_axis(ax0)
    add_panel_label(ax0, "(a)")
    ax0.set_xlabel(r"Stage $t$")
    ax0.set_ylabel(r"Window width $w_t$")
    ax0.set_xlim(1, 8)
    ax0.set_xticks(stages)
    ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax0.set_ylim(0, 400)
    ax0.set_yticks([0, 96, 192, 288, 384])

    # Panel (b): energy trajectory.
    if energy_mode == "stage":
        y_energy = {k: 100.0 * v for k, v in stage_energy.items()}
        ylabel = "Allocated energy (%)"
        ylim = (-1, 54)
        yticks = [0, 10, 20, 30, 40, 50]
        out_name = "stanford_schedule_3panel_stage_energy"
    else:
        y_energy = {k: 100.0 * v for k, v in cumulative_energy.items()}
        ylabel = "Cumulative energy (%)"
        ylim = (-2, 103)
        yticks = [0, 25, 50, 75, 100]
        out_name = "stanford_schedule_3panel_cumulative_energy"

    plot_lines(ax1, stages, y_energy)
    clean_axis(ax1)
    add_panel_label(ax1, "(b)")
    ax1.set_xlabel(r"Stage $t$")
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(1, 8)
    ax1.set_xticks(stages)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylim(*ylim)
    ax1.set_yticks(yticks)

    # Panel (c): ambiguity proxy.
    plot_lines(ax2, stages, ambiguity)
    clean_axis(ax2)
    add_panel_label(ax2, "(c)")
    ax2.set_xlabel(r"Stage $t$")
    if ambiguity_mode == "square":
        ax2.set_ylabel(r"Ambiguity proxy ($\times 10^3$ px)")
    else:
        ax2.set_ylabel(r"Ambiguity proxy ($\times 10^3$ lines)")
    ax2.set_xlim(1, 8)
    ax2.set_xticks(stages)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Let matplotlib choose the y ticks automatically but keep them sparse.
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # Shared legend. 5 entries => 3 columns, two rows is cleaner than cramming one row.
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
        handlelength=1.45,
        handletextpad=0.38,
        columnspacing=0.90,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=0.064,
        right=0.995,
        bottom=0.235,
        top=0.74,
        wspace=0.36,
    )

    suffix = f"_{ambiguity_mode}_ambiguity"
    save_all(fig, out_dir / f"{out_name}{suffix}")
    plt.close(fig)


def make_single_panel_figures(
    stages: np.ndarray,
    widths: dict[str, np.ndarray],
    cumulative_energy: dict[str, np.ndarray],
    stage_energy: dict[str, np.ndarray],
    ambiguity: dict[str, np.ndarray],
    out_dir: Path,
    energy_mode: str,
    ambiguity_mode: str,
) -> None:
    def _single(
        y_by_name: dict[str, np.ndarray],
        ylabel: str,
        ylim: tuple[float, float] | None,
        yticks: list[int] | None,
        out_name: str,
    ) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(2.62, 1.88))

        plot_lines(ax, stages, y_by_name)
        clean_axis(ax)

        ax.set_xlabel(r"Stage $t$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(1, 8)
        ax.set_xticks(stages)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if ylim is not None:
            ax.set_ylim(*ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.24),
            ncol=2,
            frameon=False,
            handlelength=1.45,
            handletextpad=0.38,
            columnspacing=0.70,
            borderaxespad=0.0,
        )

        fig.subplots_adjust(
            left=0.17,
            right=0.99,
            bottom=0.24,
            top=0.74,
        )

        save_all(fig, out_dir / out_name)
        plt.close(fig)

    _single(
        widths,
        r"Window width $w_t$",
        (0, 400),
        [0, 96, 192, 288, 384],
        "stanford_schedule_window_width",
    )

    if energy_mode == "stage":
        _single(
            {k: 100.0 * v for k, v in stage_energy.items()},
            "Allocated energy (%)",
            (-1, 54),
            [0, 10, 20, 30, 40, 50],
            "stanford_schedule_stage_energy",
        )
    else:
        _single(
            {k: 100.0 * v for k, v in cumulative_energy.items()},
            "Cumulative energy (%)",
            (-2, 103),
            [0, 25, 50, 75, 100],
            "stanford_schedule_cumulative_energy",
        )

    if ambiguity_mode == "square":
        ambiguity_ylabel = r"Ambiguity proxy ($\times 10^3$ px)"
        ambiguity_name = "stanford_schedule_ambiguity_square"
    else:
        ambiguity_ylabel = r"Ambiguity proxy ($\times 10^3$ lines)"
        ambiguity_name = "stanford_schedule_ambiguity_line"

    _single(
        ambiguity,
        ambiguity_ylabel,
        None,
        None,
        ambiguity_name,
    )


def print_schedule_summary(
    widths: dict[str, np.ndarray],
    cumulative_energy: dict[str, np.ndarray],
    stage_energy: dict[str, np.ndarray],
    ambiguity: dict[str, np.ndarray],
) -> None:
    print("\nSchedule summary")
    print("=" * 88)

    for name in SCHEDULES:
        print(f"{name}")
        print("  widths:      ", widths[name].tolist())
        print("  cumulative E:", np.round(100.0 * cumulative_energy[name], 2).tolist())
        print("  stage E:     ", np.round(100.0 * stage_energy[name], 2).tolist())
        print("  ambiguity:   ", np.round(ambiguity[name], 2).tolist())

    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Stanford2D schedule ablations in a NeurIPS-style format."
    )
    parser.add_argument(
        "--info-file",
        type=Path,
        default=Path("/working2/arctic/pdac_new/utils/gamma_stanford_multicoil_384_raw_gpu_hold.txt"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--acs-size", type=int, default=16)
    parser.add_argument("--full-size", type=int, default=384)
    parser.add_argument(
        "--energy-mode",
        type=str,
        choices=["stage", "cumulative"],
        default="stage",
        help="Use 'stage' for per-stage allocated energy, or 'cumulative' for cumulative energy.",
    )
    parser.add_argument(
        "--ambiguity-mode",
        type=str,
        choices=["square", "line"],
        default="square",
        help="Proxy for ambiguity / null-space size. 'square' is recommended.",
    )
    parser.add_argument(
        "--no-single",
        action="store_true",
        help="Only save the three-panel figure.",
    )

    args = parser.parse_args()

    set_neurips_plot_style()

    info_by_window = parse_gaussian_info(args.info_file)
    energy_by_window = build_cumulative_energy(
        info_by_window,
        acs_size=args.acs_size,
        full_size=args.full_size,
    )

    stages, widths, cumulative_energy, stage_energy, ambiguity = collect_schedule_data(
        energy_by_window=energy_by_window,
        acs_size=args.acs_size,
        ambiguity_mode=args.ambiguity_mode,
    )

    print_schedule_summary(widths, cumulative_energy, stage_energy, ambiguity)

    make_three_panel_figure(
        stages=stages,
        widths=widths,
        cumulative_energy=cumulative_energy,
        stage_energy=stage_energy,
        ambiguity=ambiguity,
        out_dir=args.out_dir,
        energy_mode=args.energy_mode,
        ambiguity_mode=args.ambiguity_mode,
    )

    if not args.no_single:
        make_single_panel_figures(
            stages=stages,
            widths=widths,
            cumulative_energy=cumulative_energy,
            stage_energy=stage_energy,
            ambiguity=ambiguity,
            out_dir=args.out_dir,
            energy_mode=args.energy_mode,
            ambiguity_mode=args.ambiguity_mode,
        )


if __name__ == "__main__":
    main()