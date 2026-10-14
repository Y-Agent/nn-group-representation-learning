"""Accepted Figure 7, drawn directly without chaining draft renderers.

The data adapter retains notebook metrics. Here we only compose the four panels.
Losses are mean CE (summed risk divided by 21**2); time is physical training time.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from pypdf import PdfReader, PdfWriter, Transformation

from . import settings as S


def axis_position(height, bottom, x=S.LEFT, width=S.PLOT_WIDTH):
    return [x / S.WIDTH, bottom / height, width / S.WIDTH, S.PLOT_HEIGHT / height]


def panel(index, upper=True):
    height = S.TOP_HEIGHT if upper else S.BOTTOM_HEIGHT
    bottom = 1.10 if upper else 1.75
    fig = plt.figure(figsize=(S.WIDTH, height))
    ax = fig.add_axes(axis_position(height, bottom))
    fig.text(
        S.LEFT / S.WIDTH,
        (5.82 if upper else 6.47) / height,
        index,
        fontsize=S.AXIS_FONT,
        va="bottom",
    )
    ax.tick_params(labelsize=S.TICK_FONT)
    ax.grid(True, alpha=0.22)
    return fig, ax


def separate_corner_labels(time_axis, right_axis, last_time):
    time_axis.figure.canvas.draw()
    for tick in time_axis.xaxis.get_major_ticks():
        if np.isclose(tick.get_loc(), last_time):
            tick.label1.set_ha("right")
    for tick in right_axis.yaxis.get_major_ticks():
        if np.isclose(tick.get_loc(), 0):
            tick.label2.set_va("bottom")
    time_axis.figure.canvas.draw()
    renderer = time_axis.figure.canvas.get_renderer()
    x = next(
        t.label1 for t in time_axis.xaxis.get_major_ticks() if np.isclose(t.get_loc(), last_time)
    )
    y = next(t.label2 for t in right_axis.yaxis.get_major_ticks() if np.isclose(t.get_loc(), 0))
    if x.get_window_extent(renderer).overlaps(y.get_window_extent(renderer)):
        raise ValueError("Terminal time tick overlaps the right-axis zero")


def verify_text_bounds(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = list(fig.texts) + list(fig.legends)
    for ax in fig.axes:
        artists += [ax.xaxis.label, ax.yaxis.label]
        for ticks, limits in [
            (ax.xaxis.get_major_ticks(), ax.get_xlim()),
            (ax.yaxis.get_major_ticks(), ax.get_ylim()),
        ]:
            for tick in ticks:
                if min(limits) <= tick.get_loc() <= max(limits):
                    artists += [tick.label1, tick.label2]
        if ax.get_legend() is not None:
            artists.append(ax.get_legend())
    for artist in artists:
        if not artist.get_visible():
            continue
        box = artist.get_window_extent(renderer)
        if (
            box.width
            and box.height
            and not (
                box.x0 >= -1
                and box.y0 >= -1
                and box.x1 <= fig.bbox.x1 + 1
                and box.y1 <= fig.bbox.y1 + 1
            )
        ):
            raise ValueError(f"Text outside figure: {artist}")


def save(fig, directory, name):
    verify_text_bounds(fig)
    for ext in ["png", "pdf"]:
        fig.savefig(directory / f"{name}.{ext}", dpi=160, bbox_inches=None)


def alignment_panel(data):
    fig, ax = panel("(i)")
    right = ax.twinx()
    cadence = max(1, len(data["time"]) // 20)
    for axis, key, style in [(ax, "alignment", "-"), (right, "rank", "--")]:
        for i in range(3):
            axis.errorbar(
                data["time"],
                data[key + "_mean"][i],
                yerr=data[key + "_std"][i],
                color=S.DIAGNOSTIC_COLORS[i],
                ls=style,
                lw=S.LINE_WIDTH,
                marker=["o", "s", "^"][i],
                ms=4,
                markerfacecolor="none",
                markeredgewidth=1.5,
                markevery=cadence,
                errorevery=cadence,
                capsize=2,
                elinewidth=1.5,
            )
        axis.tick_params(labelsize=S.TICK_FONT)
    ax.set_ylim(0, 1.06)
    right.set_ylim(0, float(np.max(data["rank_mean"] + data["rank_std"])) * 1.08)
    ax.set_xlim(-500, 40500)
    ax.set_xticks([0, 10000, 20000, 30000, 40000])
    ax.set_xlabel("Time", fontsize=S.AXIS_FONT)
    ax.set_ylabel("Avg. alignment", fontsize=S.AXIS_FONT)
    right.set_ylabel(r"Avg. $\sigma_{(2)} / \sigma_{(1)}$", fontsize=S.AXIS_FONT, labelpad=9)
    handles = [
        Line2D(
            [],
            [],
            color=S.DIAGNOSTIC_COLORS[i],
            marker=["o", "s", "^"][i],
            ls="none",
            markerfacecolor="none",
            ms=6,
            label=S.LAYER_LABELS[i],
        )
        for i in range(3)
    ]
    handles += [
        Line2D([], [], color="#444444", lw=S.LINE_WIDTH, ls=style, label=label)
        for style, label in [("-", "Alignment"), ("--", "Rank ratio")]
    ]
    right.legend(
        handles=handles,
        loc="center right",
        bbox_to_anchor=(0.98, 0.50),
        ncol=1,
        frameon=True,
        framealpha=0.95,
        fontsize=24,
        borderpad=0.35,
        handlelength=1.3,
        handletextpad=0.5,
        labelspacing=0.3,
    )
    separate_corner_labels(ax, right, S.STAGE_BOUNDARY)
    return fig


def accuracy_panel(data):
    fig, ax = panel("(ii)", upper=False)
    ax.plot(data["accuracy_time"], data["accuracy"], color=S.DIAGNOSTIC_COLORS[0], lw=S.LINE_WIDTH)
    ax.set_xlim(-500, 40500)
    ax.set_ylim(-0.04, 1.08)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([0, 10000, 20000, 30000, 40000])
    ax.set_xlabel("Time", fontsize=S.AXIS_FONT)
    ax.set_ylabel("Accuracy", fontsize=S.AXIS_FONT)
    return fig


def scale_data(runs):
    history = json.loads((runs / "f21_wide_h2/history.json").read_text())
    with (
        np.load(runs / "f21_scale_wide_h2/tied.npz") as tied,
        np.load(runs / "f21_scale_wide_h2/untied_matched.npz") as untied,
    ):
        if not np.array_equal(tied["time"], untied["time"]):
            raise ValueError("Stage-II variants have different observation times")
        sample = np.flatnonzero(np.isclose(np.mod(tied["time"], S.SAMPLE_INTERVAL), 0))
        t2 = tied["time"][sample].copy()
        t1 = np.asarray([r["time"] for r in history[:-1]])
        if history[-1]["time"] != S.STAGE_BOUNDARY or not np.array_equal(
            t1, np.arange(len(t1)) * S.SAMPLE_INTERVAL
        ):
            raise ValueError("Unexpected Stage-I observation grid")
        if not np.array_equal(t2, np.arange(len(t2)) * S.SAMPLE_INTERVAL):
            raise ValueError("Unexpected Stage-II observation grid")
        scales = untied["a"][sample].copy()
        # Preserve original cell 12: first TEN neurons, ordered at the full
        # saved endpoint (not chosen using the displayed 40k–50k window).
        neuron_ids = np.arange(10)[np.argsort(scales[-1, :10])]
        return dict(
            stage1_time=t1,
            stage1_loss=np.asarray([r["loss_sum"] for r in history[:-1]]) / 21**2,
            stage2_time=S.STAGE_BOUNDARY + t2,
            tied_loss=tied["loss_sum"][sample] / 21**2,
            untied_loss=untied["loss_sum"][sample] / 21**2,
            tied_scale=tied["a"][sample].reshape(-1).copy(),
            neuron_ids=neuron_ids,
            neuron_scale=scales[:, neuron_ids],
            cadence=max(1, len(t2) // 20),
        )


def loss_panel(data):
    fig, small = panel("(i)")
    small.set_position(axis_position(S.TOP_HEIGHT, 1.10, width=1.68))
    large = fig.add_axes(
        axis_position(S.TOP_HEIGHT, 1.10, x=S.LEFT + 2.04, width=6.36), sharey=small
    )
    scale = large.twinx()
    for ax in [small, large]:
        for x, y, color, ls in [
            (data["stage1_time"], data["stage1_loss"], "gray", "--"),
            (data["stage2_time"], data["tied_loss"], "#0D2758", "-"),
            (data["stage2_time"], data["untied_loss"], "#A32015", "--"),
        ]:
            ax.plot(x, y, color=color, ls=ls, lw=S.LINE_WIDTH)
        ax.set_ylim(-0.06, 3.2)
        ax.grid(True, alpha=0.22)
        ax.tick_params(labelsize=S.TICK_FONT)
    small.set_xlim(0, S.STAGE1_BREAK[0])
    small.set_xticks([0, 5000])
    small.set_yticks([0, 1, 2, 3])
    small.set_ylabel("Loss", fontsize=S.AXIS_FONT)
    large.set_xlim(S.STAGE1_BREAK[1], S.VISIBLE_END)
    large.set_xticks([40000, 45000, 50000])
    large.tick_params(left=False, labelleft=False)
    small.spines["right"].set_visible(False)
    large.spines["left"].set_visible(False)
    scale.spines["left"].set_visible(False)
    large.axvline(S.STAGE_BOUNDARY, color="#555555", ls=":", lw=1.8, alpha=0.7)
    (scale_line,) = scale.plot(
        data["stage2_time"],
        data["tied_scale"],
        color="#368A87",
        ls="-.",
        marker="o",
        ms=4.5,
        markevery=data["cadence"],
        lw=S.LINE_WIDTH,
    )
    scale.set_ylim(0, 1000)
    scale.set_yticks([0, 500, 1000])
    scale.grid(False)
    if data["tied_scale"][data["stage2_time"] <= S.VISIBLE_END].max() >= 1000:
        raise ValueError("Visible tied scale exceeds the approved axis limit")
    scale.tick_params(axis="y", labelcolor="#28736F", labelsize=S.TICK_FONT)
    scale.set_ylabel("Tied scale $a$", fontsize=S.AXIS_FONT, color="#28736F")
    scale_left = small.twinx()
    scale_left.set_ylim(0, 1000)
    scale_left.set_yticks([])
    for spine in scale_left.spines.values():
        spine.set_visible(False)
    for ax, xx in [(scale_left, [0, 8000]), (scale, [38000, 40000])]:
        ax.plot(xx, [0.125, 0.125], color="#368A87", ls="-.", lw=S.LINE_WIDTH)
    # Match the approved break marks (their geometry originated on a 12x8 draft).
    for ax, edge, width in [(small, 1, 0.14), (large, 0, 0.53)]:
        dx = 0.004 / width
        dy = 0.006 / 0.71
        for y in [0, 1]:
            ax.plot(
                [edge - dx, edge + dx],
                [y - dy, y + dy],
                transform=ax.transAxes,
                color="#333333",
                lw=1.6,
                clip_on=False,
            )
    fig.text(
        (S.LEFT + S.PLOT_WIDTH / 2) / S.WIDTH,
        0.15 / S.TOP_HEIGHT,
        "Time (Stage I + II)",
        fontsize=S.AXIS_FONT,
        ha="center",
        va="bottom",
    )
    scale.legend(
        large.lines[:3] + [scale_line],
        ["Stage I loss", "Tied loss", "Per-neuron loss", "Tied scale $a$"],
        loc="lower right",
        bbox_to_anchor=(0.98, 0.075),
        fontsize=24,
        framealpha=0.96,
        edgecolor="#dddddd",
        labelspacing=0.3,
    )
    separate_corner_labels(large, scale, S.VISIBLE_END)
    return fig


def neurons_panel(data):
    fig, ax = panel("(ii)", upper=False)
    keep = data["stage2_time"] <= S.VISIBLE_END
    for rank, neuron in enumerate(data["neuron_ids"]):
        ax.plot(
            data["stage2_time"][keep],
            data["neuron_scale"][keep, rank],
            color=S.NEURON_COLORS[rank],
            ls="--",
            lw=S.LINE_WIDTH,
            marker="o",
            ms=4.5,
            markevery=data["cadence"],
            label=f"$m={neuron}$",
        )
    ax.set_xlim(S.STAGE_BOUNDARY, S.VISIBLE_END)
    ax.set_ylim(bottom=-25)
    ax.set_xticks([40000, 42500, 45000, 47500, 50000])
    ax.set_xlabel("Time (Stage I + II)", fontsize=S.AXIS_FONT)
    ax.set_ylabel("Per-neuron scale", fontsize=S.AXIS_FONT)
    ax.legend(
        loc="lower right",
        fontsize=22,
        ncol=2,
        framealpha=0.96,
        edgecolor="#dddddd",
        columnspacing=0.8,
        labelspacing=0.22,
        handlelength=1.3,
        borderpad=0.4,
    )
    return fig


def compose(paths, destination, positions, width, height):
    writer = PdfWriter()
    page = writer.add_blank_page(width=width, height=height)
    for path, (x, y) in zip(paths, positions):
        page.merge_transformed_page(
            PdfReader(path).pages[0], Transformation().translate(tx=x, ty=y)
        )
    writer.write(destination)


def render(runs, output, diagnostics):
    figures = output / "figures"
    panels = output / "panels"
    panels.mkdir(exist_ok=True)
    with np.load(diagnostics) as npz:
        data = {k: npz[k].copy() for k in npz.files}
    scales = scale_data(runs)
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.linewidth": 1.2,
        }
    ):
        names = ["alignment", "accuracy", "scale-loss", "per-neuron-scale"]
        plots = [
            alignment_panel(data),
            accuracy_panel(data),
            loss_panel(scales),
            neurons_panel(scales),
        ]
        for fig, name in zip(plots, names):
            save(fig, panels, name)
            plt.close(fig)
    w, hu, hl = 72 * S.WIDTH, 72 * S.TOP_HEIGHT, 72 * S.BOTTOM_HEIGHT
    paths = [panels / f"{n}.pdf" for n in names]
    compose(
        paths[:2], figures / "frobenius21_diagnostics_compact.pdf", [(0, hl), (0, 0)], w, hu + hl
    )
    compose(paths[2:], figures / "frobenius21_scale_combined.pdf", [(0, hl), (0, 0)], w, hu + hl)
    compose(paths, output / "figure7.pdf", [(0, hl), (0, 0), (w, hl), (w, 0)], 2 * w, hu + hl)
    np.savez_compressed(output / "figure7-plotted-data.npz", **data, **scales)
    return {
        "stage1_break": S.STAGE1_BREAK,
        "stage2_visible_total_time": [S.STAGE_BOUNDARY, S.VISIBLE_END],
        "tied_scale_axis": [0, 1000],
        "neuron_ids": scales["neuron_ids"].tolist(),
        "diagnostic_metric": "original unsigned normalized complex inner product",
        "rank_summary": "original higher-dimensional selected blocks",
        "text_bounds_checked": True,
        "corner_tick_labels_do_not_overlap": True,
    }
