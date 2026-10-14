"""Accepted Figure 5 typography and annotation placement."""

from matplotlib.text import Annotation


def fix_distribution_labels(fig):
    """Fixed side and offset per series, as requested for the two distributions."""
    axes = fig.axes[:2]
    dense = len(axes[0].lines[0].get_xdata()) > 8
    fig.set_size_inches(11.8, 5.9)
    fig.set_dpi(150)
    offsets = [12, -12] if dense else [-12, 12]  # Blue, red.
    labels = []
    choices = []
    for axis, ax in enumerate(axes):
        for text in ax.texts:
            if not isinstance(text, Annotation):
                continue
            dy = offsets[axis]
            text.set_fontsize(14 if dense else 18)
            text.set_clip_on(False)
            text.set_position((0, dy))
            text.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.4))
            text.set_ha("center")
            text.set_va("bottom" if dy > 0 else "top")
            labels.append(text)
            choices.append(
                dict(
                    text=text.get_text(),
                    offset_points=[0, dy],
                    horizontal_alignment="center",
                    axis=axis,
                )
            )
        ax.tick_params(axis="y", labelsize=18)
    axes[0].tick_params(axis="x", labelsize=16, pad=6)
    for text in axes[0].get_xticklabels():
        text.set_rotation(90 if dense else 0)
        text.set_ha("right" if dense else "center")
    for text in axes[0].get_legend().get_texts():
        text.set_fontsize(22)
    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.27, top=0.96)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [text.get_window_extent(renderer).expanded(1.04, 1.08) for text in labels]
    overlap = int(sum(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :]))
    assert overlap == 0, overlap
    for text, box in zip(labels, boxes):
        bounds = text.axes.get_window_extent(renderer)
        assert bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1), (
            text.get_text(),
            box.bounds,
            bounds.bounds,
        )
    return dict(
        size_inches=[11.8, 5.9],
        axes_bounds=list(axes[0].get_position().bounds),
        numeric_labels=len(labels),
        label_overlaps=overlap,
        placements=choices,
        consistent_offsets=True,
        blue_position="above" if dense else "below",
        red_position="below" if dense else "above",
    )


def render(categories_path, sources, output):
    """Reuse the original drawing cell, with the approved typography only."""
    import json

    import matplotlib.pyplot as plt
    import numpy as np

    data = json.loads(categories_path.read_text())
    cells = json.loads((sources / "modular_addition_unshared.json").read_text())
    cell = cells["15"]
    cell = cell[cell.index("# Sort by proportion (ascending order)") :]
    original_save, original_show = plt.savefig, plt.show
    checks = {}
    try:
        for name, categories in data.items():

            def save(*args, **kwargs):
                fig = plt.gcf()
                checks[name] = fix_distribution_labels(fig)
                for extension in ["jpg", "pdf"]:
                    fig.savefig(output / f"{name}.{extension}", dpi=300, bbox_inches=None)

            plt.savefig = save
            plt.show = lambda *args, **kwargs: plt.close("all")
            with plt.rc_context(
                {
                    "font.family": "serif",
                    "font.serif": ["Palatino", "P052", "DejaVu Serif"],
                    "mathtext.fontset": "cm",
                }
            ):
                exec(
                    compile(cell, "original-distribution-cell", "exec"),
                    {"np": np, "plt": plt, "freq_group_data": categories},
                )
    finally:
        plt.savefig, plt.show = original_save, original_show
    return checks
