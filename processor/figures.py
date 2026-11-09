"""Report figures (PNG, 150 dpi, session id in the title). Captions live in main.py's summary.json."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 150
LABEL_COLOURS = {"good": "#2a78d6", "mua": "#eb6834", "noise": "#9a9893"}
LABELS = ("good", "mua", "noise")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "legend.frameon": False, "figure.facecolor": "white",
})

# Histograms with Bombcell's thresholds: (metric, x label, log x)
METRIC_PANELS = [
    ("snr", "SNR", True),
    ("amplitude_median", "|median amplitude| (uV)", True),
    ("presence_ratio", "presence ratio", False),
    ("sliding_rp_violation", "refractory contamination (sliding RP)", False),
    ("amplitude_cutoff", "amplitude cutoff (missed spikes)", False),
    ("firing_rate", "firing rate (Hz)", True),
    ("drift_ptp", "drift peak-to-peak (um)", False),
    ("peak_to_trough_duration", "trough-to-peak duration (ms)", False),
]


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", s)


def _legend(ax, labels=LABELS, counts=None, loc="upper right"):
    from matplotlib.patches import Patch

    h = [Patch(color=LABEL_COLOURS[l], label=f"{l} ({counts[l]})" if counts is not None else l) for l in labels]
    ax.legend(handles=h, loc=loc, fontsize=8)


def unit_yield(sid: str, table: pd.DataFrame, path: Path) -> Path:
    t = table.copy()
    t["area"] = t["brain_area"].replace("", "unknown").fillna("unknown")
    counts = t.groupby(["area", "label"]).size().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0)
    # order regions by depth (deepest = smallest y at the bottom)
    order = t.groupby("area")["depth"].median().sort_values().index
    counts = counts.reindex(order)
    has_both = t["bombcell_label"].notna().any() and t["unitrefine_label"].notna().any() \
        and (t["unitrefine_label"] != "").any()
    fig, axes = plt.subplots(1, 2 if has_both else 1, figsize=(10 if has_both else 6, max(2.6, 0.45 * len(counts) + 1.6)),
                             gridspec_kw={"width_ratios": [1.6, 1]} if has_both else None)
    ax = axes[0] if has_both else axes
    left = np.zeros(len(counts))
    y = np.arange(len(counts))
    for lab in LABELS:
        v = counts[lab].to_numpy()
        ax.barh(y, v, left=left, color=LABEL_COLOURS[lab], height=0.6, edgecolor="white", linewidth=1.5, label=lab)
        left += v
    for yi, tot in zip(y, left):
        ax.text(tot, yi, f"  {int(tot)}", va="center", color=MUTED, fontsize=8)
    ax.set_yticks(y, counts.index)
    ax.set_xlabel("units")
    ax.set_title("Units per brain region (deepest at the bottom)")
    tot = t["label"].value_counts().reindex(LABELS, fill_value=0)
    _legend(ax, counts=tot, loc="lower right")
    ax.set_xlim(0, max(left.max() * 1.15, 1))
    if has_both:
        ax2 = axes[1]
        brow = ["good", "mua", "non_soma", "noise"]
        ucol = ["sua", "mua", "noise"]
        ct = pd.crosstab(t["bombcell_label"].astype(str).str.replace("non_soma.*", "non_soma", regex=True),
                         t["unitrefine_label"].astype(str)).reindex(index=brow, columns=ucol, fill_value=0)
        im = ax2.imshow(ct.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=max(ct.to_numpy().max(), 1))
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                v = ct.iat[i, j]
                ax2.text(j, i, str(v), ha="center", va="center",
                         color="white" if v > 0.6 * ct.to_numpy().max() else INK, fontsize=9)
        ax2.set_xticks(range(len(ucol)), ucol)
        ax2.set_yticks(range(len(brow)), brow)
        ax2.set_xlabel("UnitRefine")
        ax2.set_ylabel("Bombcell")
        ax2.set_title("Where the two methods agree")
        for s in ax2.spines.values():
            s.set_visible(False)
        del im
    fig.suptitle(f"{sid}: unit yield by label and brain region", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def metric_distributions(sid: str, table: pd.DataFrame, thresholds: dict, path: Path) -> Path:
    flat = {}
    for sec in ("noise", "mua"):
        for m, t in thresholds.get(sec, {}).items():
            flat[m] = (sec, t)
    panels = [p for p in METRIC_PANELS if p[0] in table.columns and table[p[0]].notna().any()]
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol)) or 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.6 * nrow + 0.6), squeeze=False)
    for ax, (m, xl, logx) in zip(axes.ravel(), panels):
        v = table[m].astype(float)
        scale = 1000.0 if m == "peak_to_trough_duration" else 1.0
        v = v.abs() * scale if m == "amplitude_median" else v * scale
        finite = v[np.isfinite(v)]
        if logx:
            finite = finite[finite > 0]
        if finite.empty:
            ax.set_visible(False)
            continue
        lo, hi = finite.min(), finite.max()
        if logx:
            bins = np.logspace(np.log10(lo), np.log10(hi if hi > lo else lo * 10), 30)
            ax.set_xscale("log")
        else:
            pad = 0.05 * (hi - lo) if hi > lo else max(abs(lo) * 0.05, 0.01)
            bins = np.linspace(lo - pad, hi + pad, 30)
        data = [v[(table["label"] == lab) & np.isfinite(v)] for lab in LABELS]
        ax.hist(data, bins=bins, stacked=True, color=[LABEL_COLOURS[l] for l in LABELS], edgecolor="white",
                linewidth=0.6)
        if m in flat:
            sec, t = flat[m]
            for key in ("greater", "less"):
                x = t.get(key)
                if x is not None:
                    ax.axvline(x * scale, color=INK, lw=1.2, ls="--")
                    ax.text(x * scale, ax.get_ylim()[1] * 0.97, f" {sec} {'>' if key == 'greater' else '<'} "
                            f"{x * scale:g}", fontsize=7, color=INK, va="top")
        ax.set_xlabel(xl)
        ax.set_ylabel("units")
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    from matplotlib.patches import Patch

    cnt = table["label"].value_counts().reindex(LABELS, fill_value=0)
    fig.legend(handles=[Patch(color=LABEL_COLOURS[l], label=f"{l} ({cnt[l]})") for l in LABELS],
               loc="upper right", ncol=3, fontsize=8.5)
    fig.suptitle(f"{sid}: quality metrics by final label (dashed: Bombcell thresholds)", fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def unit_gallery(sid: str, table: pd.DataFrame, analyzer, spike_times: dict, n: int, path: Path) -> Path | None:
    """Top good units by SNR: waveform on the peak channel and neighbours, ACG, amplitude over time."""
    sel = table[table["label"] == "good"].sort_values("snr", ascending=False).head(n)
    if sel.empty:
        sel = table[table["label"] == "mua"].sort_values("snr", ascending=False).head(n)
    if sel.empty:
        return None
    tmpl_ext = analyzer.get_extension("templates")
    templates = tmpl_ext.get_templates(operator="average")
    fs = analyzer.sampling_frequency
    nbefore = tmpl_ext.nbefore
    tms = (np.arange(templates.shape[1]) - nbefore) / fs * 1000
    ccg, bins = analyzer.get_extension("correlograms").get_data()
    amps = analyzer.get_extension("spike_amplitudes").get_data(outputs="by_unit")[0]
    pos = analyzer.get_channel_locations()
    unit_index = {u: i for i, u in enumerate(analyzer.unit_ids)}
    per_row = 2
    nrow = int(np.ceil(len(sel) / per_row))
    fig, axes = plt.subplots(nrow, 3 * per_row, figsize=(13, 1.9 * nrow + 0.7), squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1, 1.6] * per_row})
    for k, (uid, row) in enumerate(sel.iterrows()):
        r, c0 = divmod(k, per_row)
        c0 *= 3
        i = unit_index[uid]
        pc = int(row["peak_channel_index"])
        d = np.abs(pos[:, 1] - pos[pc, 1]) + np.abs(pos[:, 0] - pos[pc, 0]) * 0.1
        neigh = np.argsort(d)[:4]
        ax = axes[r, c0]
        for ch in neigh[::-1]:
            ax.plot(tms, templates[i, :, ch], color=LABEL_COLOURS[row["label"]] if ch == pc else "#b5b3ad",
                    lw=1.6 if ch == pc else 0.9)
        ax.set_xlim(-1.5, 2.5)
        area = f" {row['brain_area']}" if row.get("brain_area") else ""
        ax.set_title(f"unit {uid}{area}", fontsize=8, loc="left")
        ax.set_ylabel("uV", fontsize=7)
        ax.tick_params(labelsize=7)
        if r == nrow - 1:
            ax.set_xlabel("ms", fontsize=7)
        ax = axes[r, c0 + 1]
        acg = ccg[i, i].astype(float)
        centers = (bins[:-1] + bins[1:]) / 2
        ax.bar(centers, acg, width=np.diff(bins)[0] * 0.9, color=LABEL_COLOURS[row["label"]])
        ax.axvspan(-1.5, 1.5, color="#e34948", alpha=0.12, lw=0)
        ax.set_title(f"ACG, {row['firing_rate']:.1f} Hz", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
        if r == nrow - 1:
            ax.set_xlabel("lag (ms)", fontsize=7)
        ax = axes[r, c0 + 2]
        st = spike_times.get(uid)
        a = amps.get(uid) if isinstance(amps, dict) else None
        if st is not None and a is not None and len(st) == len(a):
            ax.scatter(st, np.abs(a), s=1.5, color=LABEL_COLOURS[row["label"]], alpha=0.5, lw=0, rasterized=True)
        ax.set_title(f"amplitude over time, SNR {row['snr']:.1f}", fontsize=8, loc="left")
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("|uV|", fontsize=7)
        if r == nrow - 1:
            ax.set_xlabel("session time (s)", fontsize=7)
    for k in range(len(sel), nrow * per_row):
        r, c0 = divmod(k, per_row)
        for j in range(3):
            axes[r, c0 * 3 + j].set_visible(False)
    which = "good" if (sel["label"] == "good").all() else "best (no good units)"
    fig.suptitle(f"{sid}: top {len(sel)} {which} units by SNR", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def depth_map(sid: str, table: pd.DataFrame, channel_y: np.ndarray, channel_areas, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 7.0))
    fr = table["firing_rate"].astype(float).clip(lower=1e-3)
    amp = table["amplitude_median"].astype(float).abs().fillna(0)
    size = 8 + 40 * (amp / max(amp.max(), 1))
    for lab in ("noise", "mua", "good"):
        s = table["label"] == lab
        ax.scatter(fr[s], table.loc[s, "depth"], s=size[s], color=LABEL_COLOURS[lab], alpha=0.8 if lab != "noise" else 0.5,
                   edgecolor="white", linewidth=0.6, label=f"{lab} ({int(s.sum())})", zorder=3 if lab == "good" else 2)
    ax.set_xscale("log")
    ax.set_xlabel("firing rate (Hz, log)")
    ax.set_ylabel("depth along the probe (um)")
    if channel_areas is not None and len(channel_y):
        areas = np.asarray([str(a) if str(a) not in ("", "nan", "None") else "unknown" for a in channel_areas])
        order = np.argsort(channel_y)
        ys, ar = np.asarray(channel_y)[order], areas[order]
        start = 0
        shade = False
        for k in range(1, len(ys) + 1):
            if k == len(ys) or ar[k] != ar[start]:
                y0, y1 = ys[start], ys[k - 1]
                if shade:
                    ax.axhspan(y0, y1, color="#f0efec", zorder=0, lw=0)
                ax.text(1.01, (y0 + y1) / 2, ar[start], transform=ax.get_yaxis_transform(), fontsize=7.5,
                        color=MUTED, va="center")
                shade = not shade
                start = k
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=3, fontsize=8)
    ax.set_title(f"{sid}: firing rate vs depth (dot size = amplitude)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def agreement(sid: str, table: pd.DataFrame, summary: dict, path: Path) -> Path:
    kind = summary["kind"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    by = summary["matched_by_label"]
    labs = [l for l in LABELS if by[l]["n"] > 0]
    frac = [by[l]["matched"] / by[l]["n"] for l in labs]
    ax.bar(labs, frac, color=[LABEL_COLOURS[l] for l in labs], width=0.6)
    for x, (l, f) in enumerate(zip(labs, frac)):
        ax.text(x, f, f"{by[l]['matched']}/{by[l]['n']}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("fraction matched (agreement >= 0.5)")
    ref_name = "a ground-truth unit" if kind == "ground_truth" else "a reference unit"
    ax.set_title(f"Our units that match {ref_name}", loc="left")
    ax = axes[1]
    if kind == "ground_truth" and "_performance" in summary:
        perf = summary["_performance"]
        acc = np.sort(perf["accuracy"].to_numpy(dtype=float))[::-1]
        ax.plot(np.arange(1, len(acc) + 1), acc, color="#2a78d6", lw=2, marker="o", ms=3)
        ax.axhline(0.8, color=INK, ls="--", lw=1)
        ax.text(len(acc), 0.8, " well detected", fontsize=7, va="bottom", ha="right")
        ax.set_xlabel("ground-truth units, best first")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Ground-truth recovery: {summary['n_well_detected']} of {summary['n_reference_units']} "
                     "well detected", loc="left")
    else:
        rt = summary["_reference_table"]
        a = np.sort(rt["best_agreement"].to_numpy(dtype=float))[::-1]
        ax.plot(np.arange(1, len(a) + 1), a, color="#2a78d6", lw=2)
        ax.axhline(0.5, color=INK, ls="--", lw=1)
        ax.set_xlabel("reference units in the window, best first")
        ax.set_ylabel("best agreement with one of our units")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Reference units recovered: {int((a >= 0.5).sum())} of {len(a)}", loc="left")
    fig.suptitle(f"{sid}: agreement with the {'ground truth' if kind == 'ground_truth' else 'reference sorting'}",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
