"""
Agreement with a reference sorting or a ground truth.

The reference is an NWB file with a Units table (IBL's processed NWB of the same session, a
ground-truth simulation, any lab's earlier curation) or a CSV of spikes (unit_id, spike_time in
session seconds). Its spikes are cut to the recording's window and converted to samples of the
recording through the recording's own times, so both sortings are on the same clock.

- Reference sorting: SpikeInterface compare_two_sorters (Hungarian matching on agreement =
  matched spikes / (n1 + n2 - matched), within MATCH_DELTA_MS). A unit is "matched" when the
  agreement is at least 0.5.
- Ground truth: compare_sorter_to_ground_truth: per ground-truth unit accuracy, precision, recall;
  well detected = accuracy >= 0.8; and for each of our labels, how many units match a true unit.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("ephys-curate")

GT_WORDS = ("ground truth", "ground-truth", "groundtruth", "simulated", "simulation", "hybrid")
MATCH_THRESHOLD = 0.5
WELL_DETECTED = 0.8
# Columns of a reference Units table that say which probe a unit is on (IBL: probe00 / probe01).
PROBE_COLUMNS = ("probe", "probe_name", "probe_label", "pname", "electrode_group_name")


def _probe_key(text: str) -> str:
    """'Probe01', 'probe01', 'ElectricalSeriesProbe01AP', 'imec1' -> 'probe1'; else the lower-cased text."""
    import re

    m = re.search(r"(?:probe|imec)\s*_?0*(\d+)", str(text), flags=re.I)
    return f"probe{int(m.group(1))}" if m else str(text).strip().lower()


def probe_hint(session: dict) -> str | None:
    """Which probe the recording is, from session.json (stream, probe name, source)."""
    import re

    probe = session.get("probe") if isinstance(session.get("probe"), dict) else {}
    for v in (session.get("stream"), probe.get("name"), session.get("electrical_series"), session.get("source")):
        if v and re.search(r"(?:probe|imec)\s*_?\d+", str(v), flags=re.I):
            return _probe_key(str(v))
    return None


def _decode(v):
    return v.decode() if isinstance(v, bytes) else v


def read_reference_nwb(path: Path, probe_hint: str | None = None) -> tuple[dict, dict]:
    """{unit_name: spike_times (s)} and info (kind hints, columns, probe filter)."""
    import h5py

    info: dict = {"path": path.name, "format": "nwb"}
    with h5py.File(path, "r") as f:
        if "units" not in f:
            raise ValueError(f"{path.name} has no units table")
        u = f["units"]
        st = u["spike_times"][:]
        idx = u["spike_times_index"][:]
        ids = u["id"][:]
        desc = " ".join(str(_decode(x)) for x in (u.attrs.get("description", ""),
                                                  f.attrs.get("session_description", ""),
                                                  f["session_description"][()] if "session_description" in f else ""))
        info["description"] = desc[:300]
        info["columns"] = [k for k in u.keys() if not k.endswith("_index")]
        names = None
        for c in ("unit_name", "cluster_id", "gt_unit", "cluster_uuid"):
            if c in u and u[c].shape == ids.shape:
                names = [str(_decode(x)) for x in u[c][:]]
                info["name_column"] = c
                break
        names = names or [str(x) for x in ids]
        keep = np.ones(len(ids), dtype=bool)
        if probe_hint:
            for c in PROBE_COLUMNS:
                if c in u and u[c].shape == ids.shape:
                    vals = np.array([_probe_key(str(_decode(x))) for x in u[c][:]])
                    hit = vals == _probe_key(probe_hint)
                    if hit.any() and not hit.all():
                        keep = hit
                        info["probe_filter"] = f"{c} == {probe_hint}"
                    break
            else:
                if "electrode_group" in u:
                    try:
                        refs = u["electrode_group"][:]
                        gnames = np.array([f[r].name.split("/")[-1].lower() for r in refs])
                        hit = np.array([_probe_key(g) == _probe_key(probe_hint) for g in gnames])
                        if hit.any() and not hit.all():
                            keep = hit
                            info["probe_filter"] = f"electrode_group ~ {probe_hint}"
                    except Exception:   # noqa: BLE001 - a hint only
                        pass
        trains = {}
        start = 0
        for i, end in enumerate(idx):
            if keep[i]:
                trains[names[i]] = st[start:end].astype("float64")
            start = int(end)
    info["n_units_file"] = int(len(ids))
    return trains, info


def read_reference_csv(path: Path) -> tuple[dict, dict]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    ucol = next(cols[c] for c in ("unit_id", "cluster_id", "unit", "cluster") if c in cols)
    tcol = next(cols[c] for c in ("spike_time", "spike_times", "time", "times", "t") if c in cols)
    trains = {str(k): g[tcol].to_numpy(dtype="float64") for k, g in df.groupby(ucol)}
    return trains, {"path": path.name, "format": "csv", "description": path.stem, "n_units_file": len(trains)}


def reference_kind(info: dict, path: Path, requested: str) -> str:
    if requested in ("sorting", "ground_truth"):
        return requested
    text = f"{info.get('description', '')} {path.name}".lower().replace("_", " ")
    return "ground_truth" if any(w in text for w in GT_WORDS) else "sorting"


def to_sorting(trains: dict, recording, min_spikes: int = 1, times=None):
    """Reference spike times (session s) -> NumpySorting in samples of the recording's window."""
    import spikeinterface.full as si

    times = recording.get_times() if times is None else times
    t0, t1 = float(times[0]), float(times[-1])
    unit_trains = {}
    for name, t in trains.items():
        t = np.sort(t[(t >= t0) & (t <= t1)])
        if len(t) < min_spikes:
            continue
        frames = np.searchsorted(times, t)          # first sample at or after the spike time
        frames = np.clip(frames, 1, len(times) - 1)
        nearer_left = (t - times[frames - 1]) < (times[frames] - t)
        frames = frames - nearer_left.astype(int)   # nearest sample
        unit_trains[name] = frames.astype("int64")
    if not unit_trains:
        return None, (t0, t1)
    sorting = si.NumpySorting.from_unit_dict([unit_trains], sampling_frequency=recording.sampling_frequency)
    return sorting, (t0, t1)


def compare(ours, labels: pd.Series, ref_sorting, kind: str, delta_ms: float) -> tuple[pd.DataFrame, dict]:
    """Per our unit: matched reference unit and agreement; summary numbers."""
    import spikeinterface.full as si

    if kind == "ground_truth":
        cmp = si.compare_sorter_to_ground_truth(ref_sorting, ours, gt_name="ground truth", tested_name="ours",
                                                delta_time=delta_ms, exhaustive_gt=False)
        perf = cmp.get_performance(method="by_unit")
        m21 = cmp.hungarian_match_21       # ours -> gt
        scores = cmp.agreement_scores      # gt x ours
    else:
        cmp = si.compare_two_sorters(ours, ref_sorting, sorting1_name="ours", sorting2_name="reference",
                                     delta_time=delta_ms)
        perf = None
        m21 = cmp.hungarian_match_12       # ours -> reference
        scores = cmp.agreement_scores.T    # reference x ours
    rows = []
    for u in ours.unit_ids:
        r = m21.get(u, -1) if hasattr(m21, "get") else -1
        r = None if (r is None or str(r) in ("", "-1")) else r
        agr = float(scores.at[r, u]) if r is not None and r in scores.index and u in scores.columns else 0.0
        rows.append({"unit_id": u, "reference_match": "" if r is None else str(r),
                     "reference_agreement": round(agr, 4)})
    table = pd.DataFrame(rows).set_index("unit_id")
    matched = table["reference_agreement"] >= MATCH_THRESHOLD
    lab = labels.reindex(table.index)
    by_label = {}
    for name in ("good", "mua", "noise"):
        sel = lab == name
        by_label[name] = {"n": int(sel.sum()), "matched": int((matched & sel).sum())}
    summary = {
        "kind": kind, "delta_ms": delta_ms, "match_threshold": MATCH_THRESHOLD,
        "n_reference_units": int(ref_sorting.get_num_units()), "n_units": int(ours.get_num_units()),
        "n_matched": int(matched.sum()), "matched_by_label": by_label,
    }
    if perf is not None:
        summary.update({
            "n_well_detected": int(cmp.count_well_detected_units(WELL_DETECTED)),
            "well_detected_threshold": WELL_DETECTED,
            "median_accuracy": float(np.nanmedian(perf["accuracy"])) if len(perf) else None,
            "median_recall": float(np.nanmedian(perf["recall"])) if len(perf) else None,
            "median_precision": float(np.nanmedian(perf["precision"])) if len(perf) else None,
        })
        try:
            summary["n_redundant_units"] = int(cmp.count_redundant_units())
        except Exception:   # noqa: BLE001 - only defined for some comparisons
            pass
        perf = perf.copy()
        perf.index = perf.index.astype(str)
        summary["_performance"] = perf
    ref_rows = []
    best_ours = scores.idxmax(axis=1) if scores.shape[1] else pd.Series(dtype=object)
    for r in scores.index:
        a = float(scores.loc[r].max()) if scores.shape[1] else 0.0
        ref_rows.append({"reference_unit": str(r), "best_unit": str(best_ours.get(r, "")),
                         "best_agreement": round(a, 4),
                         "n_spikes_in_window": int(len(ref_sorting.get_unit_spike_train(r)))})
    summary["_reference_table"] = pd.DataFrame(ref_rows)
    return table, summary
