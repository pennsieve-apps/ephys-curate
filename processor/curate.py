"""
Waveforms, metrics and automated curation of one session.

1. SortingAnalyzer (SpikeInterface), sparse (channels within SPARSITY_RADIUS_UM of the unit's peak
   channel), with the AIND ephys pipeline's extensions: random_spikes (MAX_SPIKES_PER_UNIT per unit),
   noise_levels, waveforms (-2/+3 ms), templates, spike_amplitudes, amplitude_scalings,
   unit_locations (monopolar triangulation), spike_locations (grid convolution), correlograms
   (+-50 ms, 1 ms bins), template_similarity, template_metrics (incl. multi-channel), optionally
   principal_components (5 PCs per channel), and quality_metrics.
2. Bombcell (bombcell.run_bombcell_qc, which labels with SpikeInterface's bombcell_label_units):
   good / mua / noise / non_soma from waveform shape and spike-train quality thresholds.
3. UnitRefine: the two pretrained random-forest classifiers (noise vs neural, then SUA vs MUA for
   the neural units), run on the same metrics.
4. A final label from both (see final_label()).
"""
from __future__ import annotations

import json
import logging
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("ephys-curate")

QUALITY_METRICS = ["num_spikes", "firing_rate", "presence_ratio", "snr", "isi_violation", "rp_violation",
                   "sliding_rp_violation", "amplitude_cutoff", "amplitude_median", "amplitude_cv", "firing_range",
                   "drift", "sd_ratio", "synchrony", "noise_cutoff"]
PCA_QUALITY_METRICS = ["mahalanobis", "d_prime", "nearest_neighbor", "silhouette"]
BOMBCELL_DROP_BY_DEFAULT = ("isolation_distance", "l_ratio")   # bombcell: distance metrics off for acute recordings
UNITREFINE_MODELS = {
    "noise_neural": "UnitRefine_noise_neural_classifier",
    "sua_mua": "UnitRefine_sua_mua_classifier",
}
FINAL_LABELS = ("good", "mua", "noise")


def metric_windows(duration_s: float) -> dict:
    """Presence-ratio bins and drift intervals: Bombcell's 60 s, shorter for short windows (>= 10 bins)."""
    bin_s = float(min(60.0, max(duration_s / 10.0, 1.0)))
    drift_s = float(min(60.0, max(duration_s / 5.0, 2.0)))
    return {"presence_ratio_bin_duration_s": bin_s, "drift_interval_s": drift_s, "drift_min_spikes": 100}


def bombcell_thresholds(overrides: dict | None) -> dict:
    from spikeinterface.curation import bombcell_get_default_thresholds

    thr = bombcell_get_default_thresholds()
    for m in BOMBCELL_DROP_BY_DEFAULT:
        thr["mua"].pop(m, None)
    for section, metrics in (overrides or {}).items():
        if section not in thr:
            raise ValueError(f"BOMBCELL_THRESHOLDS: unknown section {section!r} (noise, mua, non-somatic)")
        if not isinstance(metrics, dict):
            raise ValueError(f"BOMBCELL_THRESHOLDS[{section!r}] must be an object of metric -> {{greater, less}}")
        for metric, value in metrics.items():
            if value is None:
                thr[section].pop(metric, None)
            elif isinstance(value, dict) and set(value) <= {"greater", "less", "abs"}:
                thr[section][metric] = {"greater": None, "less": None, **thr[section].get(metric, {}), **value}
            else:
                raise ValueError(f"BOMBCELL_THRESHOLDS[{section!r}][{metric!r}] must be "
                                 "{\"greater\": x, \"less\": y} or null")
    return thr


def session_thresholds(thresholds: dict, duration_s: float, overrides: dict) -> tuple[dict, str]:
    """Bombcell's 300-spike minimum assumes sessions of 10 min or more: in a shorter window it would
    demand a firing rate above 300 / duration (5 Hz in 60 s). Unless BOMBCELL_THRESHOLDS sets
    num_spikes, the minimum becomes 0.5 Hz x duration (= 300 spikes at 10 min)."""
    import copy

    thr = copy.deepcopy(thresholds)
    ns = thr.get("mua", {}).get("num_spikes")
    if ns is None or "num_spikes" in (overrides or {}).get("mua", {}):
        return thr, ""
    g = ns.get("greater")
    if g is not None and duration_s < 600.0:
        new = max(10.0, round(0.5 * duration_s))
        if new < g:
            ns["greater"] = new
            return thr, (f"recording of {duration_s:.0f} s: Bombcell num_spikes minimum {g:g} -> {new:g} "
                         "(0.5 Hz x duration)")
    return thr, ""


def auto_merge(sorting, recording, p: dict) -> tuple:
    """Merge over-split units with SpikeInterface's SLAy preset (Koukuntla et al. 2025).

    A light in-memory analyzer (sampled spikes, noise levels, templates; the preset computes the
    correlograms and template similarity it needs) proposes groups of units to merge; the groups are
    applied to the sorting, with spikes closer than 0.3 ms within a merged unit removed (the same
    unit counted twice). Merged units keep the id of the group's first unit.
    Returns (sorting, [(new_unit_id, [merged unit ids]), ...]).
    """
    import spikeinterface.full as si
    from spikeinterface.curation import compute_merge_unit_groups

    pre = si.create_sorting_analyzer(sorting, recording, format="memory", sparse=True, method="radius",
                                     radius_um=p["SPARSITY_RADIUS_UM"], return_in_uV=True)
    pre.compute({"random_spikes": dict(method="uniform", max_spikes_per_unit=p["MAX_SPIKES_PER_UNIT"],
                                       seed=p["SEED"]),
                 "noise_levels": dict(random_slices_kwargs=dict(seed=p["SEED"])),
                 "templates": dict(ms_before=2.0, ms_after=3.0)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        groups = compute_merge_unit_groups(pre, preset="slay", resolve_graph=True, force_copy=False)
    groups = [list(g) for g in groups if len(g) > 1]
    del pre
    if not groups:
        return sorting, []
    merged = si.apply_merges_to_sorting(sorting, groups, censor_ms=0.3, new_id_strategy="take_first")
    out = [(g[0], g) for g in groups]
    # properties of the first unit of each group (e.g. KSLabel) carry over
    for key in sorting.get_property_keys():
        try:
            vals = dict(zip(sorting.unit_ids, sorting.get_property(key)))
            merged.set_property(key, np.array([vals.get(u) for u in merged.unit_ids]))
        except Exception:   # noqa: BLE001 - a property that cannot be carried is dropped
            pass
    return merged, out


def build_analyzer(sorting, recording, work: Path, p: dict, duration_s: float):
    import spikeinterface.full as si

    if work.exists():
        shutil.rmtree(work)
    analyzer = si.create_sorting_analyzer(
        sorting, recording, format="binary_folder", folder=work, sparse=True, method="radius",
        radius_um=p["SPARSITY_RADIUS_UM"], return_in_uV=True, overwrite=True)
    win = metric_windows(duration_s)
    qm_params = {"presence_ratio": {"bin_duration_s": win["presence_ratio_bin_duration_s"]},
                 "drift": {"interval_s": win["drift_interval_s"],
                           "min_spikes_per_interval": win["drift_min_spikes"]}}
    steps = [
        ("random_spikes", dict(method="uniform", max_spikes_per_unit=p["MAX_SPIKES_PER_UNIT"], seed=p["SEED"])),
        ("noise_levels", dict(random_slices_kwargs=dict(seed=p["SEED"]))),
        ("waveforms", dict(ms_before=2.0, ms_after=3.0)),
        ("templates", dict(operators=["average", "std"])),
        ("spike_amplitudes", {}),
        ("amplitude_scalings", dict(handle_collisions=False)),
        ("unit_locations", dict(method="monopolar_triangulation")),
        ("spike_locations", dict(method="grid_convolution", seed=p["SEED"])),
        ("correlograms", dict(window_ms=50.0, bin_ms=1.0)),
        ("template_similarity", dict(method="cosine")),
        ("template_metrics", dict(include_multi_channel_metrics=True)),
    ]
    if p["PCA_METRICS"]:
        steps.append(("principal_components", dict(n_components=5, mode="by_channel_local", whiten=True)))
    # The PCA metrics are computed afterwards, one unit at a time (pca_metrics()): SpikeInterface's
    # quality_metrics prepares the PCA data of every unit's neighbourhood at once, which needs
    # several GB for a few hundred units.
    steps.append(("quality_metrics", dict(metric_names=QUALITY_METRICS, metric_params=qm_params, seed=p["SEED"])))
    timings = {}
    import time

    for name, kw in steps:
        t = time.monotonic()
        analyzer.compute(name, **kw)
        timings[name] = round(time.monotonic() - t, 2)
        log.info("  %s: %.1fs", name, timings[name])
    return analyzer, win, timings


def _rss_mb() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


_PCA_STATE: dict = {}


def _pca_one_unit(u):
    """PCA metrics of one unit (the analyzer is inherited from the parent process, see pca_metrics)."""
    from spikeinterface.metrics.quality.pca_metrics import (
        d_prime_metric, mahalanobis_metrics, nearest_neighbors_metrics, simplified_silhouette_score)

    st = _PCA_STATE
    analyzer, main, unit_ids = st["analyzer"], st["main"], st["unit_ids"]
    if analyzer.is_sparse():
        chans = analyzer.sparsity.unit_id_to_channel_ids[u]
        neigh = [o for o in unit_ids if main[o] in chans]
    else:
        chans, neigh = analyzer.channel_ids, unit_ids
    proj, idx = st["pca_ext"].get_some_projections(channel_ids=chans, unit_ids=neigh)
    labels = analyzer.sorting.unit_ids[idx]
    pcs = proj.reshape(proj.shape[0], -1)
    np.random.seed(st["seed"])
    r = {}
    for names, fn in ((("isolation_distance", "l_ratio"), lambda: mahalanobis_metrics(pcs, labels, u)),
                      (("d_prime",), lambda: (d_prime_metric(pcs, labels, u) if len(neigh) > 1 else np.nan,)),
                      (("nn_hit_rate", "nn_miss_rate"),
                       lambda: nearest_neighbors_metrics(pcs, labels, u, st["max_spikes_nn"], st["n_neighbors"])),
                      (("silhouette",), lambda: (simplified_silhouette_score(pcs, labels, u),))):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vals = fn()
        except Exception:   # noqa: BLE001 - as SpikeInterface: a metric that fails is NaN
            vals = (np.nan,) * len(names)
        r.update(dict(zip(names, (float(v) for v in vals))))
    return u, r, len(neigh), pcs.shape


def pca_metrics(analyzer, max_spikes_nn: int = 10000, n_neighbors: int = 5, seed: int = 42,
                n_jobs: int = 1) -> pd.DataFrame:
    """isolation_distance, l_ratio, d_prime, nn_hit_rate, nn_miss_rate, silhouette (simplified) per unit.

    The same functions and the same neighbourhood as SpikeInterface's quality_metrics (units whose
    main channel lies within the unit's sparsity channels, projections on those channels), but
    computed one unit at a time, so memory stays at one neighbourhood per worker (SpikeInterface
    prepares every unit's neighbourhood at once: several GB for a few hundred units). With n_jobs > 1
    on Linux, units are spread over forked worker processes.
    """
    import multiprocessing as mp
    import sys
    import time
    from concurrent.futures import ProcessPoolExecutor

    unit_ids = list(analyzer.unit_ids)
    _PCA_STATE.update(analyzer=analyzer, main=analyzer.get_main_channels(outputs="id", with_dict=True),
                      unit_ids=unit_ids, pca_ext=analyzer.get_extension("principal_components"), seed=seed,
                      max_spikes_nn=max_spikes_nn, n_neighbors=n_neighbors)
    rows = {}
    t_last = time.monotonic()

    def collect(results):
        nonlocal t_last
        for k, (u, r, n_neigh, shape) in enumerate(results):
            rows[u] = r
            if time.monotonic() - t_last > 60 or k < 2:
                log.info("  PCA metrics: %d / %d units (%d neighbour units, %d x %d features, RSS %s MB)", k + 1,
                         len(unit_ids), n_neigh, shape[0], shape[1], _rss_mb())
                t_last = time.monotonic()

    try:
        if n_jobs > 1 and sys.platform.startswith("linux") and len(unit_ids) > 1:
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("fork")) as ex:
                collect(ex.map(_pca_one_unit, unit_ids, chunksize=4))
        else:
            collect(map(_pca_one_unit, unit_ids))
    finally:
        _PCA_STATE.clear()
    return pd.DataFrame.from_dict(rows, orient="index").reindex(unit_ids)


def run_bombcell(analyzer, thresholds: dict, win: dict, out_folder: Path | None, threads: int) -> pd.DataFrame:
    import bombcell

    params = bombcell.get_default_qc_params()
    params.update({"compute_distance_metrics": False, "plot_histograms": False, "plot_waveforms": False,
                   "plot_upset": False, "presence_ratio_bin_duration_s": win["presence_ratio_bin_duration_s"],
                   "drift_interval_s": win["drift_interval_s"], "drift_min_spikes": win["drift_min_spikes"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels, _, _ = bombcell.run_bombcell_qc(analyzer, output_folder=out_folder, params=params,
                                                thresholds=thresholds, n_jobs=threads, progress_bar=False)
    return labels


def bombcell_reasons(metrics: pd.DataFrame, thresholds: dict, labels: pd.Series) -> pd.Series:
    """Which thresholds each unit failed, e.g. 'mua: snr<5, presence_ratio<0.7' (for the table)."""
    out = []
    for uid in labels.index:
        lab = labels.loc[uid]
        sections = ["noise"] if lab == "noise" else ["mua"] if lab in ("mua",) else []
        if lab.startswith("non_soma"):
            sections = ["non-somatic"]
        fails = []
        for sec in sections:
            for m, t in thresholds.get(sec, {}).items():
                if m not in metrics.columns:
                    continue
                v = metrics.loc[uid, m]
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    if sec == "noise" or m == "sliding_rp_violation":
                        fails.append(f"{m}=nan")
                    continue
                v = abs(v) if t.get("abs") else v
                g, l_ = t.get("greater"), t.get("less")
                if sec == "non-somatic":
                    if (g is not None and v <= g) or (l_ is not None and v >= l_):
                        fails.append(f"{m}={v:.3g}")
                    continue
                if g is not None and not v > g:
                    fails.append(f"{m}<={g:g}")
                if l_ is not None and not v < l_:
                    fails.append(f"{m}>={l_:g}")
        out.append(f"{lab}: " + ", ".join(fails) if fails else "")
    return pd.Series(out, index=labels.index, name="bombcell_reasons")


def run_unitrefine(analyzer, models_dir: Path, extra_metrics: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """UnitRefine's cascade (as spikeinterface.curation.unitrefine_label_units), from local models.

    Metrics a model needs that were not computed (the PCA metrics with PCA_METRICS=false) are passed
    as missing values, which the models' own imputer fills; the returned list names them.
    """
    from sklearn.exceptions import InconsistentVersionWarning
    from spikeinterface.curation.model_based_curation import (
        ModelBasedClassification, _handle_backwards_compatibility_in_metrics, load_model)

    metrics_all = analyzer.get_metrics_extension_data()
    if extra_metrics is not None and len(extra_metrics.columns):
        metrics_all = metrics_all.join(extra_metrics.reindex(metrics_all.index), how="left")
    missing_all: set[str] = set()
    results = {}
    unit_ids = list(analyzer.unit_ids)
    for step in ("noise_neural", "sua_mua"):
        folder = models_dir / UNITREFINE_MODELS[step]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            warnings.simplefilter("ignore", UserWarning)
            model, info = load_model(model_folder=folder, trust_model=True)
            m = _handle_backwards_compatibility_in_metrics(metrics_all.copy(), model_info=info)
            missing = [c for c in model.feature_names_in_ if c not in m.columns]
            for c in missing:
                m[c] = np.nan
            missing_all.update(missing)
            ids = unit_ids if step == "noise_neural" else [u for u in unit_ids
                                                             if results["noise_neural"].loc[u, "prediction"] != "noise"]
            if not ids:
                results[step] = pd.DataFrame(columns=["prediction", "probability"])
                continue
            clf = ModelBasedClassification(sorting_analyzer=analyzer, unit_ids=ids, pipeline=model)
            res = clf.predict_labels(metrics=m.loc[ids], model_info=info, set_predictions_as_properties=False)
        results[step] = res
    nn, sm = results["noise_neural"], results["sua_mua"]
    lab, prob = [], []
    for u in unit_ids:
        if nn.loc[u, "prediction"] == "noise":
            lab.append("noise")
            prob.append(float(nn.loc[u, "probability"]))
        else:
            lab.append(str(sm.loc[u, "prediction"]))
            prob.append(float(sm.loc[u, "probability"]))
    df = pd.DataFrame({"unitrefine_label": lab, "unitrefine_probability": prob}, index=unit_ids)
    df["unitrefine_noise_probability"] = [
        float(nn.loc[u, "probability"]) if nn.loc[u, "prediction"] == "noise" else 1.0 - float(nn.loc[u, "probability"])
        for u in unit_ids]
    return df, sorted(missing_all)


def final_label(bombcell: str | None, unitrefine: str | None, rule: str) -> str:
    """The final good / mua / noise label.

    consensus (default):
      good  = Bombcell 'good' AND UnitRefine 'sua' (both call it a well-isolated single unit)
      noise = at least one method calls it noise, and neither calls it a single unit
              (Bombcell 'good' or UnitRefine 'sua')
      mua   = everything else: neural activity that is not an agreed single unit, Bombcell's
              non-somatic (axonal) units, and units where the two methods disagree about noise
              against a single-unit call (kept, not thrown away)
    bombcell:   good -> good, mua / non_soma -> mua, noise -> noise
    unitrefine: sua -> good, mua -> mua, noise -> noise
    When one method's label is missing, the other's is used alone.
    """
    b = bombcell if bombcell not in (None, "", "nan") else None
    u = unitrefine if unitrefine not in (None, "", "nan") else None
    from_b = {"good": "good", "mua": "mua", "noise": "noise"}.get(b, "mua") if b else None
    from_u = {"sua": "good", "mua": "mua", "noise": "noise"}.get(u, "mua") if u else None
    if rule == "bombcell" or from_u is None:
        return from_b or from_u or "mua"
    if rule == "unitrefine" or from_b is None:
        return from_u or from_b or "mua"
    if b == "good" and u == "sua":
        return "good"
    if (b == "noise" or u == "noise") and b != "good" and u != "sua":
        return "noise"
    return "mua"


def unit_table(analyzer, recording, metrics: pd.DataFrame) -> pd.DataFrame:
    """Per unit: peak channel, brain area, location, depth, amplitude, and the metrics."""
    import spikeinterface.full as si

    unit_ids = list(analyzer.unit_ids)
    peak_ch = si.get_template_extremum_channel(analyzer, peak_sign="neg", outputs="id")
    ch_ids = list(recording.channel_ids)
    ch_index = {c: i for i, c in enumerate(ch_ids)}
    areas = recording.get_property("brain_area")
    locs = analyzer.get_extension("unit_locations").get_data()
    ch_pos = recording.get_channel_locations()
    rows = []
    for i, u in enumerate(unit_ids):
        pc = peak_ch[u]
        ci = ch_index[pc]
        rows.append({
            "unit_id": u, "peak_channel": str(pc), "peak_channel_index": int(ci),
            "brain_area": (str(areas[ci]) if areas is not None and str(areas[ci]) not in ("", "nan", "None") else ""),
            "x": float(locs[i, 0]), "y": float(locs[i, 1]),
            "z": float(locs[i, 2]) if locs.shape[1] > 2 else np.nan,
            "depth": float(locs[i, 1]),
            "peak_channel_x": float(ch_pos[ci, 0]), "peak_channel_y": float(ch_pos[ci, 1]),
        })
    df = pd.DataFrame(rows).set_index("unit_id", drop=False)
    m = metrics.reindex(df.index)
    m = m[[c for c in m.columns if c not in df.columns]]
    return pd.concat([df, m], axis=1)


def peak_waveforms(analyzer, table: pd.DataFrame, half: int = 40) -> np.ndarray:
    """Mean waveform on the peak channel, +-half samples around the trough (uV): (units, 2*half+1)."""
    tmpl_ext = analyzer.get_extension("templates")
    templates = tmpl_ext.get_templates(operator="average")          # dense (units, samples, channels)
    nbefore = tmpl_ext.nbefore
    out = np.full((len(table), 2 * half + 1), np.nan, dtype="float32")
    for i, ci in enumerate(table["peak_channel_index"].to_numpy()):
        w = templates[i, :, int(ci)]
        a, b = nbefore - half, nbefore + half + 1
        seg = w[max(a, 0):min(b, len(w))]
        out[i, max(0, -a):max(0, -a) + len(seg)] = seg
    return out


def kslabels(sorting, sorter_output: Path | None, unit_ids) -> list[str] | None:
    """Kilosort's own good/mua label, from the sorting property or cluster_KSLabel.tsv."""
    if "KSLabel" in sorting.get_property_keys():
        return [str(v) for v in sorting.get_property("KSLabel")]
    if sorter_output is not None and (sorter_output / "cluster_KSLabel.tsv").is_file():
        df = pd.read_csv(sorter_output / "cluster_KSLabel.tsv", sep="\t")
        m = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
        if any(str(u) in m for u in unit_ids):
            return [m.get(str(u), "") for u in unit_ids]
    return None


def write_json(path: Path, data) -> None:
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
        fh.write("\n")
