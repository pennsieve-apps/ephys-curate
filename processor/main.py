"""
ephys-curate: which sorted units are real, well-isolated neurons?

Inputs (conventions.md, "Spike sorting"): `recording` (ephys-prep: preprocessed SpikeInterface
binary_folder + session.json), `sorting` (spike-sort: numpy_folder + sort.json, or a Kilosort / Phy
folder), optional `reference` (an NWB Units table or spike CSV to compare with). Sessions are
matched on session_id and processed one at a time.

Output (one port `units`, the whole of OUTPUT_DIR), all under ephys-curate/:
  <session_id>_units.nwb        NWB Units table (spike_times in session seconds, labels, metrics)
  units.csv                     the same columns for every session, without spike times
  <session_id>_analyzer.zarr    the SpikeInterface SortingAnalyzer (without the raw waveforms)
  <session_id>_bombcell/        Bombcell's per-metric pass/fail tables and thresholds
  <session_id>_phy/             with EXPORT_PHY=true
  <session_id>_*.png            figures; <session_id>_reference_*.csv with a reference
  sessions.csv, summary.json, provenance.json
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import platform
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
log = logging.getLogger("ephys-curate")
log.setLevel(logging.INFO)
for noisy in ("matplotlib", "PIL", "fontTools", "numba", "h5py", "hdmf", "pynwb", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "ephys-curate"
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "LABEL_RULE": "consensus",
    "PCA_METRICS": "true",
    "UNITREFINE": "true",
    "BOMBCELL_THRESHOLDS": "",
    "SPARSITY_RADIUS_UM": "100",
    "MAX_SPIKES_PER_UNIT": "500",
    "EXPORT_PHY": "false",
    "GALLERY_UNITS": "12",
    "REFERENCE_KIND": "auto",
    "MATCH_DELTA_MS": "0.4",
    "REFERENCE_MIN_SPIKES": "10",
    "SEED": "42",
    "THREADS": "0",
}
PARAMETERS = tuple(DEFAULTS)
CITATIONS_BASE = ["10.7554/eLife.61834", "10.5281/zenodo.8172821", "10.1101/2025.03.30.645770",
                  "10.7554/eLife.78362"]


class ParameterError(ValueError):
    pass


def _bool(name: str, v: str) -> bool:
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off"):
        return False
    raise ParameterError(f"{name} must be true or false, got {v!r}")


def read_parameters(env=os.environ) -> dict:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p: dict = {}
    p["LABEL_RULE"] = raw["LABEL_RULE"].strip().lower()
    if p["LABEL_RULE"] not in ("consensus", "bombcell", "unitrefine"):
        raise ParameterError("LABEL_RULE must be consensus, bombcell or unitrefine")
    p["REFERENCE_KIND"] = raw["REFERENCE_KIND"].strip().lower().replace("-", "_")
    if p["REFERENCE_KIND"] not in ("auto", "sorting", "ground_truth"):
        raise ParameterError("REFERENCE_KIND must be auto, sorting or ground_truth")
    for k in ("PCA_METRICS", "UNITREFINE", "EXPORT_PHY"):
        p[k] = _bool(k, raw[k])
    if p["LABEL_RULE"] == "unitrefine" and not p["UNITREFINE"]:
        raise ParameterError("LABEL_RULE=unitrefine needs UNITREFINE=true")
    try:
        for k in ("MAX_SPIKES_PER_UNIT", "GALLERY_UNITS", "SEED", "THREADS", "REFERENCE_MIN_SPIKES"):
            p[k] = int(raw[k])
        for k in ("SPARSITY_RADIUS_UM", "MATCH_DELTA_MS"):
            p[k] = float(raw[k])
    except ValueError as exc:
        raise ParameterError(f"bad numeric parameter: {exc}") from exc
    if p["MAX_SPIKES_PER_UNIT"] < 50 or p["GALLERY_UNITS"] < 0 or p["THREADS"] < 0 or p["REFERENCE_MIN_SPIKES"] < 1 \
            or p["SPARSITY_RADIUS_UM"] <= 0 or p["MATCH_DELTA_MS"] <= 0:
        raise ParameterError("MAX_SPIKES_PER_UNIT >= 50, GALLERY_UNITS/THREADS >= 0, "
                             "SPARSITY_RADIUS_UM and MATCH_DELTA_MS > 0")
    bt = raw["BOMBCELL_THRESHOLDS"].strip()
    try:
        p["BOMBCELL_THRESHOLDS"] = json.loads(bt) if bt else {}
    except ValueError as exc:
        raise ParameterError(f"BOMBCELL_THRESHOLDS is not valid JSON: {exc}") from exc
    if not isinstance(p["BOMBCELL_THRESHOLDS"], dict):
        raise ParameterError("BOMBCELL_THRESHOLDS must be a JSON object")
    return p


def cgroup_cpus() -> int:
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, math.floor(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        per = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0 and per > 0:
            return max(1, q // per)
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def set_threads(threads: int) -> int:
    n = threads if threads > 0 else cgroup_cpus()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")      # SpikeInterface parallelises over processes
    if "numba" not in sys.modules:
        os.environ["NUMBA_NUM_THREADS"] = str(n)
    return n


def peak_rss_mb() -> float | None:
    try:
        import resource

        div = 1024 * 1024 if sys.platform == "darwin" else 1024
        own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / div
        kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / div
        return round(max(own, kids), 1)
    except (ImportError, OSError):
        return None


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version()}
    for pkg in ("spikeinterface", "bombcell", "scikit-learn", "skops", "pynwb", "hdmf", "numpy", "pandas",
                "scipy", "numba", "zarr", "probeinterface", "matplotlib"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def _clean(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return _clean(obj.item())
        except (ValueError, AttributeError):
            return str(obj)
    return obj


def write_json(path: Path, data) -> None:
    with open(path, "w") as fh:
        json.dump(_clean(data), fh, indent=2, default=str)
        fh.write("\n")


def models_dir() -> Path | None:
    for c in (os.environ.get("UNITREFINE_MODELS_DIR"), "/opt/unitrefine", str(REPO_ROOT / "models")):
        if c and (Path(c) / "UnitRefine_noise_neural_classifier").is_dir():
            return Path(c)
    return None


def session_times(recording, session: dict, sid: str, warnings_out: list):
    """Sample times on the session clock: the recording's own times (its timestamps or t_start).

    A safety net: SpikeInterface drops a time vector when a preprocessed (filtered) recording is
    saved, so a saved recording can come back starting at 0 s. When session.json says the window
    starts elsewhere, the times are rebuilt from session.json's t_start and t_stop (a linear map,
    i.e. the effective sampling rate of the original timestamps), with a warning.
    """
    import numpy as np

    times = recording.get_times()
    n = len(times)
    if recording.has_time_vector():
        return times, "recording timestamps"
    t0, t1 = session.get("t_start"), session.get("t_stop")
    fs = recording.sampling_frequency
    if t0 is not None and abs(float(times[0]) - float(t0)) > 1.0 / fs:
        if t1 is not None and float(t1) > float(t0) and n > 1:
            times = np.linspace(float(t0), float(t1), n)
            how = "session.json t_start..t_stop (linear)"
        else:
            times = float(t0) + np.arange(n) / fs
            how = "session.json t_start + n / fs"
        msg = (f"{sid}: the recording starts at {recording.get_times()[0]:.3f} s but session.json says "
               f"t_start = {float(t0):.3f} s; spike times use {how}. ephys-prep should save the recording "
               "with its times")
        log.warning(msg)
        warnings_out.append(msg)
        return times, how
    return times, "recording t_start + n / fs"


class Heartbeat:
    """Logs 'still working' at least every minute while a long step runs."""

    def __init__(self, what: str, every: float = 60.0):
        self.what, self.every, self.t0 = what, every, time.monotonic()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.every):
            log.info("%s: still working (%.0f s)", self.what, time.monotonic() - self.t0)

    def __enter__(self):
        self._th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()


CAPTIONS = {
    "unit_yield": ("Unit yield. Left: how many sorted units fall in each brain region (by the channel where "
                   "each unit is largest), coloured by the final label: good = well-isolated single neurons "
                   "(Bombcell and UnitRefine agree), mua = multi-unit activity or units the methods disagree on, "
                   "noise = not neural. Right: Bombcell's label against UnitRefine's for the same units; counts "
                   "on the diagonal are units the two methods agree on."),
    "metrics": ("Quality metrics of every unit, stacked by final label. Dashed lines are Bombcell's thresholds: a "
                "unit outside a noise threshold (waveform shape) is noise, outside a mua threshold (amplitude, "
                "SNR, refractory-period contamination, presence, drift) is multi-unit. Good units should sit on "
                "the passing side of every line; look for a metric where many units fall just outside."),
    "gallery": ("The best good units (highest SNR). Per unit: the mean waveform on its peak channel (coloured) "
                "and three neighbouring channels (grey); the autocorrelogram, where a single neuron shows an "
                "empty dip in the shaded +-1.5 ms refractory window; and spike amplitude over the session, which "
                "should be steady (a drop to a hard floor means spikes are being lost, a drift means the probe "
                "moved)."),
    "depth": ("Firing rate against position along the probe, one dot per unit (size = amplitude), with the brain "
              "regions of the channels on the right. Look for where good units cluster (cell layers), for regions "
              "with no units, and for bands of noise units (often a noisy or out-of-brain stretch of the probe)."),
    "agreement": ("Agreement with the reference. Left: the fraction of our good, mua and noise units whose spikes "
                  "match a reference unit (agreement >= 0.5; within {delta} ms): good units should match far more "
                  "often than noise. Right: for each reference unit in the window, the best agreement with any of "
                  "our units ({kind})."),
}


def main() -> int:
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 1
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        p = read_parameters()
    except ParameterError as exc:
        log.error("%s", exc)
        return 2
    threads = set_threads(p["THREADS"])

    import numpy as np
    import pandas as pd
    import spikeinterface.full as si

    from processor import curate as C
    from processor import figures as F
    from processor import reference as R
    from processor.inputs import discover
    from processor.nwb import write_units_nwb

    try:
        thresholds = C.bombcell_thresholds(p["BOMBCELL_THRESHOLDS"])
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    si.set_global_job_kwargs(n_jobs=threads, chunk_duration="1s", progress_bar=False)
    np.random.seed(p["SEED"])

    found = discover(input_dir)
    for name, reason in found.skipped.items():
        log.warning("%s: skipped, %s", name, reason)
    sessions = sorted(set(found.recordings) | set(found.sortings))
    if not found.recordings or not found.sortings:
        log.error("need a recording (ephys-prep) and a sorting (spike-sort or Kilosort) per session; found "
                  "%d recordings and %d sortings under %s", len(found.recordings), len(found.sortings), input_dir)
        return 1
    mdir = models_dir() if p["UNITREFINE"] else None
    warnings_out = list(found.warnings)
    if p["UNITREFINE"] and mdir is None:
        msg = "UnitRefine models not found (UNITREFINE_MODELS_DIR, /opt/unitrefine, models/): Bombcell labels only"
        log.warning(msg)
        warnings_out.append(msg)
    log.info("%d sessions; label rule %s, PCA metrics %s, UnitRefine %s, %d threads", len(sessions),
             p["LABEL_RULE"], p["PCA_METRICS"], bool(mdir), threads)

    top = output_dir / APP_NAME
    top.mkdir(parents=True, exist_ok=True)
    work_root = Path(os.environ.get("TMPDIR", "/tmp")) / f"{APP_NAME}-work"
    statuses: dict = {}
    summaries: dict = {}
    figures: list = []
    tables: list = []
    all_units = []
    sheet_rows = []
    inputs_used = []
    citations = list(CITATIONS_BASE)

    for sid in sessions:
        ts = time.monotonic()
        rec_in, sort_in = found.recordings.get(sid), found.sortings.get(sid)
        if rec_in is None or sort_in is None:
            why = "no recording" if rec_in is None else "no sorting"
            log.warning("%s: skipped, %s for this session", sid, why)
            statuses[sid] = {"status": "skipped", "reason": why, "seconds": 0.0}
            warnings_out.append(f"{sid}: {why}")
            continue
        sheet = found.sheet.get(sid, {})
        try:
            recording = si.load(rec_in.folder)
            if sort_in.kind == "kilosort":
                sorting = si.read_kilosort(sort_in.folder, keep_good_only=False)
            else:
                sorting = si.load(sort_in.folder)
            inputs_used += [str(rec_in.folder.relative_to(input_dir)), str(sort_in.folder.relative_to(input_dir))]
            if rec_in.session_json:
                inputs_used.append(str(rec_in.session_json.relative_to(input_dir)))
            if abs(sorting.sampling_frequency - recording.sampling_frequency) > 1e-3 * recording.sampling_frequency:
                raise ValueError(f"sorting at {sorting.sampling_frequency} Hz, recording at "
                                 f"{recording.sampling_frequency} Hz: not the same recording")
            n_samples = recording.get_num_samples()
            last = max((int(sorting.get_unit_spike_train(u)[-1]) for u in sorting.unit_ids
                        if len(sorting.get_unit_spike_train(u))), default=0)
            if last >= n_samples + int(0.01 * recording.sampling_frequency):
                raise ValueError(f"spikes up to sample {last} but the recording has {n_samples} samples: the "
                                 "sorting was made on a different (longer) recording")
            sorting = si.remove_excess_spikes(sorting, recording)
            counts = sorting.count_num_spikes_per_unit()
            empty = [u for u, n in counts.items() if n < 2]
            if empty:
                log.info("%s: %d units with fewer than 2 spikes dropped", sid, len(empty))
                sorting = sorting.remove_units(empty)
            if sorting.get_num_units() == 0:
                raise ValueError("no units with spikes")
            ks = C.kslabels(sorting, sort_in.sorter_output, sorting.unit_ids)
            duration = recording.get_duration()
            times, time_source = session_times(recording, rec_in.session, sid, warnings_out)
            log.info("%s: %d units, %d channels, %.1f s (%.3f-%.3f s on the session clock), sorting %s (%s)",
                     sid, sorting.get_num_units(), recording.get_num_channels(), duration, times[0], times[-1],
                     sort_in.kind, sort_in.sort.get("sorter", "?"))

            sess_thr, thr_note = C.session_thresholds(thresholds, duration, p["BOMBCELL_THRESHOLDS"])
            if thr_note:
                log.info("%s: %s", sid, thr_note)
            work = work_root / sid
            with Heartbeat(f"{sid}: analyzer"):
                analyzer, win, timings = C.build_analyzer(sorting, recording, work, p, duration)
            t_b = time.monotonic()
            with Heartbeat(f"{sid}: bombcell"):
                bc = C.run_bombcell(analyzer, sess_thr, win, top / f"{sid}_bombcell", threads)
            timings["bombcell"] = round(time.monotonic() - t_b, 2)
            metrics = analyzer.get_metrics_extension_data()
            ur, ur_missing = None, []
            if mdir is not None:
                t_u = time.monotonic()
                try:
                    ur, ur_missing = C.run_unitrefine(analyzer, mdir)
                    if ur_missing:
                        msg = (f"{sid}: UnitRefine ran without {', '.join(ur_missing)} (imputed): "
                               "set PCA_METRICS=true for the metrics the models were trained on")
                        log.warning(msg)
                        warnings_out.append(msg)
                except Exception as exc:   # noqa: BLE001 - fall back to Bombcell
                    msg = f"{sid}: UnitRefine failed ({exc}); Bombcell labels only"
                    log.warning(msg)
                    warnings_out.append(msg)
                timings["unitrefine"] = round(time.monotonic() - t_u, 2)

            table = C.unit_table(analyzer, recording, metrics)
            table.insert(0, "session_id", sid)
            table.insert(1, "unit_name", [str(u) for u in table.index])
            table["bombcell_label"] = bc["bombcell_label"].reindex(table.index).astype(str)
            table["bombcell_reasons"] = C.bombcell_reasons(metrics, sess_thr, bc["bombcell_label"]).reindex(
                table.index).fillna("")
            if ur is not None:
                for c in ur.columns:
                    table[c] = ur[c].reindex(table.index)
            else:
                table["unitrefine_label"] = ""
            table["label"] = [C.final_label(b, u if isinstance(u, str) else None, p["LABEL_RULE"])
                              for b, u in zip(table["bombcell_label"], table["unitrefine_label"])]
            if ks is not None:
                table["kslabel"] = ks
            for k in ("subject_id", "condition", "group"):
                if sheet.get(k):
                    table[k] = sheet[k]

            # spike times on the session clock (the recording's own times)
            spike_times = {}
            for u in table.index:
                fr = sorting.get_unit_spike_train(u)
                spike_times[u] = times[np.clip(fr, 0, len(times) - 1)]

            # reference / ground truth
            ref_summary = None
            ref_in = found.references.get(sid)
            if ref_in is not None:
                try:
                    if ref_in.kind == "nwb":
                        trains, rinfo = R.read_reference_nwb(ref_in.path, R.probe_hint(rec_in.session))
                    else:
                        trains, rinfo = R.read_reference_csv(ref_in.path)
                    kind = R.reference_kind(rinfo, ref_in.path, p["REFERENCE_KIND"])
                    ref_sorting, _ = R.to_sorting(trains, recording, p["REFERENCE_MIN_SPIKES"], times)
                    if ref_sorting is None:
                        raise ValueError(f"no reference unit with {p['REFERENCE_MIN_SPIKES']}+ spikes inside the "
                                         "recording's window")
                    rtab, ref_summary = R.compare(sorting, table["label"], ref_sorting, kind, p["MATCH_DELTA_MS"])
                    ref_summary["file"] = rinfo
                    table["reference_match"] = rtab["reference_match"].reindex(table.index).fillna("")
                    table["reference_agreement"] = rtab["reference_agreement"].reindex(table.index).fillna(0.0)
                    ref_summary["_reference_table"].to_csv(top / f"{sid}_reference_units.csv", index=False)
                    tables.append(f"{sid}_reference_units.csv")
                    if "_performance" in ref_summary:
                        ref_summary["_performance"].to_csv(top / f"{sid}_groundtruth_performance.csv",
                                                           index_label="groundtruth_unit")
                        tables.append(f"{sid}_groundtruth_performance.csv")
                    inputs_used.append(str(ref_in.path.relative_to(input_dir)))
                    log.info("%s: reference %s (%s): %d reference units in the window, %d of our units matched "
                             "(good %d/%d)%s", sid, ref_in.path.name, kind, ref_summary["n_reference_units"],
                             ref_summary["n_matched"], ref_summary["matched_by_label"]["good"]["matched"],
                             ref_summary["matched_by_label"]["good"]["n"],
                             f", {ref_summary['n_well_detected']} well detected" if kind == "ground_truth" else "")
                except Exception as exc:   # noqa: BLE001 - the comparison is optional
                    msg = f"{sid}: reference comparison failed: {exc}"
                    log.warning(msg)
                    warnings_out.append(msg)
                    ref_summary = None

            # NWB
            waveforms = C.peak_waveforms(analyzer, table)
            nwb_path = top / f"{sid}_units.nwb"
            write_units_nwb(nwb_path, sid, rec_in.session, recording, table,
                            [spike_times[u] for u in table.index], waveforms, (float(times[0]), float(times[-1])),
                            app_version(), sheet)

            # figures
            sess_figs = []
            def fig(kind, fn, *a):
                path = top / f"{F.safe_name(sid)}_{kind}.png"
                try:
                    out = fn(sid, *a, path)
                    if out is not None:
                        cap = CAPTIONS[{"unit_yield": "unit_yield", "metrics": "metrics", "unit_gallery": "gallery",
                                        "depth_map": "depth", "agreement": "agreement"}[kind]]
                        if kind == "agreement":
                            cap = cap.format(delta=p["MATCH_DELTA_MS"],
                                             kind="ground truth: how well each true neuron was recovered"
                                             if ref_summary["kind"] == "ground_truth" else "reference sorting")
                        sess_figs.append({"path": f"{APP_NAME}/{path.name}", "caption": cap})
                except Exception as exc:   # noqa: BLE001 - a figure is a convenience
                    log.exception("%s: figure %s failed: %s", sid, kind, exc)
            fig("unit_yield", F.unit_yield, table)
            fig("metrics", F.metric_distributions, table, sess_thr)
            if p["GALLERY_UNITS"] > 0:
                fig("unit_gallery", F.unit_gallery, table, analyzer, spike_times, p["GALLERY_UNITS"])
            fig("depth_map", F.depth_map, table, recording.get_channel_locations()[:, 1],
                recording.get_property("brain_area"))
            if ref_summary is not None:
                fig("agreement", F.agreement, table, ref_summary)
            figures.extend(sess_figs)

            # Phy, analyzer
            if p["EXPORT_PHY"]:
                with Heartbeat(f"{sid}: phy export"):
                    si.export_to_phy(analyzer, top / f"{sid}_phy", compute_pc_features=False,
                                     compute_amplitudes=True, copy_binary=False, remove_if_exists=True,
                                     verbose=False)
            # The raw waveform snippets (GBs) and the PCA models (pickled sklearn objects, which zarr
            # will not load without allowing unsafe pickles) are not kept; their results are.
            for ext in ("waveforms", "principal_components"):
                if analyzer.has_extension(ext):
                    analyzer.delete_extension(ext)
            zpath = top / f"{sid}_analyzer.zarr"
            if zpath.exists():
                shutil.rmtree(zpath)
            analyzer.save_as(format="zarr", folder=zpath)
            del analyzer
            shutil.rmtree(work, ignore_errors=True)

            counts = table["label"].value_counts().to_dict()
            bcc = table["bombcell_label"].value_counts().to_dict()
            urc = table["unitrefine_label"].replace("", np.nan).dropna().value_counts().to_dict()
            agree = None
            if ur is not None:
                b3 = table["bombcell_label"].map(lambda b: {"good": "good", "noise": "noise"}.get(b, "mua"))
                u3 = table["unitrefine_label"].map({"sua": "good", "mua": "mua", "noise": "noise"})
                agree = round(float((b3 == u3).mean()), 4)
            good = table[table["label"] == "good"]
            regions = table.assign(area=table["brain_area"].replace("", "unknown")).groupby("area")["label"] \
                .value_counts().unstack(fill_value=0)
            s = {
                "bombcell_thresholds_note": thr_note or None,
                "n_units": int(len(table)), "n_good": int(counts.get("good", 0)), "n_mua": int(counts.get("mua", 0)),
                "n_noise": int(counts.get("noise", 0)), "bombcell": {k: int(v) for k, v in bcc.items()},
                "unitrefine": {k: int(v) for k, v in urc.items()}, "bombcell_unitrefine_agreement": agree,
                "label_rule": p["LABEL_RULE"], "duration_s": round(duration, 3),
                "t_start": float(times[0]), "t_stop": float(times[-1]), "time_source": time_source,
                "n_channels": int(recording.get_num_channels()),
                "sorter": sort_in.sort.get("sorter") or ("kilosort" if sort_in.kind == "kilosort" else None),
                "sorting_format": sort_in.kind,
                "median_snr_good": float(good["snr"].median()) if len(good) else None,
                "median_firing_rate_good": float(good["firing_rate"].median()) if len(good) else None,
                "units_by_region": {a: {k: int(v) for k, v in r.items()} for a, r in regions.iterrows()},
                "nwb": f"{APP_NAME}/{nwb_path.name}",
            }
            if ks is not None:
                s["kilosort_good"] = int(sum(1 for k in ks if k == "good"))
            if ref_summary is not None:
                s["reference"] = {k: v for k, v in ref_summary.items() if not k.startswith("_")}
                if ref_summary["kind"] == "ground_truth":
                    citations.append("10.1101/2024.12.04.626805") if "001250" in str(ref_in.path) else None
            for k in ("subject_id", "condition", "group"):
                if sheet.get(k):
                    s[k] = sheet[k]
            summaries[sid] = s
            all_units.append(table)
            secs = round(time.monotonic() - ts, 2)
            statuses[sid] = {"status": "ok", "seconds": secs, "timings": timings,
                             "unitrefine_missing_metrics": ur_missing}
            sheet_rows.append({"session_id": sid, "path": str(rec_in.folder.relative_to(input_dir)),
                               "subject_id": sheet.get("subject_id", (rec_in.session.get("subject") or {}).get(
                                   "subject_id", "") if isinstance(rec_in.session.get("subject"), dict) else ""),
                               "condition": sheet.get("condition", ""), "group": sheet.get("group", ""),
                               "status": "ok", "n_units": s["n_units"], "n_good": s["n_good"], "n_mua": s["n_mua"],
                               "n_noise": s["n_noise"]})
            log.info("%s: %d units -> good %d, mua %d, noise %d (Bombcell %s; UnitRefine %s), %.1fs, peak RSS %s MB",
                     sid, s["n_units"], s["n_good"], s["n_mua"], s["n_noise"], s["bombcell"], s["unitrefine"],
                     secs, peak_rss_mb())
        except Exception as exc:   # one bad session is skipped, not fatal
            log.exception("%s: failed: %s", sid, exc)
            statuses[sid] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                             "seconds": round(time.monotonic() - ts, 2)}
            sheet_rows.append({"session_id": sid, "path": str(rec_in.folder.relative_to(input_dir)),
                               "subject_id": sheet.get("subject_id", ""), "condition": sheet.get("condition", ""),
                               "group": sheet.get("group", ""), "status": "failed", "n_units": "", "n_good": "",
                               "n_mua": "", "n_noise": ""})
            shutil.rmtree(work_root / sid, ignore_errors=True)

    ok = [s for s, v in statuses.items() if v["status"] == "ok"]
    if all_units:
        units = pd.concat(all_units, ignore_index=True)
        units = units.drop(columns=["peak_channel_index", "unit_id"], errors="ignore")
        units.to_csv(top / "units.csv", index=False, float_format="%.6g")
        tables.insert(0, "units.csv")
    if sheet_rows:
        with open(top / "sessions.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sheet_rows[0].keys()))
            w.writeheader()
            w.writerows(sheet_rows)
        tables.append("sessions.csv")
    write_json(top / "bombcell_thresholds.json", thresholds)
    write_json(top / "summary.json", {
        "app": APP_NAME, "version": app_version(), "sessions": summaries,
        "figures": figures, "tables": [f"{APP_NAME}/{t}" for t in tables], "warnings": warnings_out,
    })
    write_json(top / "provenance.json", {
        "app": APP_NAME, "version": app_version(), "variant": "cpu", "tools": tool_versions(),
        "parameters": p, "inputs": sorted(set(inputs_used)),
        "started": started.isoformat().replace("+00:00", "Z"),
        "finished": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_seconds": round(time.monotonic() - t0, 2), "peak_rss_mb": peak_rss_mb(), "gpu": None,
        "sessions": statuses, "citations": sorted(set(c for c in citations if c)),
    })
    shutil.rmtree(work_root, ignore_errors=True)
    if not ok:
        log.error("every session failed")
        return 1
    log.info("done: %d of %d sessions, %.1fs, peak RSS %s MB -> %s", len(ok), len(statuses),
             time.monotonic() - t0, peak_rss_mb(), top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
