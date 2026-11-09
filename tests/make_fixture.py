"""
Writes a small synthetic spike-sorting session laid out as ephys-prep and spike-sort write it, with
units of known quality, so the tests (and a review run) can check that curation finds them.

    python -m tests.make_fixture OUT [--layout linked|flat|both] [--engine auto|spikeinterface|numpy]
                                     [--duration 60] [--kilosort]

The session (sub-synth01_ses-a1b2c3d4) is a 32-channel two-column probe (20 um pitch) at 30 kHz,
starting at t = 600 s on the session clock (the recording is a window, as ephys-prep's T_START
makes it). Channels with y < 160 um are in "CA1", the rest in "VISp" (channel property
brain_area). The sorting has:

  units 0-7   eight ground-truth units, spike times as simulated              -> should be good
  unit  8     ground-truth unit 8 plus 40 % random extra spikes (contaminated) -> should be mua
  units 9-11  random spike times with no spike in the signal                   -> should be noise
  unit  12    every other spike of ground-truth unit 7 (unit 7 keeps the rest): an over-split
              unit                                                             -> merged back into 7

The reference (ground truth) is every simulated unit's spike times, as a CSV
(<session>_groundtruth_spikes.csv: unit_id, spike_time in session seconds); the tests also write
it as an NWB Units table with pynwb.

The recording is simulated with SpikeInterface's generate_ground_truth_recording when SpikeInterface
is installed (tests, `make fixture`), else with a small numpy simulator (the platform's review
generator image has numpy but not SpikeInterface). Either way the files are written here by hand in
SpikeInterface's binary_folder / numpy_folder formats, so this script needs only numpy.

Layouts: linked/ = one folder per input port (recording/, sorting/, reference/), as a workflow
delivers them; flat/ = everything side by side.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

SESSION_ID = "sub-synth01_ses-a1b2c3d4"
FS = 30000.0
T_START = 600.0
N_GT = 9
GOOD_UNITS = list(range(8))
MUA_UNITS = [8]
NOISE_UNITS = [9, 10, 11]
SPLIT_UNIT = (7, 12)        # unit 7 over-split into 7 and 12
GAIN_UV = 0.195            # uV per bit of the int16 traces (as an Intan/NP2-like gain)
SI_VERSION = "0.105.4"


def probe_positions(n_channels: int = 32) -> np.ndarray:
    half = n_channels // 2
    return np.array([[0.0, 20.0 * i] for i in range(half)] + [[20.0, 20.0 * i] for i in range(half)])


def brain_areas(pos: np.ndarray) -> np.ndarray:
    return np.where(pos[:, 1] < 160.0, "CA1", "VISp")


# --------------------------------------------------------------------------- simulation


def simulate_spikeinterface(duration: float, seed: int):
    import spikeinterface.full as si

    rec, gt = si.generate_ground_truth_recording(
        durations=[duration], sampling_frequency=FS, num_channels=32, num_units=N_GT, seed=seed,
        generate_probe_kwargs=dict(num_columns=2, num_contact_per_column=16, xpitch=20.0, ypitch=20.0,
                                   contact_shapes="circle", contact_shape_params={"radius": 6}),
        generate_sorting_kwargs=dict(firing_rates=(12.0, 20.0), refractory_period_ms=2.0),
        generate_unit_locations_kwargs=dict(margin_um=10.0, minimum_z=10.0, maximum_z=40.0,
                                            minimum_distance=25.0, max_iteration=200, distance_strict=False),
        noise_kwargs=dict(noise_levels=7.5, strategy="on_the_fly"),
    )
    traces = rec.get_traces(return_in_uV=False).astype(np.float32)    # uV
    pos = rec.get_channel_locations()
    trains = [gt.get_unit_spike_train(u) for u in gt.unit_ids]
    return traces, pos, trains


def simulate_numpy(duration: float, seed: int):
    rng = np.random.default_rng(seed)
    n = int(duration * FS)
    pos = probe_positions(32)
    nch = len(pos)
    traces = rng.normal(0.0, 7.5, size=(n, nch)).astype(np.float32)
    t = np.arange(-60, 90) / FS * 1000.0                                   # ms, 5 ms
    trains = []
    ys = np.linspace(30.0, 290.0, N_GT)
    for u in range(N_GT):
        rate = rng.uniform(12.0, 20.0)
        isi = rng.exponential(1.0 / rate, size=int(rate * duration * 2) + 20) + 0.002
        st = np.cumsum(isi)
        st = st[st < duration - 0.01]
        frames = np.unique((st * FS).astype(np.int64))
        frames = frames[frames > 100]
        trains.append(frames)
        loc = np.array([rng.uniform(0, 20), ys[u], rng.uniform(15, 35)])
        width = rng.uniform(0.25, 0.45)
        wf = -np.exp(-0.5 * (t / (width * 0.6)) ** 2) + 0.35 * np.exp(-0.5 * ((t - 0.6) / 0.35) ** 2)
        wf[t < -1.2] = 0
        amp = rng.uniform(120.0, 220.0)
        d = np.sqrt(((pos - loc[:2]) ** 2).sum(1) + loc[2] ** 2)
        gains = amp * (loc[2] / d) ** 2
        tmpl = wf[:, None] * gains[None, :]
        for f in frames:
            a, b = f - 60, f + 90
            if a >= 0 and b <= n:
                traces[a:b] += tmpl
    return traces, pos, trains


# --------------------------------------------------------------------------- writers (SpikeInterface folder formats)


def _dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_probegroup(folder: Path, pos: np.ndarray) -> None:
    n = len(pos)
    probe = {
        "ndim": 2, "si_units": "um", "annotations": {"model_name": "synthetic 2-column", "manufacturer": ""},
        "contact_annotations": {},
        "contact_positions": pos.tolist(),
        "contact_plane_axes": [[[1.0, 0.0], [0.0, 1.0]]] * n,
        "contact_shapes": ["circle"] * n,
        "contact_shape_params": [{"radius": 6.0}] * n,
        "probe_planar_contour": [[-20.0, float(pos[:, 1].max() + 20)], [-20.0, -20.0], [10.0, -100.0],
                                 [40.0, -20.0], [40.0, float(pos[:, 1].max() + 20)]],
        "device_channel_indices": list(range(n)),
        "contact_ids": [str(i) for i in range(n)],
        "shank_ids": [""] * n,
    }
    _dump(folder / "probegroup.json", {"specification": "probeinterface", "version": "0.4.2",
                                       "probes": [probe], "probe_ids": ["0"]})


def write_binary_folder(folder: Path, traces_uv: np.ndarray, pos: np.ndarray, areas: np.ndarray) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    n_ch = traces_uv.shape[1]
    data = np.clip(np.round(traces_uv / GAIN_UV), -32768, 32767).astype("<i2")
    data.tofile(folder / "traces_cached_seg0.raw")
    ids = [str(i) for i in range(n_ch)]
    gains, offsets = [GAIN_UV] * n_ch, [0.0] * n_ch
    _dump(folder / "binary.json", {
        "class": "spikeinterface.core.binaryrecordingextractor.BinaryRecordingExtractor",
        "module": "spikeinterface", "version": SI_VERSION,
        "kwargs": {"file_paths": ["traces_cached_seg0.raw"], "sampling_frequency": FS, "t_starts": [T_START],
                   "num_channels": n_ch, "dtype": "<i2", "channel_ids": ids, "time_axis": 0, "file_offset": 0,
                   "gain_to_uV": gains, "offset_to_uV": offsets, "is_filtered": True,
                   "file_timestamps_paths": None},
        "annotations": {"is_filtered": True},
        "properties": {"group": None, "gain_to_uV": gains, "offset_to_uV": offsets, "gain_to_physical_unit": None,
                       "offset_to_physical_unit": None, "physical_unit": None},
        "relative_paths": True})
    _dump(folder / "si_folder.json", {
        "class": "spikeinterface.core.binaryfolder.BinaryFolderRecording", "module": "spikeinterface",
        "version": SI_VERSION, "kwargs": {"folder_path": "."}, "annotations": {"is_filtered": True},
        "properties": {}, "relative_paths": True})
    _dump(folder / "annotations.json", {"is_filtered": True})
    write_probegroup(folder, pos)
    props = folder / "properties"
    props.mkdir(exist_ok=True)
    np.save(props / "group.npy", np.zeros(n_ch, dtype="int64"))
    np.save(props / "gain_to_uV.npy", np.array(gains))
    np.save(props / "offset_to_uV.npy", np.array(offsets))
    np.save(props / "brain_area.npy", areas.astype("U"))


def write_numpy_folder(folder: Path, trains: list[np.ndarray], unit_ids: list[str], props: dict | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    dtype = [("sample_index", "int64"), ("unit_index", "int64"), ("segment_index", "int64")]
    n = sum(len(t) for t in trains)
    spikes = np.zeros(n, dtype=dtype)
    i = 0
    for ui, tr in enumerate(trains):
        spikes["sample_index"][i:i + len(tr)] = tr
        spikes["unit_index"][i:i + len(tr)] = ui
        i += len(tr)
    order = np.lexsort((spikes["unit_index"], spikes["sample_index"]))
    np.save(folder / "spikes.npy", spikes[order])
    _dump(folder / "numpysorting_info.json", {"sampling_frequency": FS, "unit_ids": unit_ids, "num_segments": 1})
    _dump(folder / "si_folder.json", {
        "class": "spikeinterface.core.sortingfolder.NumpyFolderSorting", "module": "spikeinterface",
        "version": SI_VERSION, "kwargs": {"folder_path": ".", "mmap_mode": None}, "annotations": {},
        "properties": {}, "relative_paths": True})
    _dump(folder / "annotations.json", {})
    pdir = folder / "properties"
    pdir.mkdir(exist_ok=True)
    for k, v in (props or {}).items():
        np.save(pdir / f"{k}.npy", np.asarray(v))


def write_kilosort_folder(folder: Path, trains: list[np.ndarray], unit_ids: list[int], labels: list[str],
                          n_channels: int) -> None:
    """The files of a Kilosort / Phy output folder that SpikeInterface's read_kilosort needs."""
    folder.mkdir(parents=True, exist_ok=True)
    times = np.concatenate(trains).astype("uint64")
    clusters = np.concatenate([np.full(len(t), u, dtype="int32") for t, u in zip(trains, unit_ids)])
    order = np.argsort(times, kind="stable")
    np.save(folder / "spike_times.npy", times[order])
    np.save(folder / "spike_clusters.npy", clusters[order])
    np.save(folder / "spike_templates.npy", clusters[order].astype("uint32"))
    np.save(folder / "amplitudes.npy", np.ones(len(times), dtype="float32"))
    (folder / "params.py").write_text(
        "dat_path = 'recording.dat'\n"
        f"n_channels_dat = {n_channels}\n"
        "dtype = 'int16'\n"
        "offset = 0\n"
        f"sample_rate = {FS}\n"
        "hp_filtered = True\n")
    with open(folder / "cluster_KSLabel.tsv", "w") as fh:
        fh.write("cluster_id\tKSLabel\n")
        for u, lab in zip(unit_ids, labels):
            fh.write(f"{u}\t{lab}\n")
    with open(folder / "cluster_group.tsv", "w") as fh:
        fh.write("cluster_id\tgroup\n")
        for u, lab in zip(unit_ids, labels):
            fh.write(f"{u}\t{'good' if lab == 'good' else 'mua'}\n")


# --------------------------------------------------------------------------- the session


def build(duration: float = 60.0, seed: int = 7, engine: str = "auto"):
    if engine == "auto":
        try:
            import spikeinterface  # noqa: F401
            engine = "spikeinterface"
        except ImportError:
            engine = "numpy"
    if engine == "spikeinterface":
        traces, pos, gt_trains = simulate_spikeinterface(duration, seed)
    else:
        traces, pos, gt_trains = simulate_numpy(duration, seed)
    rng = np.random.default_rng(seed + 1)
    n = traces.shape[0]
    sorted_trains = [np.asarray(gt_trains[u], dtype=np.int64) for u in GOOD_UNITS]
    for u in MUA_UNITS:                     # a contaminated unit: 40 % random extra spikes
        base = np.asarray(gt_trains[u], dtype=np.int64)
        extra = rng.integers(200, n - 200, size=int(0.4 * len(base)) + 5)
        sorted_trains.append(np.unique(np.concatenate([base, extra])))
    for _ in NOISE_UNITS:                    # nothing in the signal: noise clusters
        k = int(rng.uniform(6.0, 10.0) * duration)
        sorted_trains.append(np.unique(rng.integers(200, n - 200, size=k)))
    a, b = SPLIT_UNIT
    full = sorted_trains[a]
    sorted_trains[a] = full[0::2]
    sorted_trains.append(full[1::2])
    return dict(engine=engine, traces=traces, pos=pos, gt_trains=[np.asarray(t, dtype=np.int64) for t in gt_trains],
                sorted_trains=sorted_trains, duration=n / FS)


def session_json(sim) -> dict:
    n_ch = len(sim["pos"])
    return {
        "session_id": SESSION_ID, "source": f"synthetic ({sim['engine']})", "stream": "probe00",
        "probe": {"model": "synthetic 2-column", "serial": "0"}, "sampling_frequency": FS,
        "t_start": T_START, "t_stop": T_START + sim["duration"], "duration_s": sim["duration"],
        "n_channels": n_ch, "bad_channels": [], "motion_corrected": False, "motion_method": None,
        "time_source": "t_start", "session_start_time": "2026-11-09T09:00:00+00:00",
        "identifier": "synthetic-a1b2c3d4", "session_description": "synthetic ground-truth session",
        "subject": {"subject_id": "synth01", "species": "Mus musculus", "sex": "U"},
    }


def write_reference_csv(path: Path, sim) -> None:
    with open(path, "w") as fh:
        fh.write("unit_id,spike_time\n")
        for u, tr in enumerate(sim["gt_trains"]):
            for f in tr:
                fh.write(f"gt{u},{T_START + f / FS:.6f}\n")


def write_reference_nwb(path: Path, sim) -> None:
    """The ground truth as an NWB Units table (needs pynwb; used by the tests)."""
    from datetime import datetime

    from pynwb import NWBHDF5IO, NWBFile

    nwb = NWBFile(session_description="synthetic ground truth", identifier="synthetic-a1b2c3d4-gt",
                  session_start_time=datetime.fromisoformat("2026-11-09T09:00:00+00:00"))
    from pynwb.misc import Units

    nwb.units = Units(name="units", description="ground-truth units of the synthetic recording")
    nwb.add_unit_column("gt_unit", "ground-truth unit name")
    for u, tr in enumerate(sim["gt_trains"]):
        nwb.add_unit(spike_times=T_START + tr / FS, gt_unit=f"gt{u}")
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


def write(out: Path, layout: str = "both", engine: str = "auto", duration: float = 60.0, kilosort: bool = False,
          seed: int = 7, sim=None) -> dict:
    out = Path(out)
    sim = sim or build(duration, seed, engine)
    layouts = ["linked", "flat"] if layout == "both" else [layout]
    for lay in layouts:
        base = out / lay
        if base.exists():
            shutil.rmtree(base)
        rec_root = base / "recording" if lay == "linked" else base
        sort_root = base / "sorting" if lay == "linked" else base
        ref_root = base / "reference" if lay == "linked" else base
        sess = rec_root / "ephys-prep" / SESSION_ID
        write_binary_folder(sess / "recording", sim["traces"], sim["pos"], brain_areas(sim["pos"]))
        _dump(sess / "session.json", session_json(sim))
        n_units = len(sim["sorted_trains"])
        if kilosort:
            labels = ["good"] * len(GOOD_UNITS) + ["mua"] * (n_units - len(GOOD_UNITS) - 1) + ["good"]
            write_kilosort_folder(sort_root / f"{SESSION_ID}_kilosort4", sim["sorted_trains"], list(range(n_units)),
                                  labels, len(sim["pos"]))
        else:
            ss = sort_root / "spike-sort" / SESSION_ID
            write_numpy_folder(ss / "sorting", sim["sorted_trains"], [str(i) for i in range(n_units)])
            _dump(ss / "sort.json", {"session_id": SESSION_ID, "sorter": "synthetic", "version": "0",
                                     "parameters": {}, "n_units": n_units, "wall_seconds": 0.0})
        ref_root.mkdir(parents=True, exist_ok=True)
        write_reference_csv(ref_root / f"{SESSION_ID}_groundtruth_spikes.csv", sim)
    return sim


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--layout", choices=["linked", "flat", "both"], default="both")
    ap.add_argument("--engine", choices=["auto", "spikeinterface", "numpy"], default="auto")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--kilosort", action="store_true", help="write the sorting as a Kilosort/Phy folder")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    sim = write(a.out, a.layout, a.engine, a.duration, a.kilosort, a.seed)
    print(f"wrote {a.layout} fixture ({sim['engine']}, {sim['duration']:.0f} s, {len(sim['sorted_trains'])} units) "
          f"to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
