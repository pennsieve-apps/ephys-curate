"""The synthetic fixture through the processor (no Docker): labels, NWB, times, reference, layouts."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from processor import curate as C
from processor import main as M
from processor.inputs import discover, subses_id
from tests import make_fixture as FX

SID = FX.SESSION_ID
REPO = Path(__file__).resolve().parent.parent
needs_models = pytest.mark.skipif(M.models_dir() is None and not os.environ.get("UNITREFINE_MODELS_DIR"),
                                  reason="UnitRefine models not downloaded (make models)")


@pytest.fixture(scope="session")
def sim():
    return FX.build(duration=60.0, seed=7, engine="spikeinterface")


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory, sim) -> Path:
    out = tmp_path_factory.mktemp("fixture")
    FX.write(out, "both", sim=sim)
    FX.write(out / "ks", "linked", kilosort=True, sim=sim)
    nwb_ref = out / "nwbref"
    FX.write(nwb_ref, "linked", sim=sim)
    (nwb_ref / "linked" / "reference" / f"{SID}_groundtruth_spikes.csv").unlink()
    FX.write_reference_nwb(nwb_ref / "linked" / "reference" / f"{SID}_desc-groundtruth_ecephys.nwb", sim)
    return out


def run(monkeypatch, input_dir: Path, output_dir: Path, **params) -> int:
    for k in M.PARAMETERS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("THREADS", "2")
    for k, v in params.items():
        monkeypatch.setenv(k, str(v))
    return M.main()


@pytest.fixture(scope="session")
def default_run(fixture_dir, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("out-default")
    mp = pytest.MonkeyPatch()
    try:
        assert run(mp, fixture_dir / "linked", out) == 0
    finally:
        mp.undo()
    return out / "ephys-curate"


def test_session_ids():
    assert subses_id("sub-NR-0031_ses-caa5dddc-9290-4e27-9f5e-575ba3598614_desc-processed.nwb") == \
        "sub-NR-0031_ses-caa5dddc"
    assert subses_id("sub-F35_ses-OF-F35-S1_behavior.nwb") == "sub-F35_ses-OF-F35-S1"


def test_discovery_both_layouts(fixture_dir):
    for lay in ("linked", "flat"):
        f = discover(fixture_dir / lay)
        assert set(f.recordings) == {SID} and set(f.sortings) == {SID} and set(f.references) == {SID}
        assert f.sortings[SID].kind == "spikeinterface"
    k = discover(fixture_dir / "ks" / "linked")
    assert k.sortings[SID].kind == "kilosort"


def test_final_label_rule():
    assert C.final_label("good", "sua", "consensus") == "good"
    assert C.final_label("good", "mua", "consensus") == "mua"
    assert C.final_label("noise", "noise", "consensus") == "noise"
    assert C.final_label("noise", "mua", "consensus") == "noise"
    assert C.final_label("mua", "noise", "consensus") == "noise"
    assert C.final_label("noise", "sua", "consensus") == "mua"       # disagreement against a single unit: kept
    assert C.final_label("good", "noise", "consensus") == "mua"
    assert C.final_label("non_soma", "sua", "consensus") == "mua"
    assert C.final_label("good", None, "consensus") == "good"         # UnitRefine missing: Bombcell alone
    assert C.final_label("non_soma", "sua", "bombcell") == "mua"
    assert C.final_label("mua", "sua", "unitrefine") == "good"


def test_parameters():
    with pytest.raises(M.ParameterError):
        M.read_parameters({"LABEL_RULE": "vote"})
    with pytest.raises(M.ParameterError):
        M.read_parameters({"BOMBCELL_THRESHOLDS": "{not json"})
    p = M.read_parameters({"BOMBCELL_THRESHOLDS": '{"mua": {"snr": {"greater": 3}}}'})
    thr = C.bombcell_thresholds(p["BOMBCELL_THRESHOLDS"])
    assert thr["mua"]["snr"]["greater"] == 3 and "isolation_distance" not in thr["mua"]
    with pytest.raises(ValueError):
        C.bombcell_thresholds({"bogus": {}})


@needs_models
def test_outputs(default_run):
    names = {p.name for p in default_run.iterdir()}
    for f in ("units.csv", "sessions.csv", "summary.json", "provenance.json", "bombcell_thresholds.json",
              f"{SID}_units.nwb", f"{SID}_analyzer.zarr", f"{SID}_bombcell", f"{SID}_unit_yield.png",
              f"{SID}_metrics.png", f"{SID}_unit_gallery.png", f"{SID}_depth_map.png", f"{SID}_agreement.png",
              f"{SID}_reference_units.csv", f"{SID}_groundtruth_performance.csv"):
        assert f in names, f
    prov = json.loads((default_run / "provenance.json").read_text())
    assert list(prov)[:6] == ["app", "version", "variant", "tools", "parameters", "inputs"]
    for k in ("started", "finished", "wall_seconds", "peak_rss_mb", "gpu", "sessions", "citations"):
        assert k in prov
    summ = json.loads((default_run / "summary.json").read_text())
    assert all(f["caption"] for f in summ["figures"]) and len(summ["figures"]) == 5


@needs_models
def test_labels_recover_planted_units(default_run):
    u = pd.read_csv(default_run / "units.csv", dtype={"unit_name": str})
    lab = dict(zip(u["unit_name"], u["label"]))
    assert all(lab[str(i)] == "good" for i in FX.GOOD_UNITS), lab
    assert all(lab[str(i)] == "mua" for i in FX.MUA_UNITS), lab
    assert all(lab[str(i)] == "noise" for i in FX.NOISE_UNITS), lab
    assert set(u["bombcell_label"]) <= {"good", "mua", "noise", "non_soma"}
    assert set(u["unitrefine_label"]) <= {"sua", "mua", "noise"}
    assert set(u["brain_area"]) <= {"CA1", "VISp"}


@needs_models
def test_nwb_units_session_clock(default_run, sim):
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(default_run / f"{SID}_units.nwb"), "r") as io:
        nwb = io.read()
        df = nwb.units.to_dataframe()
        assert nwb.identifier == "synthetic-a1b2c3d4"
        assert nwb.subject.subject_id == "synth01"
        for c in ("spike_times", "label", "bombcell_label", "unitrefine_label", "brain_area", "peak_channel",
                  "waveform_mean", "snr", "presence_ratio", "electrodes", "obs_intervals"):
            assert c in df.columns, c
        assert nwb.units["waveform_mean"].data[0].shape == (81,)
        st = df.loc[0, "spike_times"]
        expected = FX.T_START + sim["gt_trains"][0] / FX.FS
        assert np.allclose(st, expected)                 # session seconds, from 600 s, not re-zeroed
        assert st.min() >= 600.0
        assert np.nanmin(nwb.units["waveform_mean"].data[0]) < -30   # uV, trough at the centre
        assert np.argmin(nwb.units["waveform_mean"].data[0]) == 40


@needs_models
def test_ground_truth_agreement(default_run):
    s = json.loads((default_run / "summary.json").read_text())["sessions"][SID]["reference"]
    assert s["kind"] == "ground_truth"
    assert s["n_well_detected"] >= 8
    assert s["matched_by_label"]["good"]["matched"] == s["matched_by_label"]["good"]["n"] == 8
    assert s["matched_by_label"]["noise"]["matched"] == 0


@needs_models
def test_flat_kilosort_and_nwb_reference(fixture_dir, tmp_path, monkeypatch):
    out = tmp_path / "ks"
    assert run(monkeypatch, fixture_dir / "ks" / "linked", out, PCA_METRICS="false", GALLERY_UNITS=0) == 0
    u = pd.read_csv(out / "ephys-curate" / "units.csv", dtype={"unit_name": str})
    assert "kslabel" in u.columns
    assert (u.set_index("unit_name").loc[[str(i) for i in FX.NOISE_UNITS], "label"] == "noise").all()
    summ = json.loads((out / "ephys-curate" / "summary.json").read_text())
    assert any("imputed" in w for w in summ["warnings"])
    out2 = tmp_path / "nwbref"
    assert run(monkeypatch, fixture_dir / "nwbref" / "linked", out2, PCA_METRICS="false", GALLERY_UNITS=0,
               EXPORT_PHY="true") == 0
    s = json.loads((out2 / "ephys-curate" / "summary.json").read_text())["sessions"][SID]
    assert s["reference"]["kind"] == "ground_truth" and s["reference"]["n_well_detected"] >= 8
    assert (out2 / "ephys-curate" / f"{SID}_phy" / "params.py").is_file()


def test_bad_inputs(tmp_path, monkeypatch):
    (tmp_path / "in").mkdir()
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out") == 1
    assert run(monkeypatch, tmp_path / "in", tmp_path / "out", LABEL_RULE="x") == 2


def test_numpy_engine_fixture_loads(tmp_path):
    import spikeinterface.full as si

    FX.write(tmp_path, "linked", engine="numpy", duration=10.0)
    r = si.load(tmp_path / "linked" / "recording" / "ephys-prep" / SID / "recording")
    s = si.load(tmp_path / "linked" / "sorting" / "spike-sort" / SID / "sorting")
    assert r.get_times()[0] == pytest.approx(FX.T_START)
    assert list(r.get_property("brain_area")[:1]) == ["CA1"]
    assert s.get_num_units() == 12


def test_streaming_pca_metrics_match_spikeinterface(fixture_dir):
    """curate.pca_metrics (one unit at a time) gives SpikeInterface's quality_metrics PCA values."""
    import spikeinterface.full as si

    base = fixture_dir / "linked"
    r = si.load(base / "recording" / "ephys-prep" / SID / "recording")
    s = si.load(base / "sorting" / "spike-sort" / SID / "sorting")
    a = si.create_sorting_analyzer(s, r, format="memory", sparse=True, method="radius", radius_um=100)
    a.compute(["random_spikes", "noise_levels", "waveforms", "templates"], n_jobs=1, progress_bar=False)
    a.compute("principal_components", n_components=5, mode="by_channel_local", n_jobs=1, progress_bar=False)
    a.compute("quality_metrics", metric_names=C.PCA_QUALITY_METRICS, seed=42, n_jobs=1, progress_bar=False)
    ref = a.get_extension("quality_metrics").get_data()
    mine = C.pca_metrics(a, seed=42)
    for c in mine.columns:
        assert np.allclose(ref[c].to_numpy(float), mine[c].to_numpy(float), equal_nan=True, rtol=1e-4), c
