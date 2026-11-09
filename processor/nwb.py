"""
The session's NWB file with the Units table.

spike_times are seconds on the session clock: the recording's own times (its t_start, or its
timestamps when ephys-prep kept them), never re-zeroed. The file copies session_start_time,
session_description, subject and timestamps_reference_time from session.json (which
ephys-prep copies from the source NWB), so it lines up with the session's other NWB files.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Columns written with their own descriptions; every other column of the table gets a generic one.
DESCRIPTIONS = {
    "unit_name": "unit id in the sorting (spike-sort / Kilosort cluster id)",
    "peak_channel": "channel id with the largest (negative) template amplitude",
    "brain_area": "brain area of the peak channel (recording channel property brain_area), '' when unknown",
    "x": "unit location x (um, probe coordinates; monopolar triangulation)",
    "y": "unit location y (um, along the probe; monopolar triangulation)",
    "z": "unit distance from the probe plane (um; monopolar triangulation)",
    "depth": "unit position along the probe (um, = y; 0 = probe tip side of the contacts' coordinates)",
    "label": "final curation label: good / mua / noise (rule in the README, 'Labels')",
    "bombcell_label": "Bombcell label: good / mua / noise / non_soma",
    "bombcell_reasons": "thresholds the unit failed in Bombcell ('' for good units)",
    "unitrefine_label": "UnitRefine label: sua / mua / noise",
    "unitrefine_probability": "UnitRefine classifier probability of its label",
    "unitrefine_noise_probability": "UnitRefine probability that the unit is noise",
    "kslabel": "Kilosort's own label (good / mua)",
    "reference_match": "matched reference / ground-truth unit (compare_two_sorters), '' when none",
    "reference_agreement": "agreement score with the matched reference unit (0-1)",
}
SKIP = {"unit_id", "peak_channel_index", "session_id"}


def _dt(value, default=None):
    if not value:
        return default
    try:
        d = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return default


def write_units_nwb(path: Path, session_id: str, session: dict, recording, table: pd.DataFrame,
                    spike_times: list[np.ndarray], waveforms: np.ndarray, t_window: tuple[float, float],
                    app_version: str, sheet_row: dict | None = None) -> dict:
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.file import Subject

    notes = {}
    start = _dt(session.get("session_start_time"))
    if start is None:
        start = datetime(1970, 1, 1, tzinfo=timezone.utc)
        notes["session_start_time"] = "unknown (no source NWB): 1970-01-01T00:00:00Z; times are seconds of the recording"
    src = session.get("source") or "the recording"
    notes_text = (f"derived from {Path(str(src)).name} (identifier {session.get('identifier') or 'unknown'}) by "
                  f"ephys-curate {app_version}: units curated with a SpikeInterface SortingAnalyzer, Bombcell and "
                  "UnitRefine.")
    kw = dict(
        session_description=session.get("session_description") or f"{session_id}: curated spike-sorted units",
        identifier=str(uuid.uuid4()), session_start_time=start,
        timestamps_reference_time=_dt(session.get("timestamps_reference_time"), start),
        session_id=str(session.get("nwb_session_id") or session_id),
        source_script=f"ephys-curate {app_version}", source_script_file_name="ephys-curate",
        notes=notes_text,
    )
    for k in ("experimenter", "institution", "lab", "keywords", "experiment_description"):
        v = session.get(k)
        if v not in (None, "", []):
            kw[k] = list(v) if k in ("experimenter", "keywords") and isinstance(v, (list, tuple)) else (
                [v] if k in ("experimenter", "keywords") else str(v))
    kw.setdefault("keywords", ["electrophysiology", "spike sorting", "curation", "units"])
    nwb = NWBFile(**kw)
    subj = session.get("subject") or {}
    if isinstance(subj, dict) and (subj or (sheet_row or {}).get("subject_id")):
        kw = {k: subj[k] for k in ("subject_id", "species", "sex", "age", "description", "genotype", "strain",
                                   "weight")
              if subj.get(k) not in (None, "")}
        if subj.get("date_of_birth"):
            dob = _dt(subj["date_of_birth"])
            if dob is not None:
                kw["date_of_birth"] = dob
        if "subject_id" not in kw and (sheet_row or {}).get("subject_id"):
            kw["subject_id"] = str(sheet_row["subject_id"])
        if kw.get("sex") not in (None, "M", "F", "U", "O"):
            kw.pop("sex")
        try:
            nwb.subject = Subject(**kw)
        except (TypeError, ValueError):
            pass

    probe = session.get("probe") or {}
    device = nwb.create_device(name=str(probe.get("model") or "probe"),
                               description=f"serial {probe.get('serial', '')}".strip(),
                               manufacturer=str(probe["manufacturer"]) if probe.get("manufacturer") else None)
    group = nwb.create_electrode_group(name=str(session.get("stream") or "probe"),
                                       description="recording channels (after ephys-prep)",
                                       location="see electrodes.location", device=device)
    pos = recording.get_channel_locations()
    areas = recording.get_property("brain_area")
    nwb.add_electrode_column("channel_name", "channel id in the recording")
    nwb.add_electrode_column("rel_x", "contact x on the probe (um)")
    nwb.add_electrode_column("rel_y", "contact y on the probe (um)")
    for i, cid in enumerate(recording.channel_ids):
        area = str(areas[i]) if areas is not None else ""
        nwb.add_electrode(group=group, location=area if area not in ("", "nan", "None") else "unknown",
                          channel_name=str(cid), rel_x=float(pos[i, 0]), rel_y=float(pos[i, 1]))

    from pynwb.misc import Units

    nwb.units = Units(
        name="units",
        description=("Units curated by ephys-curate. spike_times in seconds on the session clock; "
                     "waveform_mean: mean waveform on the peak channel, 81 samples centred on the trough, uV; "
                     "label = final good/mua/noise"),
        waveform_rate=float(recording.sampling_frequency), waveform_unit="microvolts",
        resolution=1.0 / float(recording.sampling_frequency))
    cols = [c for c in table.columns if c not in SKIP]
    for c in cols:
        nwb.add_unit_column(c if c != "unit_id" else "unit_name", DESCRIPTIONS.get(c, f"{c} (SpikeInterface metric)"))
    t0, t1 = t_window
    try:
        ids_int = len({int(str(u)) for u in table["unit_id"]}) == len(table)
    except ValueError:
        ids_int = False
    for i, (_, row) in enumerate(table.iterrows()):
        extra = {}
        for c in cols:
            v = row[c]
            if table[c].dtype == object or isinstance(v, str):
                extra[c] = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
            elif table[c].dtype == bool:
                extra[c] = bool(v)
            else:
                extra[c] = float(v) if v is not None else np.nan
        uid = row["unit_id"]
        if ids_int:
            extra["id"] = int(uid)
        nwb.add_unit(spike_times=np.asarray(spike_times[i], dtype="float64"),
                     obs_intervals=[[t0, t1]], electrodes=[int(row["peak_channel_index"])],
                     electrode_group=group, waveform_mean=waveforms[i].astype("float32"), **extra)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return notes
