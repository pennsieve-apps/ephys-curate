"""
Finding the inputs and pairing them into sessions.

Ports (placement: subfolder; without links everything may also sit side by side at the root of
INPUT_DIR, so files are found by name and content, recursively, following symlinks):

  recording   ephys-prep: ephys-prep/<session_id>/recording/ (SpikeInterface binary_folder, or any
              folder SpikeInterface saved a recording to) and ephys-prep/<session_id>/session.json
  sorting     spike-sort: spike-sort/<session_id>/sorting/ (SpikeInterface numpy_folder),
              spike-sort/<session_id>/sort.json and sorter_output/ (Kilosort's own files); or a
              Kilosort / Phy folder (params.py, spike_times.npy, spike_clusters.npy) from a lab
              that sorted elsewhere
  reference   optional: an NWB file with a Units table (e.g. IBL's processed NWB of the session,
              or a ground-truth simulation), or a CSV of reference spikes (unit_id, spike_time)

Sessions are matched on session_id (conventions.md, "Sessions"): session.json / sort.json say it;
otherwise a sessions.csv sheet, a sub-<a>_ses-<b> in a file or folder name, or the folder name.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("ephys-curate")

SUBSES = re.compile(r"sub-(?P<sub>[A-Za-z0-9-]+)_ses-(?P<ses>[A-Za-z0-9-]+)")
UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]+$")
GENERIC_NAMES = {"recording", "sorting", "sorter_output", "kilosort4", "kilosort", "ks4", "phy", "output",
                 "spike-sort", "ephys-prep", "reference", "input", "data"}


def sanitise(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", name).strip("-") or "session"


def subses_id(text: str) -> str | None:
    """sub-<a>_ses-<b> -> session id, <b> cut to 8 characters when it is a UUID (conventions.md)."""
    m = SUBSES.search(text)
    if not m:
        return None
    ses = m.group("ses")
    if UUID.match(ses):
        ses = ses[:8]
    return f"sub-{m.group('sub')}_ses-{ses}"


def path_session_id(path: Path, root: Path) -> str:
    """Rule 2 then 3 of conventions.md: a sub-/ses- name in the path (deepest first), else the stem."""
    rel = path.relative_to(root) if path.is_relative_to(root) else path
    for part in reversed(rel.parts):
        sid = subses_id(part)
        if sid:
            return sid
    for part in reversed(rel.parts):
        stem = part.split(".")[0] if not Path(part).suffix == ".zarr" else part[:-5]
        if stem.lower() not in GENERIC_NAMES:
            return sanitise(stem)
    return sanitise(path.stem)


def walk_dirs(root: Path):
    """(dirpath, dirnames, filenames) under root, following symlinks, sorted, hidden entries skipped."""
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:      # a symlink loop
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        yield Path(dirpath), dirnames, sorted(f for f in filenames if not f.startswith("."))


def _read_json(path: Path) -> dict:
    try:
        with open(path) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def si_folder_kind(folder: Path) -> str | None:
    """'recording' / 'sorting' for a folder SpikeInterface saved, else None."""
    info = folder / "si_folder.json"
    if not info.is_file():
        if (folder / ".zattrs").is_file() or (folder / "zarr.json").is_file():
            attrs = _read_json(folder / ".zattrs")
            cls = str(attrs.get("class", "") or (attrs.get("dict") or {}).get("class", ""))
            if "Recording" in cls or "channel_ids" in attrs:
                return "recording"
            if "Sorting" in cls or "unit_ids" in attrs:
                return "sorting"
        return None
    cls = str(_read_json(info).get("class", ""))
    if "Sorting" in cls or (folder / "numpysorting_info.json").is_file() or (folder / "spikes.npy").is_file():
        return "sorting"
    if "Recording" in cls or (folder / "binary.json").is_file():
        return "recording"
    return None


def is_kilosort_folder(names: set[str]) -> bool:
    return "spike_times.npy" in names and ("spike_clusters.npy" in names or "spike_templates.npy" in names) \
        and "params.py" in names


@dataclass
class RecordingInput:
    session_id: str
    folder: Path
    session_json: Path | None
    session: dict[str, Any]


@dataclass
class SortingInput:
    session_id: str
    folder: Path
    kind: str                      # "spikeinterface" | "kilosort"
    sort_json: Path | None
    sort: dict[str, Any]
    sorter_output: Path | None     # Kilosort's own folder next to a spike-sort numpy_folder (for KSLabel)


@dataclass
class ReferenceInput:
    session_id: str
    path: Path
    kind: str                      # "nwb" | "csv"


@dataclass
class Inputs:
    recordings: dict[str, RecordingInput]
    sortings: dict[str, SortingInput]
    references: dict[str, ReferenceInput]
    sheet: dict[str, dict[str, Any]]
    sheet_paths: dict[str, str]
    skipped: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    files_seen: list[str] = field(default_factory=list)


def _port_root(root: Path, port: str) -> Path:
    folder = root / port
    return folder if folder.is_dir() else root


def read_sheet(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str], Path | None]:
    sheets = []
    for d, _, files in walk_dirs(root):
        if "sessions.csv" in files:
            sheets.append(d / "sessions.csv")
    if not sheets:
        return {}, {}, None
    if len(sheets) > 1:
        log.info("%d sessions.csv files; using %s", len(sheets), sheets[0])
    rows: dict[str, dict[str, Any]] = {}
    paths: dict[str, str] = {}
    try:
        with open(sheets[0], newline="") as fh:
            for r in csv.DictReader(fh):
                sid = (r.get("session_id") or "").strip()
                if not sid:
                    continue
                if not SESSION_ID.match(sid):
                    sid = sanitise(sid)
                rows[sid] = {k: v for k, v in r.items() if k is not None}
                if r.get("path"):
                    paths[sid] = r["path"].strip()
    except (OSError, ValueError) as exc:
        log.warning("cannot read %s: %s", sheets[0], exc)
    return rows, paths, sheets[0]


def _sheet_match(path: Path, sheet_paths: dict[str, str]) -> str | None:
    parts = set(path.parts)
    names = {p for p in parts} | {path.name, path.stem}
    for sid, p in sheet_paths.items():
        pn = Path(p).name
        if pn and (pn in names or Path(pn).stem in names):
            return sid
    return None


def discover(input_dir: Path) -> Inputs:
    sheet, sheet_paths, _ = read_sheet(input_dir)
    out = Inputs({}, {}, {}, sheet, sheet_paths)

    def session_for(path: Path, explicit: str | None = None) -> str:
        if explicit and SESSION_ID.match(str(explicit)):
            return str(explicit)
        if explicit:
            return sanitise(str(explicit))
        hit = _sheet_match(path, sheet_paths)
        return hit or path_session_id(path, input_dir)

    # --- recordings ---------------------------------------------------------------------------
    rec_root = _port_root(input_dir, "recording")
    sort_root = _port_root(input_dir, "sorting")
    ref_root = input_dir / "reference" if (input_dir / "reference").is_dir() else input_dir

    for d, dirnames, files in walk_dirs(rec_root):
        kind = si_folder_kind(d)
        if kind is None:
            continue
        dirnames[:] = []            # never look inside a saved SpikeInterface folder
        if kind != "recording":
            continue
        sj = d.parent / "session.json"
        session = _read_json(sj) if sj.is_file() else {}
        sid = session_for(d if not session else sj.parent, session.get("session_id"))
        if sid in out.recordings:
            out.skipped[str(d)] = f"a second recording for session {sid}"
            continue
        out.recordings[sid] = RecordingInput(sid, d, sj if sj.is_file() else None, session)
        out.files_seen.append(str(d.relative_to(input_dir)))

    # --- sortings -----------------------------------------------------------------------------
    ks_candidates: list[tuple[Path, set[str]]] = []
    for d, dirnames, files in walk_dirs(sort_root):
        names = set(files)
        if is_kilosort_folder(names):
            ks_candidates.append((d, names))
            dirnames[:] = []
            continue
        kind = si_folder_kind(d)
        if kind is None:
            continue
        dirnames[:] = []
        if kind != "sorting":
            continue
        sj = d.parent / "sort.json"
        sort = _read_json(sj) if sj.is_file() else {}
        explicit = sort.get("session_id")
        if not explicit and d.name in ("sorting",) and d.parent != sort_root:
            explicit = d.parent.name
        sid = session_for(d, explicit)
        so = d.parent / "sorter_output"
        so = so if so.is_dir() and is_kilosort_folder({p.name for p in so.iterdir()}) else None
        if sid in out.sortings:
            out.skipped[str(d)] = f"a second sorting for session {sid}"
            continue
        out.sortings[sid] = SortingInput(sid, d, "spikeinterface", sj if sj.is_file() else None, sort, so)
        out.files_seen.append(str(d.relative_to(input_dir)))
    for d, names in ks_candidates:
        # Kilosort's own folder next to a spike-sort numpy_folder is only read for KSLabel.
        if (d.parent / "sorting").is_dir() and si_folder_kind(d.parent / "sorting") == "sorting":
            continue
        sj = d.parent / "sort.json"
        sort = _read_json(sj) if sj.is_file() else {}
        explicit = sort.get("session_id")
        if not explicit and d.name.lower() in ("sorter_output",) and d.parent != sort_root:
            explicit = d.parent.name
        sid = session_for(d, explicit)
        if sid in out.sortings:
            out.skipped[str(d)] = f"a second sorting for session {sid}"
            continue
        out.sortings[sid] = SortingInput(sid, d, "kilosort", sj if sj.is_file() else None, sort, d)
        out.files_seen.append(str(d.relative_to(input_dir)))

    # --- references ---------------------------------------------------------------------------
    for d, dirnames, files in walk_dirs(ref_root):
        if si_folder_kind(d) is not None or is_kilosort_folder(set(files)):
            dirnames[:] = []
            continue
        for f in files:
            p = d / f
            low = f.lower()
            if low.endswith(".nwb"):
                kind = "nwb"
            elif low.endswith(".csv") and low != "sessions.csv" and _looks_like_spike_csv(p):
                kind = "csv"
            else:
                continue
            sid = session_for(p)
            if sid in out.references:
                # Prefer NWB over CSV, else keep the first.
                if out.references[sid].kind == "csv" and kind == "nwb":
                    out.skipped[str(out.references[sid].path)] = f"an NWB reference is also given for {sid}"
                else:
                    out.skipped[str(p)] = f"a second reference for session {sid}"
                    continue
            out.references[sid] = ReferenceInput(sid, p, kind)
            out.files_seen.append(str(p.relative_to(input_dir)))

    # --- pairing when names differ -------------------------------------------------------------
    if len(out.recordings) == 1 and len(out.sortings) == 1:
        (rid,), (sid,) = out.recordings.keys(), out.sortings.keys()
        if rid != sid:
            msg = (f"recording session {rid!r} and sorting session {sid!r} have different ids; "
                   f"pairing them because they are the only ones (session id {rid!r})")
            log.warning(msg)
            out.warnings.append(msg)
            s = out.sortings.pop(sid)
            s.session_id = rid
            out.sortings[rid] = s
    if len(out.recordings) == 1 and len(out.references) == 1:
        (rid,), (fid,) = out.recordings.keys(), out.references.keys()
        if rid != fid:
            msg = f"reference {out.references[fid].path.name} is paired with session {rid!r} (the only session)"
            log.info(msg)
            r = out.references.pop(fid)
            r.session_id = rid
            out.references[rid] = r
    return out


def _looks_like_spike_csv(path: Path) -> bool:
    try:
        with open(path, newline="") as fh:
            header = next(csv.reader(fh), [])
    except (OSError, StopIteration, UnicodeDecodeError):
        return False
    cols = {h.strip().lower() for h in header}
    return bool(cols & {"unit_id", "cluster_id", "unit", "cluster"}) and bool(
        cols & {"spike_time", "spike_times", "time", "times", "t"})
