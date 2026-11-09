"""
Downloads the pretrained UnitRefine classifiers from Hugging Face (MIT licence) at pinned commits
and checks the model files' sha256, so the image runs with no network (Dockerfile; `make models`).

    python scripts/fetch_models.py OUT_DIR
"""
import hashlib
import sys
from pathlib import Path

MODELS = {
    # repo id: (commit, {file: sha256 or None})
    "SpikeInterface/UnitRefine_noise_neural_classifier": (
        "280cd07ff8310286ded5c0cfdb382e92e2a43a98",
        {"best_model.skops": "f946d77324393959a49e2b83df2083326ea5a934e068eb48f2a362a11e551067",
         "model_info.json": None, "metadata.json": None, "README.md": None}),
    "SpikeInterface/UnitRefine_sua_mua_classifier": (
        "9ee216e7e0e7dbf0778cb07a8d1753700dbbad3b",
        {"best_model.skops": "9f0417c29c1b9828add06ea1b26740c853152c87e3bb1dafeb006c52e5ec04c4",
         "model_info.json": None, "metadata.json": None, "README.md": None}),
}


def main(out: Path) -> int:
    from huggingface_hub import hf_hub_download

    for repo, (rev, files) in MODELS.items():
        dest = out / repo.split("/")[1]
        dest.mkdir(parents=True, exist_ok=True)
        for name, sha in files.items():
            p = Path(hf_hub_download(repo, name, revision=rev, local_dir=dest))
            if sha:
                got = hashlib.sha256(p.read_bytes()).hexdigest()
                if got != sha:
                    print(f"{repo}/{name}: sha256 {got} != {sha}", file=sys.stderr)
                    return 1
        (dest / "REVISION").write_text(f"{repo}@{rev}\n")
        for junk in dest.glob(".cache"):
            import shutil

            shutil.rmtree(junk, ignore_errors=True)
        print(f"{repo}@{rev[:10]} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
