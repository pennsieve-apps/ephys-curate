FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /processor

# Pinned Python dependencies (wheels only).
COPY requirements.txt /processor/requirements.txt
RUN pip install --only-binary=:all: -r /processor/requirements.txt

# UnitRefine's pretrained classifiers (MIT), at pinned Hugging Face commits, sha256-checked:
# the run has no network.
COPY scripts/fetch_models.py /processor/scripts/fetch_models.py
RUN python /processor/scripts/fetch_models.py /opt/unitrefine && rm -rf /root/.cache

COPY processor/ /processor/processor/
COPY VERSION LICENSE THIRD_PARTY.md /processor/

# Read-only paths only: on Lambda the image's /tmp and HOME are not there, and the app runs as an
# arbitrary unprivileged user. Writable caches (matplotlib, numba, pynwb) are put under $TMPDIR at
# startup (processor/main.py).
ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    USER=app \
    MPLBACKEND=Agg \
    HF_HUB_OFFLINE=1 \
    UNITREFINE_MODELS_DIR=/opt/unitrefine

# A passwd entry for uid 1000 (getpass) and an import check (caches in a throw-away dir).
RUN echo "app:x:1000:1000:app:/tmp:/usr/sbin/nologin" >> /etc/passwd
USER 1000
RUN d=$(mktemp -d) && MPLCONFIGDIR=$d XDG_CACHE_HOME=$d NUMBA_CACHE_DIR=$d HOME=$d \
    python -c "import matplotlib.pyplot, spikeinterface.full, bombcell, pynwb, skops.io, sklearn; from processor import main" \
    && rm -rf $d

ENTRYPOINT ["python3", "-m", "processor.entry"]
