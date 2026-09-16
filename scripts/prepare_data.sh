#!/bin/bash
# ---------------------------------------------------------------------------
# Prepare the Harvard-FairVision dataset for the training scripts.
#
# The Hugging Face dataset (harvardairobotics/FairVision) is shipped as one
# `dataset.zip` per disease inside the Hub cache. The training code expects the
# archive to be extracted and to contain the folders `train`, `val` and `test`
# (the archives contain them as `Training`, `Validation` and `Test`).
#
# This script:
#   1. locates the downloaded snapshot in the Hugging Face cache,
#   2. extracts the requested disease archive(s) into DATA_ROOT,
#   3. adds the lowercase `train` / `val` / `test` aliases used by the code,
#   4. copies the `data_summary_<disease>.csv` metadata file next to them.
#
# Usage:
#   ./scripts/prepare_data.sh                 # DR only (default)
#   ./scripts/prepare_data.sh AMD             # AMD only
#   ./scripts/prepare_data.sh "AMD DR Glaucoma"
#   ./scripts/prepare_data.sh DR /path/to/snapshot   # explicit snapshot folder
#
# Environment overrides:
#   DATA_ROOT  Where data is extracted  (default: /home/jupyter-vshein/data/harvard/FairVision)
#   HF_CACHE / HF_HOME  Hugging Face cache root (default: $HOME/.cache/huggingface,
#                       falling back to /home/*/.cache/huggingface)
# ---------------------------------------------------------------------------
set -euo pipefail

DISEASES=${1:-DR}
SNAP_ARG=${2:-}
DATA_ROOT=${DATA_ROOT:-/home/jupyter-vshein/data/harvard/FairVision}

# Locate the downloaded snapshot: an explicit argument wins, then $HF_CACHE,
# then the usual Hugging Face cache locations (including other users' caches).
find_snapshot() {
    if [ -n "${SNAP_ARG}" ]; then
        echo "${SNAP_ARG%/}/"
        return 0
    fi
    local roots=()
    [ -n "${HF_CACHE:-}" ] && roots+=("${HF_CACHE}")
    [ -n "${HF_HOME:-}" ]  && roots+=("${HF_HOME}")
    roots+=("$HOME/.cache/huggingface")
    for d in /home/*/.cache/huggingface; do
        [ -d "$d" ] && roots+=("$d")
    done

    local r hit
    for r in "${roots[@]}"; do
        hit=$(ls -d "${r}"/hub/datasets--harvardairobotics--FairVision/snapshots/*/ 2>/dev/null | sort | tail -1 || true)
        if [ -z "${hit}" ]; then
            hit=$(ls -d "${r}"/datasets--harvardairobotics--FairVision/snapshots/*/ 2>/dev/null | sort | tail -1 || true)
        fi
        if [ -n "${hit}" ]; then
            echo "${hit}"
            return 0
        fi
    done
    return 1
}

SNAP=$(find_snapshot || true)
if [ -z "${SNAP}" ]; then
    echo "ERROR: no FairVision snapshot found. Looked in \$HF_CACHE, \$HF_HOME," >&2
    echo "       \$HOME/.cache/huggingface and /home/*/.cache/huggingface." >&2
    echo "Download the dataset first, e.g.:" >&2
    echo "  hf download harvardairobotics/FairVision --repo-type dataset" >&2
    echo "or pass the snapshot folder explicitly:" >&2
    echo "  ./scripts/prepare_data.sh DR /path/to/snapshot" >&2
    exit 1
fi
echo "Using snapshot: ${SNAP}"

mkdir -p "${DATA_ROOT}"

for DISEASE in ${DISEASES}; do
    ZIP="${SNAP}${DISEASE}/Dataset/dataset.zip"
    if [ ! -f "${ZIP}" ]; then
        echo "WARNING: ${ZIP} not found, skipping ${DISEASE}." >&2
        continue
    fi

    echo "==> Extracting ${DISEASE} to ${DATA_ROOT}/${DISEASE} ..."
    mkdir -p "${DATA_ROOT}/${DISEASE}"
    # -n: never overwrite existing files, so the script can be re-run/resumed cheaply.
    unzip -q -n "${ZIP}" -x '__MACOSX/*' -d "${DATA_ROOT}/${DISEASE}"

    # The code reads <DISEASE>/train, <DISEASE>/val and <DISEASE>/test.
    ln -sfn Training   "${DATA_ROOT}/${DISEASE}/train"
    ln -sfn Validation "${DATA_ROOT}/${DISEASE}/val"
    ln -sfn Test       "${DATA_ROOT}/${DISEASE}/test"

    # Copy the per-disease metadata csv (race / gender / ethnicity / age / ...).
    LOWER=$(echo "${DISEASE}" | tr '[:upper:]' '[:lower:]')
    SUMMARY="${SNAP}${DISEASE}/ReadMe/data_summary_${LOWER}.csv"
    [ -f "${SUMMARY}" ] && cp -f "${SUMMARY}" "${DATA_ROOT}/${DISEASE}/"

    echo "    done: $(ls -d "${DATA_ROOT}/${DISEASE}"/*/ | tr '\n' ' ')"
done

echo
echo "Data is ready under ${DATA_ROOT}"
echo "Run the experiments with:  DATASET_DIR=${DATA_ROOT}/ ./scripts/train_dr_vit.sh"
