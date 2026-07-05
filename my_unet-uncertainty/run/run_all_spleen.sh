#!/usr/bin/env bash
#
# Self-contained overnight MC-Dropout UNCERTAINTY + CALIBRATION experiment on
# Task09 Spleen (GPU VM, e.g. L4 24GB).
#
# It does EVERYTHING itself: git clone (branch 'uncertainty_spleen', sub-folder
# 'my_unet-uncertainty/') -> deps -> download Spleen -> preprocess (axial orient +
# body-crop + resize) -> for each fold train + test + calibrate + dump uncertainty
# for the two models (baseline 'mcdropout' and train-time-calibrated 'mcdropout_cal')
# -> print a Dice / ECE / temperature summary. Nothing morphological is used.
#
# You do NOT clone anything by hand -- this script clones for you. Put it on the VM
# and launch it so it survives an SSH disconnect. Easiest bootstrap:
#   curl -O https://raw.githubusercontent.com/ThanoSnake/my_unet/uncertainty_spleen/my_unet-uncertainty/run/run_all_spleen.sh
#   nohup bash run_all_spleen.sh &        # progress: tail -f ~/spleen-run/run_*.log
# In the morning copy off the VM:  ~/spleen-run/repo/my_unet-uncertainty/results/
#
# Assumes PyTorch (with CUDA) is already installed (standard on a Deep Learning VM);
# everything else is pip-installed below. torch/numpy are NOT reinstalled.

set -uo pipefail   # -u: error on unset vars; pipefail through tee. NOT -e: a 3am step
                   # failure must not throw away the runs that already finished.

# ============================ CONFIG (edit these) ============================
REPO_URL="https://github.com/ThanoSnake/my_unet.git"
BRANCH="uncertainty_spleen"                 # branch that holds the *_spleen work
PROJECT_SUBDIR="my_unet-uncertainty"        # sub-folder inside the repo
WORKDIR="${WORKDIR:-$HOME/spleen-run}"      # where the repo + logs live
TASK="Task09_Spleen"
DATA_TAR_URL="https://msd-for-monai.s3-us-west-2.amazonaws.com/${TASK}.tar"

FOLDS="0"                                   # "0" tonight; "0 1 2 3 4" for full CV (much longer)

# preprocessing / model / GPU knobs (L4 24GB)
SIZE=256                  # preprocessing in-plane size (baked into the npy)
PATCH=256                 # train/eval slice size (must be <= SIZE)
BATCH=8                   # 256x256 fits batch 8 on an L4 (bf16); try 12/16 if no OOM, 4 if OOM
WORKERS="${WORKERS:-8}"   # workers for the (expensive) TRAIN augmentation only; val/test run
                          # single-process, so a second forked pool never aborts CUDA. If the
                          # TRAIN pool itself ever aborts workers, drop to WORKERS=0.
EPOCHS=150
PATIENCE=12
DROPOUT=0.4
MC=30                     # MC stochastic passes T
CALW=1.0                  # lambda for the train-time SB-ECE term (try 5, 10 for a stronger prior)
FGMARGIN=3                # empty axial slices kept around the organ (train balance)
# ============================================================================

mkdir -p "$WORKDIR"
LOG="$WORKDIR/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1                 # log EVERYTHING (incl. clone) to console AND file

echo "################ spleen uncertainty  $(date '+%F %T') ################"
echo "repo=$REPO_URL  branch=$BRANCH  subdir=$PROJECT_SUBDIR  workdir=$WORKDIR"
echo "task=$TASK folds='$FOLDS'  size=$SIZE patch=$PATCH batch=$BATCH workers=$WORKERS mc=$MC calw=$CALW"
echo "log -> $LOG"

run() {   # run() "label" cmd... : header + timing, CONTINUE on failure (overnight-safe)
    local label="$1"; shift
    echo ""; echo "===== [$(date '+%F %T')] $label ====="
    local t0=$SECONDS
    "$@"; local rc=$?
    echo "----- $label done in $((SECONDS - t0))s (exit $rc) -----"
    [ $rc -eq 0 ] || echo "!!! FAILED: $label (continuing) !!!"
    return 0
}

# ---- 1. clone (or force-update) the repo on the uncertainty_spleen branch ----
REPO_DIR="$WORKDIR/repo"
if [ -d "$REPO_DIR/.git" ]; then
    echo "repo present -> force-updating to origin/$BRANCH (tracked code only; data/ & results/ untouched)"
    git -C "$REPO_DIR" fetch origin "$BRANCH" \
        && git -C "$REPO_DIR" checkout "$BRANCH" \
        && git -C "$REPO_DIR" reset --hard FETCH_HEAD \
        || echo "WARN: could not update; using existing checkout"
else
    git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR" \
        || { echo "git clone of branch '$BRANCH' failed -> aborting"; exit 1; }
fi
cd "$REPO_DIR/$PROJECT_SUBDIR" || { echo "cannot cd $REPO_DIR/$PROJECT_SUBDIR"; exit 1; }
echo "on branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)  |  cwd: $(pwd)"

# sanity: the *_spleen files must actually be on this branch / sub-folder
[ -f "run_preprocessing_mc_spleen.py" ] || {
    echo "ERROR: run_preprocessing_mc_spleen.py not found in $(pwd)."
    echo "       Push the *_spleen files to branch '$BRANCH' under '$PROJECT_SUBDIR/' first."
    exit 1; }

# ---- 2. package dirs need __init__.py (may be .gitignored -> recreate; harmless if present) ----
for d in datasets datasets/two_dim datasets/three_dim networks loss_functions evaluation utilities; do
    mkdir -p "$d"; touch "$d/__init__.py"
done

# ---- 3. dependencies (torch/numpy assumed preinstalled on the DL VM -> NOT reinstalled) ----
echo ""; echo "===== [$(date '+%F %T')] pip install deps ====="
PKGS="medpy nibabel SimpleITK batchgenerators==0.21 scipy scikit-image matplotlib pandas"
python3 -m pip install --break-system-packages $PKGS 2>/dev/null || python3 -m pip install $PKGS || \
    echo "WARN: pip install returned non-zero; continuing (deps may already be present)"

# fail early & clearly if a core dep won't import (e.g. batchgenerators vs numpy 2.x)
python3 - <<'PY' || { echo "FATAL: core deps import failed (see above). Try: pip install 'numpy<2'  then re-run."; exit 1; }
import torch, importlib
for m in ("batchgenerators", "medpy", "nibabel", "scipy", "skimage", "matplotlib"):
    importlib.import_module(m)
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"))
print("core deps import OK")
PY

# ---- 4. data: reuse if already preprocessed, else download + preprocess ----
export TASK
export DATA_DIR="$PWD/data/$TASK"
PREP_DIR="$DATA_DIR/preprocessed"
SPLITS="$DATA_DIR/splits.pkl"

have_prep() {   # true only if splits + a >=2-channel npy (image+label) exist
    [ -f "$SPLITS" ] || return 1
    local f
    f=$(find "$PREP_DIR" -maxdepth 1 -name '*.npy' -print -quit 2>/dev/null)
    [ -n "$f" ] || return 1
    python3 - "$f" <<'PY'
import sys, numpy as np
sys.exit(0 if np.load(sys.argv[1], mmap_mode="r").shape[0] >= 2 else 1)
PY
}

if have_prep; then
    echo "preprocessed data (>=2 channels) + splits present -> skipping download/preprocess"
else
    if [ ! -d "$DATA_DIR/imagesTr" ]; then
        echo ""; echo "===== [$(date '+%F %T')] download raw $TASK (~1.5 GB) ====="
        mkdir -p data
        ( cd data && curl -O "$DATA_TAR_URL" && tar -xf "${TASK}.tar" ) \
            || { echo "DOWNLOAD FAILED -> aborting"; exit 1; }
    fi
    run "preprocess (axial orient + body-crop + resize $SIZE)" python3 run_preprocessing_mc_spleen.py --size "$SIZE"
    have_prep || { echo "PREPROCESS did not produce a 2-channel npy + splits -> aborting"; exit 1; }
fi

# shared eval args for test / uncertainty / calibrate
EVAL="--patch-size $PATCH --batch-size $BATCH --num-workers $WORKERS --dropout-p $DROPOUT --out-dir results"

# ---- 5. experiments: per fold, both models. skip-if-exists makes re-runs idempotent ----
for FOLD in $FOLDS; do
    echo ""; echo "################  FOLD $FOLD  ################"

    # ===== A. baseline MC-Dropout (Dice + CE) =====
    if [ -f "results/mcdropout_f${FOLD}_best.pth" ]; then
        echo "===== skip train mcdropout f$FOLD (best.pth already exists) ====="
    else
        run "train mcdropout f$FOLD" python3 train_mc_spleen.py --fold "$FOLD" --tag mcdropout \
            --patch-size "$PATCH" --batch-size "$BATCH" --num-workers "$WORKERS" \
            --epochs "$EPOCHS" --patience "$PATIENCE" --dropout-p "$DROPOUT" \
            --fg-margin "$FGMARGIN" --out-dir results
    fi
    run "test mcdropout f$FOLD"             python3 test_mc_spleen.py        --fold "$FOLD" --tag mcdropout $EVAL
    run "uncertainty raw mcdropout f$FOLD"  python3 uncertainty_mc_spleen.py --fold "$FOLD" --tag mcdropout $EVAL --mc-samples "$MC" --temperature 1.0 --save-volumes
    run "calibrate mcdropout f$FOLD"        python3 calibrate_mc_spleen.py   --fold "$FOLD" --tag mcdropout $EVAL
    run "uncertainty +T mcdropout f$FOLD"   python3 uncertainty_mc_spleen.py --fold "$FOLD" --tag mcdropout $EVAL --mc-samples "$MC" --save-volumes

    # ===== B. train-time calibrated MC-Dropout (Dice + CE + lambda*SB-ECE) =====
    if [ -f "results/mcdropout_cal_f${FOLD}_best.pth" ]; then
        echo "===== skip train mcdropout_cal f$FOLD (best.pth already exists) ====="
    else
        run "train mcdropout_cal f$FOLD" python3 train_mc_recalibrate_spleen.py --fold "$FOLD" --tag mcdropout_cal \
            --patch-size "$PATCH" --batch-size "$BATCH" --num-workers "$WORKERS" \
            --epochs "$EPOCHS" --patience "$PATIENCE" --dropout-p "$DROPOUT" \
            --cal-weight "$CALW" --fg-margin "$FGMARGIN" --out-dir results
    fi
    run "test mcdropout_cal f$FOLD"             python3 test_mc_spleen.py        --fold "$FOLD" --tag mcdropout_cal $EVAL
    run "uncertainty raw mcdropout_cal f$FOLD"  python3 uncertainty_mc_spleen.py --fold "$FOLD" --tag mcdropout_cal $EVAL --mc-samples "$MC" --temperature 1.0 --save-volumes
    run "calibrate mcdropout_cal f$FOLD"        python3 calibrate_mc_spleen.py   --fold "$FOLD" --tag mcdropout_cal $EVAL
    run "uncertainty +T mcdropout_cal f$FOLD"   python3 uncertainty_mc_spleen.py --fold "$FOLD" --tag mcdropout_cal $EVAL --mc-samples "$MC" --save-volumes
done

# ---- 6. summary: Dice + foreground/macro ECE + temperature across the four settings ----
echo ""; echo "===== [$(date '+%F %T')] summary ====="
OUT_DIR="results" FOLDS="$FOLDS" python3 - <<'PY' || echo "WARN: summary failed"
import os, json
out_dir = os.environ.get("OUT_DIR", "results")
folds = os.environ.get("FOLDS", "0").split()
unc = os.path.join(out_dir, "uncertainty")

def dice_of(tag, f):
    p = os.path.join(out_dir, f"{tag}_f{f}_scores.json")
    if not os.path.exists(p):
        return None
    mean = json.load(open(p)).get("results", {}).get("mean", {})
    vals = [md["Dice"] for md in mean.values() if isinstance(md, dict) and md.get("Dice") is not None]
    return sum(vals) / len(vals) if vals else None

def ece_of(fn):
    p = os.path.join(unc, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p)); c = d["calibration"]
    return c["foreground_ece"], c["macro_foreground_ece"], d.get("temperature", 1.0)

print(f"\n{'model / fold':<22}{'meanDice':>9}")
for f in folds:
    for tag in ("mcdropout", "mcdropout_cal"):
        d = dice_of(tag, f)
        print(f"{tag+' f'+f:<22}{('%.4f'%d) if d is not None else 'n/a':>9}")

print(f"\n{'setting':<26}{'fg_ECE':>9}{'macro_ECE':>11}{'T':>7}")
for f in folds:
    for label, fn in [
        (f"f{f} baseline raw",   f"mcdropout_f{f}_uncertainty.json"),
        (f"f{f} baseline +T",    f"mcdropout_f{f}_cal_uncertainty.json"),
        (f"f{f} calibrated raw", f"mcdropout_cal_f{f}_uncertainty.json"),
        (f"f{f} calibrated +T",  f"mcdropout_cal_f{f}_cal_uncertainty.json"),
    ]:
        r = ece_of(fn)
        if r:
            macro = f"{r[1]:.4f}" if r[1] is not None else "n/a"
            print(f"{label:<26}{r[0]:>9.4f}{macro:>11}{r[2]:>7.3f}")
PY

cp "$LOG" results/ 2>/dev/null || true       # keep a copy of the log next to the results
echo ""; echo "################ ALL DONE  $(date '+%F %T') ################"
echo "results in: $REPO_DIR/$PROJECT_SUBDIR/results/"
ls -1 results/ 2>/dev/null | sed 's/^/  /'
echo ""
echo "Copy off the VM, e.g.:"
echo "  gcloud compute scp --recurse <user>@<vm>:$REPO_DIR/$PROJECT_SUBDIR/results ./"
