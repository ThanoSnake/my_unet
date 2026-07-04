#!/usr/bin/env bash
#
# Self-contained overnight loss-experiment on Task08 HepaticVessel (GPU VM, e.g. L4).
#
# It does EVERYTHING itself: git clone (branch 'losses') -> deps -> download data ->
# preprocess (boundary SDF, float16) -> train + test the 3 loss combos -> compare.
# Trained weights (<tag>_f<fold>_best.pth), scores (<tag>_f<fold>_scores.json),
# learning curves (<tag>_f<fold>_train.json) and the full log all land in
# <WORKDIR>/repo/results/ .
#
# You do NOT need to clone anything by hand -- this script clones for you.
# Put this file on the VM and launch it so it survives an SSH disconnect:
#   nohup bash run_losses.sh &        # progress: tail -f ~/losses-run/run_*.log
# In the morning copy off the VM:  ~/losses-run/repo/results/
#
# Assumes PyTorch (with CUDA) is already installed (standard on a Deep Learning VM);
# everything else is pip-installed below. torch/numpy are NOT reinstalled.

set -uo pipefail   # -u: error on unset vars; pipefail through tee. NOT -e: a 3am combo failure
                   # must not throw away the runs that already finished.

# ============================ CONFIG (edit these) ============================
REPO_URL="https://github.com/ThanoSnake/my_unet.git"   # <-- your repo (edit if different)
BRANCH="losses"                                         # <-- the branch to run
WORKDIR="${WORKDIR:-$HOME/losses-run}"                  # where the repo + logs live
TASK="Task08_HepaticVessel"

FOLDS="0"                                               # "0" tonight; "0 1 2 3 4" for full CV (much longer)
LOSSES="dice_ce ftversky_ce_boundary cldice_dice_ce"   # baseline + Combo 1 + Combo 2

# training budget / GPU knobs (L4 24GB)
EPOCHS=400
PATIENCE=30
ITERS=250                 # batches per epoch (epoch != full pass)
PATCH=128
BATCH=16                  # L4-safe with clDice/full-slice memory; try 24 (thanasis) if no OOM, 8 if OOM
WORKERS=8                 # data-loading processes; set 0 if you ever hit a CUDA-fork error
VAL_EVERY=3
VAL_CASES=15
VAL_BATCH=8               # full slices per val forward (run_val_dice auto-halves on OOM anyway)
FG_FRACTION=0.33

# loss hyper-params (Combo 1 = Focal-Tversky + CE + lam*Boundary; Combo 2 = clDice + Dice + CE)
TV_A=0.3; TV_B=0.7; FOCAL_G=1.3333333
B_MAX=0.5; B_WARM=40
CL_W=0.5; CL_IT=10
# ============================================================================

mkdir -p "$WORKDIR"
LOG="$WORKDIR/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1                 # log EVERYTHING (incl. clone) to console AND file

echo "################ loss-experiment  $(date '+%F %T') ################"
echo "repo=$REPO_URL  branch=$BRANCH  workdir=$WORKDIR"
echo "task=$TASK folds='$FOLDS' losses='$LOSSES'  batch=$BATCH patch=$PATCH workers=$WORKERS"
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

# ---- 1. clone (or update) the repo on the 'losses' branch ----
REPO_DIR="$WORKDIR/repo"
if [ -d "$REPO_DIR/.git" ]; then
    echo "repo present -> updating branch $BRANCH"
    git -C "$REPO_DIR" checkout "$BRANCH" 2>/dev/null || true
    git -C "$REPO_DIR" pull --ff-only 2>/dev/null || echo "WARN: could not update; using existing checkout"
else
    git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$REPO_DIR" \
        || { echo "git clone of branch '$BRANCH' failed -> aborting"; exit 1; }
fi
cd "$REPO_DIR" || { echo "cannot cd $REPO_DIR"; exit 1; }
echo "on branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

# ---- 2. package dirs need __init__.py (they may be .gitignored -> recreate; harmless if present) ----
for d in datasets datasets/two_dim datasets/three_dim networks loss_functions evaluation utilities; do
    mkdir -p "$d"; touch "$d/__init__.py"
done

# ---- 3. dependencies (torch/numpy assumed preinstalled on the DL VM -> NOT reinstalled) ----
echo ""; echo "===== [$(date '+%F %T')] pip install deps ====="
PKGS="medpy nibabel SimpleITK batchgenerators==0.21 scipy scikit-image pandas"
python3 -m pip install --break-system-packages $PKGS 2>/dev/null || python3 -m pip install $PKGS || \
    echo "WARN: pip install returned non-zero; continuing (deps may already be present)"

# ---- 4. data: reuse if already preprocessed with the boundary maps, else fetch + preprocess ----
export TASK
export DATA_DIR="$REPO_DIR/data/$TASK"
PREP_DIR="$DATA_DIR/preprocessed"
SPLITS="$DATA_DIR/splits.pkl"

have_prep() {   # true only if splits + a >=4-channel npy exist (image+label+2 SDF for HepaticVessel)
    [ -f "$SPLITS" ] || return 1
    local f; f=$(ls "$PREP_DIR"/*.npy 2>/dev/null | head -1) || return 1
    [ -n "$f" ] || return 1
    python3 - "$f" <<'PY'
import sys, numpy as np
sys.exit(0 if np.load(sys.argv[1], mmap_mode="r").shape[0] >= 4 else 1)
PY
}

if have_prep; then
    echo "preprocessed data (>=4 channels) present -> skipping download/preprocess"
else
    if [ ! -d "$DATA_DIR/imagesTr" ]; then
        echo ""; echo "===== [$(date '+%F %T')] download raw $TASK (~7 GB) ====="
        mkdir -p data
        ( cd data && curl -O "https://msd-for-monai.s3-us-west-2.amazonaws.com/${TASK}.tar" && tar -xf "${TASK}.tar" ) \
            || { echo "DOWNLOAD FAILED -> aborting"; exit 1; }
    fi
    run "preprocess (boundary SDF, float16)" python3 run_preprocessing_losses.py
    have_prep || { echo "PREPROCESS did not produce >=4-channel npy -> aborting"; exit 1; }
fi

# ---- 5. train + test each loss combo on each fold ----
for FOLD in $FOLDS; do
  for LOSS in $LOSSES; do
    run "train $LOSS fold$FOLD" python3 train_losses.py --loss "$LOSS" --fold "$FOLD" \
        --epochs "$EPOCHS" --patience "$PATIENCE" --iters-per-epoch "$ITERS" \
        --patch-size "$PATCH" --batch-size "$BATCH" --val-every "$VAL_EVERY" \
        --val-cases "$VAL_CASES" --val-batch "$VAL_BATCH" --fg-fraction "$FG_FRACTION" \
        --num-workers "$WORKERS" --tversky-alpha "$TV_A" --tversky-beta "$TV_B" \
        --focal-gamma "$FOCAL_G" --boundary-max "$B_MAX" --boundary-warmup "$B_WARM" \
        --cldice-weight "$CL_W" --cldice-iters "$CL_IT" --out-dir results
    run "test $LOSS fold$FOLD" python3 test_losses.py --tag "$LOSS" --fold "$FOLD" \
        --num-workers "$WORKERS" --out-dir results
  done
done

# ---- 6. aggregate + compare (Dice / ASSD per class, delta vs the dice_ce baseline) ----
for LOSS in $LOSSES; do
    run "fold-mean $LOSS" python3 train_eval.py --fold-mean "$LOSS"
done
MEAN_FILES=""
for LOSS in $LOSSES; do MEAN_FILES="$MEAN_FILES ${LOSS}_mean_scores.json"; done
run "compare" python3 train_eval.py --compare $MEAN_FILES

cp "$LOG" results/ 2>/dev/null || true       # keep a copy of the log next to the results
echo ""; echo "################ ALL DONE  $(date '+%F %T') ################"
echo "results in: $REPO_DIR/results/"
ls -1 results/ | sed 's/^/  /'
echo ""
echo "Copy the whole folder off the VM, e.g.:"
echo "  gcloud compute scp --recurse <user>@<vm>:$REPO_DIR/results ./"
