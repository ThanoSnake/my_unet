#!/usr/bin/env bash
#
# End-to-end Spleen (Task09) MC-Dropout + calibration pipeline for a GCP
# Deep-Learning VM (L4, 24 GB).
#
# RECOMMENDED USAGE (bootstrap): copy just this file to the VM home and run it:
#     bash run_all_spleen.sh
# The FIRST run clones the repo, downloads the dataset and preprocesses; EVERY
# LATER run only `git pull`s the repo (refresh), skips the download + preprocessing
# if already present, and skips training for any model whose checkpoint exists
# (set FORCE_RETRAIN=1 to retrain).
#
# GPU is used for train + test + uncertainty; the data loaders use WORKERS CPU
# processes for augmentation/IO. Everything is overridable from the environment:
#     FOLDS="0 1 2 3 4" WORKERS=8 BATCH=8 bash run_all_spleen.sh
#
set -euo pipefail

# ----------------------------- configuration ---------------------------------
REPO_URL="${REPO_URL:-https://github.com/ThanoSnake/my_unet.git}"
REPO_DIR="${REPO_DIR:-my_unet}"
TASK="${TASK:-Task09_Spleen}"
DATA_TAR_URL="${DATA_TAR_URL:-https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar}"

FOLDS="${FOLDS:-0}"          # set to "0 1 2 3 4" for full 5-fold CV
SIZE="${SIZE:-256}"          # preprocessing in-plane size (baked into the npy)
PATCH="${PATCH:-256}"        # training/eval slice size (must be <= SIZE)
BATCH="${BATCH:-8}"          # 256x256 fits batch 8 on an L4 with bf16 autocast
WORKERS="${WORKERS:-8}"      # data-loader CPU workers (set 0 on a fork/CUDA error)
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-12}"
DROPOUT="${DROPOUT:-0.4}"
MC="${MC:-30}"               # MC stochastic passes T
CALW="${CALW:-1.0}"          # lambda for the train-time SB-ECE term
FGMARGIN="${FGMARGIN:-3}"    # empty axial slices kept around the organ (train)
OUT_DIR="${OUT_DIR:-results}"
PYTHON="${PYTHON:-python}"

# ----------------------------- clone (1st time) / pull (later) ---------------
if [ -f "run_preprocessing_mc_spleen.py" ]; then
    echo "==> already inside the repo: $(pwd)"
    if [ "${SKIP_PULL:-0}" != "1" ]; then
        echo "==> refreshing (git pull)"; git pull --ff-only || echo "   (pull skipped/failed; continuing)"
    fi
elif [ -d "$REPO_DIR" ]; then
    echo "==> refreshing $REPO_DIR (git pull)"
    git -C "$REPO_DIR" pull --ff-only || echo "   (pull skipped/failed; continuing)"
    cd "$REPO_DIR"
else
    echo "==> cloning $REPO_URL"
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi
echo "==> working dir: $(pwd)"

# ----------------------------- python deps -----------------------------------
# torch is intentionally omitted so the VM's pre-installed CUDA build is untouched.
echo "==> installing python deps (keeping the pre-installed CUDA torch)"
$PYTHON -m pip install --quiet --no-input \
    medpy nibabel SimpleITK "batchgenerators==0.21" scipy scikit-image matplotlib pandas
$PYTHON - <<'PY'
import torch
print("==> torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"))
PY
# fail early & clearly if a core dep won't import (e.g. batchgenerators vs numpy)
$PYTHON - <<'PY'
import importlib
for m in ("batchgenerators", "medpy", "nibabel", "scipy", "skimage", "matplotlib"):
    importlib.import_module(m)
print("==> core deps import OK")
PY

# ----------------------------- download dataset (1st time) -------------------
mkdir -p data
if [ ! -d "data/$TASK/imagesTr" ]; then
    echo "==> downloading $TASK"
    wget -c "$DATA_TAR_URL" -O "data/${TASK}.tar"
    echo "==> extracting"
    tar -xf "data/${TASK}.tar" -C data
    find "data/$TASK" -name '._*' -delete 2>/dev/null || true
else
    echo "==> $TASK already present, skipping download"
fi

export TASK
export DATA_DIR="$(pwd)/data/$TASK"
echo "==> TASK=$TASK  DATA_DIR=$DATA_DIR"

# ----------------------------- preprocessing (1st time) ----------------------
if [ ! -f "$DATA_DIR/splits.pkl" ]; then
    echo "==> preprocessing (axial orient + body-crop + resize ${SIZE})"
    $PYTHON run_preprocessing_mc_spleen.py --size "$SIZE"
else
    echo "==> preprocessed data + splits present, skipping"
fi

mkdir -p "$OUT_DIR"

# ----------------------------- experiments -----------------------------------
common_eval="--patch-size $PATCH --batch-size $BATCH --num-workers $WORKERS --dropout-p $DROPOUT --out-dir $OUT_DIR"

for f in $FOLDS; do
    echo ""
    echo "################  FOLD $f  ################"

    # ---- A. baseline MC-Dropout (Dice + CE) ----
    if [ "${FORCE_RETRAIN:-0}" = "1" ] || [ ! -f "$OUT_DIR/mcdropout_f${f}_best.pth" ]; then
        echo "==> [A] train baseline mcdropout (fold $f)"
        $PYTHON train_mc_spleen.py --fold "$f" --tag mcdropout \
            --patch-size "$PATCH" --batch-size "$BATCH" --num-workers "$WORKERS" \
            --epochs "$EPOCHS" --patience "$PATIENCE" --dropout-p "$DROPOUT" \
            --fg-margin "$FGMARGIN" --out-dir "$OUT_DIR"
    else
        echo "==> [A] mcdropout f$f checkpoint exists -> skip training (FORCE_RETRAIN=1 to retrain)"
    fi

    echo "==> [A] test + uncertainty(raw) + temperature + uncertainty(+T)"
    $PYTHON test_mc_spleen.py        --fold "$f" --tag mcdropout $common_eval
    $PYTHON uncertainty_mc_spleen.py --fold "$f" --tag mcdropout $common_eval --mc-samples "$MC" --temperature 1.0 --save-volumes
    $PYTHON calibrate_mc_spleen.py   --fold "$f" --tag mcdropout $common_eval
    $PYTHON uncertainty_mc_spleen.py --fold "$f" --tag mcdropout $common_eval --mc-samples "$MC" --save-volumes

    # ---- B. train-time calibrated MC-Dropout (Dice + CE + lambda*SB-ECE) ----
    if [ "${FORCE_RETRAIN:-0}" = "1" ] || [ ! -f "$OUT_DIR/mcdropout_cal_f${f}_best.pth" ]; then
        echo "==> [B] train calibrated mcdropout_cal (fold $f)"
        $PYTHON train_mc_recalibrate_spleen.py --fold "$f" --tag mcdropout_cal \
            --patch-size "$PATCH" --batch-size "$BATCH" --num-workers "$WORKERS" \
            --epochs "$EPOCHS" --patience "$PATIENCE" --dropout-p "$DROPOUT" \
            --cal-weight "$CALW" --fg-margin "$FGMARGIN" --out-dir "$OUT_DIR"
    else
        echo "==> [B] mcdropout_cal f$f checkpoint exists -> skip training (FORCE_RETRAIN=1 to retrain)"
    fi

    echo "==> [B] test + uncertainty(raw) + temperature + uncertainty(+T)"
    $PYTHON test_mc_spleen.py        --fold "$f" --tag mcdropout_cal $common_eval
    $PYTHON uncertainty_mc_spleen.py --fold "$f" --tag mcdropout_cal $common_eval --mc-samples "$MC" --temperature 1.0 --save-volumes
    $PYTHON calibrate_mc_spleen.py   --fold "$f" --tag mcdropout_cal $common_eval
    $PYTHON uncertainty_mc_spleen.py --fold "$f" --tag mcdropout_cal $common_eval --mc-samples "$MC" --save-volumes
done

# ----------------------------- summary ---------------------------------------
echo ""
echo "==> summary: Dice, and foreground/macro ECE + temperature T across the four settings"
OUT_DIR="$OUT_DIR" FOLDS="$FOLDS" $PYTHON - <<'PY'
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

echo ""
echo "==> DONE. Weights + *_scores.json in $OUT_DIR/ ; panels, reliability diagrams and *_uncertainty.json in $OUT_DIR/uncertainty/"
