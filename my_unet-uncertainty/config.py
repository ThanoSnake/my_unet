import os
import json
from pathlib import Path

# file lives at project root
PROJECT_ROOT = Path(__file__).resolve().parent

# which task to use
TASK = os.environ.get("TASK", "Task08_HepaticVessel")


CHANNEL = int(os.environ.get("CHANNEL", "0"))

def resolve_dir(env_var: str, candidates: list[Path], default: Path) -> Path:
    # set manually to override
    val = os.environ.get(env_var)
    if val:
        return Path(val)
    # first existing candidate
    for path in candidates:
        if path.exists():
            return path
    # default fallback
    return default

def _load_meta(data_dir: Path):
    meta = data_dir / "dataset.json"
    if meta.exists():
        d = json.loads(meta.read_text())
        modality = d.get("modality", {"0": "MRI"})
        labels = {int(k): v for k, v in d.get("labels", {}).items() if int(k) != 0}
        return modality, labels
    return {"0": "MRI"}, {}

DATA_DIR = resolve_dir(
    env_var="DATA_DIR",
    candidates=[
        PROJECT_ROOT / "data" / TASK,
    ],
    default=PROJECT_ROOT / "data" / TASK,
)

IMAGES_DIR = DATA_DIR / "imagesTr"
LABELS_DIR = DATA_DIR / "labelsTr"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
SPLITS_FILE = DATA_DIR / "splits.pkl"

MODALITY, LABELS = _load_meta(DATA_DIR)
NUM_CLASSES = (max(LABELS) + 1) if LABELS else 3
