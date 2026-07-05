#
# One-time minimal preprocessing for the MC-Dropout track (no morphology)
#
# Produces 2-channel (image, label) npy + the seeded k-fold splits.pkl, exactly
# like run_preprocessing.py but without the top-hat / bottom-hat channels.
#
# Kaggle (writes into DATA_DIR -> point it at a WRITABLE copy of the raw task):
#   DATA_DIR=/kaggle/working/data/Task04_Hippocampus TASK=Task04_Hippocampus \
#       python run_preprocessing_mc.py
#

import config
from datasets.preprocessing_plain import preprocess_data_plain
from datasets.create_splits import create_splits

if __name__ == "__main__":
    preprocess_data_plain(root_dir=str(config.DATA_DIR), modality=config.MODALITY, channel=config.CHANNEL)
    create_splits(output_dir=str(config.DATA_DIR), image_dir=str(config.PREPROCESSED_DIR))
    print("Minimal (2-channel) preprocessing done.")
