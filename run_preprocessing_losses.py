#
# One-time preprocessing for the loss-experiment track (image, label, + boundary SDF)
#
# Produces the (2 + K)-channel npy (see datasets/preprocessing_boundary.py) plus
# the seeded k-fold splits.pkl, mirroring run_preprocessing_mc.py. clDice needs no
# precompute, so this single dataset serves BOTH loss combos
# (ftversky_ce_boundary and cldice_dice_ce).
#
# Kaggle (writes into DATA_DIR -> a WRITABLE copy of the raw task):
#   DATA_DIR=/kaggle/working/data/Task08_HepaticVessel TASK=Task08_HepaticVessel \
#       python run_preprocessing_losses.py
#

import config
from datasets.preprocessing_boundary import preprocess_data_boundary
from datasets.create_splits import create_splits

if __name__ == "__main__":
    preprocess_data_boundary(root_dir=str(config.DATA_DIR), modality=config.MODALITY,
                             channel=config.CHANNEL, num_classes=config.NUM_CLASSES)
    create_splits(output_dir=str(config.DATA_DIR), image_dir=str(config.PREPROCESSED_DIR))
    print(f"Boundary preprocessing done "
          f"({config.NUM_CLASSES - 1} distance-map channel(s), classes={config.NUM_CLASSES}).")
