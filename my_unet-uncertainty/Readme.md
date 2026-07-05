# Morphological U-Net

The project investigates whether morphological variations of the U-Net improves segmentation over the unaltered baseline. This is done in the context of the Medical Segmentation Decathlon. Morphological variants include top and/or bottom hat residuals as additional input channels, either precomputed with a fixed structuring element or computed by a morphological block with a trainable structuring element and a morphological loss function added to the preexisting loss.

## Setup & Run

Data available in http://medicaldecathlon.com. Place in `./data/` or add path to config.

Install requirements :
```
pip install -r requirements.txt
```

Preprocess data :
```
python3 run_preprocessing.py
```

Train and test (5-fold cross-validation) :

Baseline Model :
```
for f in 0 1 2 3 4; do
    python3 train_eval.py --tag baseline --fold $f
done
```

Static Residuals (one or both) :
```
for f in 0 1 2 3 4; do
    python3 train_eval.py --tag tophat --tophat --fold $f
done
```

Trainable Residuals (one or both) :
```
for f in 0 1 2 3 4; do
    python3 train_eval.py --tag morphblock --morph-block --tophat --fold $f
done
```

Morph Loss Function (could add residuals, fixed or trainable) :
```
for f in 0 1 2 3 4; do
    python3 train_eval.py --tag morphloss --morph-loss --fold $f
done
```

Mean score over folds :
```
for t in baseline tophat morphblock morphloss; do
    python3 train_eval.py --fold-mean $t
done
```

Compare results :
```
python3 train_eval.py --compare baseline_mean_scores.json tophat_mean_scores.json morphblock_mean_scores.json morphloss_mean_scores.json
```

## Attribution & License

Derived from the MIC-DKFZ [`basic_unet_example`](https://github.com/MIC-DKFZ/basic_unet_example). 

Copyright © German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC). Licensed under the Apache License 2.0. 

This is a modified derivative work, original per-file copyright headers are retained. [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## References

[1] Antonelli, M., et al. "The Medical Segmentation Decathlon." Nature Communications, 2022.

[2] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation. " International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.

[3] Çiçek, Özgün, et al. "3D U-Net: learning dense volumetric segmentation from sparse annotation. "International conference on medical image computing and computer-assisted intervention. Springer, Cham, 2016.
