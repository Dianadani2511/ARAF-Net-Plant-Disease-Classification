# ARAF-Net: Attention-Refined Multi-Scale Feature Learning for Plant Disease Classification
## Overview

This repository is being prepared for the research project **ARAF-Net**, a deep learning framework for plant disease classification using attention-refined multi-scale feature learning.

The proposed architecture combines a pretrained ResNet-34 backbone, channel attention, spatial attention, and multi-scale feature fusion.

## Dataset

The experiments use a selected subset of the PlantVillage dataset.

* **Total images:** 9,213
* **Number of disease classes:** 7
* **Training images:** 7,370
* **Validation images:** 921
* **Testing images:** 922
* **Image size used by the models:** 224 × 224 pixels

The seven selected classes are:

1. Corn Common Rust
2. Corn Northern Leaf Blight
3. Potato Early Blight
4. Potato Late Blight
5. Tomato Bacterial Spot
6. Tomato Early Blight
7. Tomato Late Blight

Dataset source: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)

The dataset images are not included in this repository.

## Model Architecture

ARAF-Net consists of the following components:

1. ImageNet-pretrained ResNet-34 backbone
2. Channel attention module
3. Spatial attention module
4. Multi-scale feature fusion using 1 × 1, 3 × 3, and 5 × 5 convolutional branches
5. Global average pooling
6. Seven-class classification layer

## Experimental Results

The following test results were obtained in the reported experiments.

| Experiment | Model                                     | Test Accuracy |
| ---------- | ----------------------------------------- | ------------: |
| A          | ResNet-34 baseline                        |        99.57% |
| B          | ResNet-34 + Channel Attention             |        99.57% |
| C          | ResNet-34 + Spatial Attention             |        98.37% |
| D          | ResNet-34 + Channel and Spatial Attention |        99.35% |
| E          | Full ARAF-Net with Multi-Scale Fusion     |        99.13% |

The results show that the complete ARAF-Net did not outperform the ResNet-34 baseline on this selected dataset. The experiments are intended to examine the contribution of individual architectural components.

These results were obtained using a controlled PlantVillage-based dataset and should not be interpreted as evidence of equivalent performance on field-acquired images.

## Implementation Status

The repository structure and source files are being prepared. Training, evaluation, and reproduction instructions will be documented after the implementation files have been checked against the experiments.

## Citation

**Title:** ARAF-Net: Attention-Refined Multi-Scale Feature Learning for Plant Disease Classification

Author and publication details will be added after the final manuscript information is confirmed.

## License

A software license will be selected before the source code is released for reuse.
