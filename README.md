# ARAF-Net: Attention-Refined Multi-Scale Feature Learning for Plant Disease Classification

## Overview

**ARAF-Net** is a deep learning architecture for plant disease classification from leaf images. The implementation combines a **ResNet-34 backbone**, **channel attention**, **spatial attention**, and **multi-scale feature fusion** to learn discriminative visual representations.

The current repository contains the core ARAF-Net model implementation in `models.py`.

## Research Objectives

The main objectives of this work are to:

- Develop an attention-refined deep learning architecture for plant disease classification.
- Enhance feature representation using channel and spatial attention.
- Capture disease-related patterns at multiple spatial scales.
- Evaluate the contribution of individual architectural components through ablation experiments.
- Provide a clear and reproducible implementation for research use.

## Key Contributions

ARAF-Net integrates the following components:

- **ResNet-34 backbone** for deep residual feature extraction.
- **Channel Attention** using average- and max-pooled channel descriptors.
- **Spatial Attention** using channel-wise average and maximum descriptors.
- **Multi-Scale Feature Extraction** using parallel 1×1, 3×3, and 5×5 convolutions.
- **Multi-Scale Feature Fusion** using a 1×1 convolution.
- **Global Average Pooling** followed by a fully connected classification layer.
- **Ablation models** to study the contribution of attention and multi-scale feature fusion.

## Dataset

The experiments use the **PlantVillage** dataset.

Original dataset:

- PlantVillage Dataset: https://github.com/spMohanty/PlantVillage-Dataset

The current experimental setup uses seven plant-disease classes from the PlantVillage data used in the study.

> **Note:** The dataset should be obtained from its original source and used according to its applicable license and terms.

## ARAF-Net Architecture

ARAF-Net uses ResNet-34 for feature extraction and progressively refines the extracted representation using attention and multi-scale feature fusion.

### Architecture Flow

```text
Input Leaf Image
       │
       ▼
ResNet-34 Backbone
       │
       ▼
Channel Attention
       │
       ▼
Spatial Attention
       │
       ▼
Multi-Scale Feature Extraction
   ┌────┼────┐
   ▼    ▼    ▼
  1×1  3×3  5×5
  Conv Conv Conv
   └────┼────┘
        ▼
   1×1 Fusion
        │
        ▼
Global Average Pooling
        │
        ▼
Fully Connected Layer
        │
        ▼
Plant Disease Classes
```

### Attention Modules

**Channel Attention:**  
Average pooling and max pooling are used to generate channel descriptors. These descriptors are passed through a shared lightweight transformation to produce channel-wise attention weights.

**Spatial Attention:**  
Channel-wise average and maximum projections are combined and processed using a 7×7 convolution to generate spatial attention weights.

### Multi-Scale Feature Fusion

The refined feature representation is processed through parallel convolutional branches with **1×1, 3×3, and 5×5 kernels**. The resulting features are concatenated and projected through a **1×1 convolution** to obtain the fused representation.

## Ablation Models

The implementation supports the following variants:

| Model | Configuration |
|---|---|
| A | ResNet-34 baseline |
| B | ResNet-34 + Channel Attention |
| C | ResNet-34 + Spatial Attention |
| D | ResNet-34 + Channel + Spatial Attention |
| E | **ARAF-Net:** ResNet-34 + Channel + Spatial Attention + Multi-Scale Feature Fusion |

These variants enable systematic analysis of the contribution of each major component.

## Repository Structure

The current repository is intentionally kept simple:

```text
ARAF-Net-Plant-Disease-Classification/
│
├── README.md
└── models.py
```

## Model Configuration

The implementation currently uses:

| Configuration | Value |
|---|---|
| Backbone | ResNet-34 |
| Input image size | 224 × 224 |
| Image format | RGB |
| Number of classes | 7 |
| Channel-attention reduction factor | 16 |
| Spatial-attention kernel | 7 × 7 |
| Multi-scale kernels | 1 × 1, 3 × 3, 5 × 5 |
| Classifier | Global Average Pooling + Linear |
| Backbone weights | ImageNet weights by default |

## Requirements

The model implementation is based on **Python, PyTorch, and Torchvision**.

A typical environment includes:

```text
Python 3.x
PyTorch
Torchvision
```

Additional packages can be installed as required by the experimental training notebook or data-processing workflow.

## Installation

Clone the repository:

```bash
git clone https://github.com/Dianadani2511/ARAF-Net-Plant-Disease-Classification.git
cd ARAF-Net-Plant-Disease-Classification
```

## Model Usage

The core model can be imported directly from `models.py`:

```python
from models import ARAFNet

model = ARAFNet(num_classes=7, pretrained=False)
print(model)
```

Set `pretrained=True` when ImageNet-pretrained ResNet-34 weights are desired and an internet connection is available for downloading the weights.

## Experimental Results

Final experimental results will be reported after the complete experimental configuration and baseline comparisons have been finalized.

| Metric | ARAF-Net |
|---|---:|
| Accuracy | To be reported |
| Precision | To be reported |
| Recall | To be reported |
| F1-score | To be reported |

The repository will be updated with the final verified results and supporting evaluation outputs when they are ready.

## Reproducibility

For reproducible research, the study should document:

- Dataset source and selected classes.
- Dataset split and preprocessing.
- Model configuration.
- Training hyperparameters.
- Baseline and ablation settings.
- Evaluation metrics.
- Random seed, where applicable.

## Research Paper

**Title:**  
*ARAF-Net: Attention-Refined Multi-Scale Feature Learning for Plant Disease Classification*

**Author:**  
Dr. D. Paulin Diana Dani

**Affiliation:**  
Department of Computer Science and Engineering  
Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology  
Chennai, India

Publication details will be added after the paper is formally published.

## Citation

A formal citation entry will be added when the paper and repository metadata are finalized.

## License

The appropriate license will be specified before final public release of the research code.

## Contact

**Dr. D. Paulin Diana Dani**  
Assistant Professor  
Department of Computer Science and Engineering  
Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology  
Chennai, India
