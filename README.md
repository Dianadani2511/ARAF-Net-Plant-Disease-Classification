# ARAF-Net: Attention-Refined Multi-Scale Feature Learning for Plant Disease Classification

## Overview

**ARAF-Net** is a deep learning framework designed for automated plant disease classification from leaf images. The proposed approach focuses on learning discriminative visual representations through **attention-refined feature learning and multi-scale feature extraction**.

The repository provides the implementation, experimental notebooks, configuration details, and evaluation resources required to reproduce the experiments reported in the associated research paper.

## Research Objectives

The main objectives of this work are to:

* Develop an attention-refined deep learning architecture for plant disease classification.
* Capture disease-related visual patterns at multiple feature scales.
* Improve the discriminative representation of plant leaf features.
* Evaluate the proposed model using standard classification metrics.
* Provide a reproducible implementation for research and academic use.

## Key Contributions

The major components investigated in ARAF-Net include:

* **Attention-Refined Feature Learning** for emphasizing informative image regions.
* **Multi-Scale Feature Learning** for capturing disease characteristics at different spatial levels.
* **Residual Feature Learning** to support effective deep feature extraction.
* **End-to-End Plant Disease Classification** using a deep neural network.
* **Reproducible Experimental Pipeline** covering training, validation, and testing.

## Dataset

The experiments use the **PlantVillage** dataset.

Original dataset:

* PlantVillage Dataset: https://github.com/spMohanty/PlantVillage-Dataset

The repository will document the exact classes, dataset split, preprocessing procedure, and experimental configuration used in the final study.

> **Note:** The dataset used for experimentation should be obtained from its original source and used according to its applicable license and terms.

## ARAF-Net Architecture

The proposed ARAF-Net architecture is designed to combine:

1. Deep residual feature extraction
2. Multi-scale feature representation
3. Attention-based feature refinement
4. Feature fusion
5. Final disease classification

An architecture diagram will be added to this repository after the final model architecture is confirmed.

```text
Input Leaf Image
       │
       ▼
Feature Extraction
       │
       ▼
Multi-Scale Feature Learning
       │
       ▼
Attention Refinement
       │
       ▼
Feature Fusion
       │
       ▼
Classification Head
       │
       ▼
Plant Disease Class
```

## Repository Structure

```text
ARAF-Net-Plant-Disease-Classification/
│
├── README.md
├── requirements.txt
├── CITATION.cff
├── LICENSE
│
├── notebooks/
│   ├── data_preparation.ipynb
│   ├── training.ipynb
│   └── evaluation.ipynb
│
├── models/
│   └── araf_net.py
│
├── scripts/
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── figures/
│   ├── confusion_matrix/
│   └── metrics/
│
└── checkpoints/
    └── README.md
```

## Requirements

The implementation is based on Python and PyTorch.

Example environment:

```text
Python 3.x
PyTorch
Torchvision
NumPy
Pandas
Scikit-learn
Matplotlib
Pillow
Hugging Face Datasets
```

The final `requirements.txt` will contain the exact package versions used for the published experiments.

## Installation

Clone the repository after it has been published:

```bash
git clone https://github.com/Dianadani2511/ARAF-Net-Plant-Disease-Classification.git
cd ARAF-Net-Plant-Disease-Classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training

The training pipeline will include:

* Dataset loading
* Image preprocessing
* Data splitting
* Model initialization
* Model training
* Validation
* Model checkpointing

Example:

```bash
python scripts/train.py
```

The exact training configuration will be documented with the final experimental setup.

## Evaluation

The trained model can be evaluated using:

```bash
python scripts/evaluate.py
```

The evaluation will report standard classification metrics, including:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

## Experimental Results

Final experimental results will be added after the complete experimental configuration and model implementation have been finalized.

| Metric    |       ARAF-Net |
| --------- | -------------: |
| Accuracy  | To be reported |
| Precision | To be reported |
| Recall    | To be reported |
| F1-score  | To be reported |

Additional comparisons with appropriate baseline models will be included where applicable.

## Reproducibility

To support reproducible research, the repository will provide:

* Dataset information
* Preprocessing details
* Model implementation
* Training configuration
* Evaluation scripts
* Required software packages
* Experimental results
* Random seed information, where applicable

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

A formal citation entry will be provided through `CITATION.cff` once the paper and repository metadata are finalized.

## License

The appropriate open-source license will be specified before publication of the repository.

## Contact

**Dr. D. Paulin Diana Dani**
Assistant Professor
Department of Computer Science and Engineering
Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology
Chennai, India
