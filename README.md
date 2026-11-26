# Blood Cell Classification with Classical Machine Learning

![Python](https://img.shields.io/badge/Python-3.12.7-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5.1-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-green)

## Overview

This project explores the application of classical machine learning to microscopic blood cell image classification.

The main objective is to investigate whether handcrafted morphological, color and texture features can effectively distinguish eight blood cell types without using deep learning.

The project covers the complete machine learning workflow, from image processing and feature extraction to model optimization and performance evaluation.

## Dataset

The project uses the Peripheral Blood Cell (PBC) dataset:

- 17,092 RGB microscopy images
- 8 blood cell classes
- 360 × 363 pixel resolution
- stratified 80/20 train-test split

Dataset source: [PBC Dataset – Mendeley Data](https://doi.org/10.17632/snkd93bnjr.1)

## Project workflow

1. Image segmentation using Otsu thresholding
2. Extraction of 18 morphological, color and texture features
3. Exploratory data analysis
4. Feature selection (18 → 13 features)
5. Comparison of eight classical machine learning models
6. Hyperparameter optimization using GridSearchCV
7. Ensemble learning using hard voting and stacking
8. Final evaluation and error analysis

## Results

The best-performing model was a tuned Support Vector Machine (SVM) with an RBF kernel and 13 selected features.

| Metric | Score |
|---|---:|
| Accuracy | **92.98%** |
| Balanced accuracy | **92.30%** |
| Macro-F1 | **92.54%** |

Best hyperparameters: `C = 50`, `gamma = 0.05`.

Neither hard voting nor stacking improved the performance of the single tuned SVM.

## Technologies

- Python
- NumPy
- pandas
- scikit-learn
- scikit-image
- Matplotlib
- seaborn

## Project structure

- `ML/main.py` – complete machine learning workflow
- `ML/feature_extraction.py` – image segmentation and feature extraction
- `ML/requirements.txt` – required library versions
- `ML/results/` – experimental results and visualizations

## Getting started

Clone the repository:

```bash
git clone https://github.com/mnowakowska1/ML-project.git
cd ML-project/ML
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

If the saved feature and label arrays are available, image processing is automatically skipped. Otherwise, the original PBC dataset is required.

## Conclusions

The results demonstrate that classical machine learning combined with handcrafted image features can achieve strong classification performance without deep learning.

Hyperparameter optimization substantially improved SVM performance, while additional model complexity through ensemble learning did not provide further improvement.

This project is intended for educational and research purposes, not clinical diagnosis. The reported results come from a holdout set also used for model comparison.
