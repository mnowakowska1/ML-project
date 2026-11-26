# blood cell classification with classical machine learning

Introduction to Machine Learning project, Lodz University of Technology (Modelling and Data Science).
Authors: Maria Nowakowska and Alicja Pietrzak.

We classify microscopic blood cell images from the PBC dataset into eight classes
(basophil, eosinophil, erythroblast, ig, lymphocyte, monocyte, neutrophil, platelet)
using handcrafted morphological, color and texture features and classical machine learning models.

## project files

- `feature_extraction.py` - otsu segmentation and extraction of the 18 original features, with caching
- `main.py` - the whole workflow, from loading the data to the final evaluation
- `requirements.txt` - required python libraries
- `results/` - figures and result tables created by `main.py`

## requirements

The project was developed and tested with Python 3.12.7. The library versions used in our experiments
are pinned in `requirements.txt`.

## how to run

1. install the libraries:

   ```
   pip install -r requirements.txt
   ```

2. put the PBC dataset in the project folder, so that the class folders are inside
   `PBC_dataset_normal_DIB/` (for example `PBC_dataset_normal_DIB/basophil/`).
   if the dataset is stored somewhere else, change `dataset_path` at the top of `main.py`.

3. run `main.py` (in PyCharm: right click on `main.py` and choose run, or from the terminal):

   ```
   python main.py
   ```

If `pbc_features_18.npy` and `pbc_labels.npy` already exist and are valid, feature extraction is skipped
and the images are not needed. If either file is missing or invalid, the features are extracted from the
images again and both files are saved.

Each plot window has to be closed before the script continues. All figures are also saved in `results/`.
The full run includes two grid searches and both ensembles, so it can take a while.

## workflow in main.py

1. loading the PBC dataset (or the saved features)
2. otsu segmentation and extraction of 18 features
3. saving the features and labels to `pbc_features_18.npy` and `pbc_labels.npy`
4. exploratory analysis: dataset size, first five rows, class distribution, correlation matrix
5. initial random forest with all 18 features
6. random forest feature importance
7. feature selection from 18 to 13 features
8. initial svm with the 13 selected features
9. comparison of eight classical models with 5-fold stratified cross-validation
10. hyperparameter tuning with GridSearchCV (svm, histgradientboosting, random forest)
11. additional svm fine-tuning with an expanded grid
12. hard voting ensemble
13. stacking ensemble
14. final comparison, classification report, confusion matrix and class-level performance

## notes on methodology

- all experiments use a stratified 80/20 train-test split with `random_state=42`
- from the baseline comparison onwards, `StandardScaler` is inside a `Pipeline`, so it is fitted only on training data
- the initial svm experiment scaled the whole dataset before splitting, which causes data leakage, because the scaler
  is fitted on data that is later used for testing. it is kept to show how the project developed, and a corrected
  version with a pipeline is run right after it
- GridSearchCV uses macro-f1 as the optimization criterion
- cross-validation scores are calculated on the training set, holdout scores on the 20% test set
- the final model is chosen using the holdout set, so its holdout scores are not a fully independent estimate
- the correlation matrix used for feature selection is calculated on the whole dataset
- if the segmentation of an image fails (empty cell or nucleus mask), the image gets a vector of zeros
  instead of being removed, so the number of rows always equals the number of images

## reproducibility of the image order

Images are read in the order returned by `os.listdir`. Python does not guarantee any particular order here,
it depends on the operating system and file system, and it can change, for example after the dataset is copied
or unpacked again. The order of the rows affects which images end up in the training and test sets, because
`train_test_split` works on row positions.

This is why the saved `pbc_features_18.npy` and `pbc_labels.npy` files are used for all experiments.
If the features are extracted again, the class order and feature order stay the same, but the order of images
within each class may differ, and the results can be slightly different from ours. Small differences in feature
values are also possible with other library versions, because JPEG decoding can differ slightly.
