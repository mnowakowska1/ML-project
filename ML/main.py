import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# loading data and extracting features
classes = [
    'basophil', 'eosinophil', 'erythroblast', 'ig',
    'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
]

feature_labels = [
    'cell_area', 'cell_diameter', 'nucleus_area', 'nucleus_diameter',
    'solidity', 'eccentricity', 'circularity', 'nc_ratio',
    'nucleus_r', 'nucleus_g', 'nucleus_b',
    'cytoplasm_r', 'cytoplasm_g', 'cytoplasm_b',
    'glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'technical_flag'
]

if not os.path.exists("pbc_features_18.npy"):
    print("extracting 18 features...")

    data_path = "/Users/marysianowakowska/politechnika/5 semestrer/intro to ML/project/ML/PBC_dataset_normal_DIB"

    images = []
    labels = []

    for idx, cls in enumerate(classes):
        folder = os.path.join(data_path, cls)

        if not os.path.isdir(folder):
            print(f"folder not found: {cls} in {folder}")
            continue

        for file in os.listdir(folder):
            if file.endswith('.jpg') and not file.startswith('.'):
                img_path = os.path.join(folder, file)
                try:
                    img = imread(img_path)
                    images.append(img)
                    labels.append(idx)
                except Exception as e:
                    print(f"error loading {file}: {e}")

    x_data = np.array(images, dtype=object)
    y_data = np.array(labels)
    print(f"loaded {len(x_data)} images")

    features_list = []

    for idx, img in enumerate(x_data):
        gray_img = rgb2gray(img)

        thresh = threshold_otsu(gray_img)
        cell_mask = gray_img < thresh
        nucleus_mask = gray_img < (thresh * 0.7)
        cyto_mask = cell_mask & (~nucleus_mask)

        cell_props = regionprops(cell_mask.astype(int))
        nuc_props = regionprops(nucleus_mask.astype(int))

        if len(cell_props) > 0 and len(nuc_props) > 0:
            c_reg = max(cell_props, key=lambda r: r.area)
            n_reg = max(nuc_props, key=lambda r: r.area)

            # morphological features
            cell_area = c_reg.area
            cell_diam = c_reg.equivalent_diameter_area
            nuc_area = n_reg.area
            nuc_diam = n_reg.equivalent_diameter_area

            solidity = n_reg.solidity
            eccentricity = n_reg.eccentricity
            nuc_perim = n_reg.perimeter
            circularity = (4 * np.pi * nuc_area) / (nuc_perim ** 2) if nuc_perim > 0 else 0

            cyto_area = np.sum(cyto_mask)
            nc_ratio = nuc_area / cyto_area if cyto_area > 0 else 0

            # color features (mean rgb)
            n_r = img[nucleus_mask, 0].mean() if np.any(nucleus_mask) else 0
            n_g = img[nucleus_mask, 1].mean() if np.any(nucleus_mask) else 0
            n_b = img[nucleus_mask, 2].mean() if np.any(nucleus_mask) else 0

            c_r = img[cyto_mask, 0].mean() if np.any(cyto_mask) else 0
            c_g = img[cyto_mask, 1].mean() if np.any(cyto_mask) else 0
            c_b = img[cyto_mask, 2].mean() if np.any(cyto_mask) else 0

            # texture features (glcm)
            gray_8bit = (gray_img * 255).astype(np.uint8)
            glcm = graycomatrix(gray_8bit, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]

            flag = 1

            features_list.append([
                cell_area, cell_diam, nuc_area, nuc_diam,
                solidity, eccentricity, circularity, nc_ratio,
                n_r, n_g, n_b, c_r, c_g, c_b,
                contrast, homogeneity, energy, flag
            ])
        else:
            features_list.append([0] * 18)

    x_features = np.array(features_list)
    print(f"extracted {x_features.shape[1]} features for {len(x_features)} images")

    np.save("pbc_features_18.npy", x_features)
    np.save("pbc_labels.npy", y_data)


# analysing the dataset
x_features = np.load("pbc_features_18.npy")
y_data = np.load("pbc_labels.npy")

print("shape of data: {}".format(x_features.shape))
print("shape of labels: {}".format(y_data.shape))


# data lookup
data = pd.DataFrame(x_features, columns=feature_labels)
print("\nfirst 5 rows of the dataset:")
print(data.head(5))


# class distrib
labels_series = pd.Series(y_data).map(lambda x: classes[x])
print("\nclass distribution:")
print(labels_series.value_counts(normalize=False))

# correlation matrix
plt.figure(figsize=(14, 12))
sns.heatmap(data.drop(columns=['technical_flag']).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("feature correlation matrix")
plt.tight_layout()
plt.show()


# splitting the dataset into test and train data (18 features)
X_train, X_test, y_train, y_test = train_test_split(
    x_features, y_data, test_size=0.2, random_state=42, stratify=y_data
)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


# initial model (random forest) for feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("\nrandom forest (all 18 features)")
print(f"balanced accuracy: {balanced_accuracy_score(y_test, y_pred_rf):.4f}")
print(f"macro-F1: {f1_score(y_test, y_pred_rf, average='macro'):.4f}")
print("\nclassification report (random forest):")
print(classification_report(y_test, y_pred_rf, target_names=classes))


# feature importance ranking
importances = rf_model.feature_importances_
ranking = pd.DataFrame({'feature': feature_labels, 'importance': importances})
ranking = ranking.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(ranking['feature'], ranking['importance'], color='steelblue')
plt.gca().invert_yaxis()
plt.xlabel('importance')
plt.ylabel('feature')
plt.title('feature importance ranking (random forest)')
plt.tight_layout()
plt.show()


# feature selection (reduce to 13 features)
drop_cols = [
    'technical_flag',
    'nucleus_diameter',
    'cell_diameter',
    'glcm_energy',
    'glcm_homogeneity'
]

data_selected = data.drop(columns=drop_cols)
print("shape of selected data: {}".format(data_selected.shape))


# normalization of data
scaler = StandardScaler()
scaler.fit(data_selected)
x_selected_scaled = scaler.transform(data_selected)

X_train_sel, X_test_sel, y_train, y_test = train_test_split(
    x_selected_scaled, y_data, test_size=0.2, random_state=42, stratify=y_data
)


# SVM model on 13 features
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_sel, y_train)
y_pred_svm = svm_model.predict(X_test_sel)

print("\nSVM (13 selected features)")
print(f"balanced accuracy: {balanced_accuracy_score(y_test, y_pred_svm):.4f}")
print(f"macro-F1: {f1_score(y_test, y_pred_svm, average='macro'):.4f}")


# classification report (SVM - 13 features)
print("\nclassification report (SVM - 13 features):")
print(classification_report(y_test, y_pred_svm, target_names=classes))


# assess results - confusion matrix
cm = confusion_matrix(np.asarray(y_test), np.asarray(y_pred_svm))
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)
plt.title('confusion matrix - SVM (13 features)')
plt.tight_layout()
plt.show()