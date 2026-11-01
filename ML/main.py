import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

# check if file with all 18 features already exists
if not os.path.exists("pbc_features_18.npy"):
    print("extracting 18 features for the first time, please wait...")

    data_path = "/Users/marysianowakowska/politechnika/5 semestrer/intro to ML/project/ML/PBC_dataset_normal_DIB"

    classes = [
        'basophil', 'eosinophil', 'erythroblast', 'ig',
        'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
    ]

    images = []
    labels = []

    # load images
    for id, cls in enumerate(classes):
        folder = os.path.join(data_path, cls)

        if not os.path.isdir(folder):
            print(f"missing folder: {cls} at {folder}")
            continue

        for file in os.listdir(folder):
            if file.endswith('.jpg') and not file.startswith('.'):
                img_path = os.path.join(folder, file)
                try:
                    img = imread(img_path)
                    images.append(img)
                    labels.append(id)
                except Exception as e:
                    print(f"error loading {file}: {e}")

    x_data = np.array(images, dtype=object)
    y_data = np.array(labels)
    print(f"loaded {len(x_data)} images")

    features_list = []

    # extract all 18 features
    for id, img in enumerate(x_data):
        gray_img = rgb2gray(img)

        # 1. masks for whole cell, nucleus and cytoplasm
        thresh = threshold_otsu(gray_img)
        cell_mask = gray_img < thresh
        nucleus_mask = gray_img < (thresh * 0.7)
        cyto_mask = cell_mask & (~nucleus_mask)

        cell_props = regionprops(cell_mask.astype(int))
        nuc_props = regionprops(nucleus_mask.astype(int))

        if len(cell_props) > 0 and len(nuc_props) > 0:
            c_reg = max(cell_props, key=lambda r: r.area)
            n_reg = max(nuc_props, key=lambda r: r.area)

            # --- 8 morphological features ---
            # 4 size features
            cell_area = c_reg.area
            cell_diam = c_reg.equivalent_diameter_area
            nuc_area = n_reg.area
            nuc_diam = n_reg.equivalent_diameter_area

            # 3 shape features (nucleus)
            solidity = n_reg.solidity
            eccentricity = n_reg.eccentricity
            nuc_perim = n_reg.perimeter
            circularity = (4 * np.pi * nuc_area) / (nuc_perim ** 2) if nuc_perim > 0 else 0

            # 1 ratio feature (nucleus to cytoplasm)
            cyto_area = np.sum(cyto_mask)
            nc_ratio = nuc_area / cyto_area if cyto_area > 0 else 0

            # --- 6 color features (mean rgb) ---
            n_r = img[nucleus_mask, 0].mean() if np.any(nucleus_mask) else 0
            n_g = img[nucleus_mask, 1].mean() if np.any(nucleus_mask) else 0
            n_b = img[nucleus_mask, 2].mean() if np.any(nucleus_mask) else 0

            c_r = img[cyto_mask, 0].mean() if np.any(cyto_mask) else 0
            c_g = img[cyto_mask, 1].mean() if np.any(cyto_mask) else 0
            c_b = img[cyto_mask, 2].mean() if np.any(cyto_mask) else 0

            # --- 3 texture features (glcm) ---
            gray_8bit = (gray_img * 255).astype(np.uint8)
            glcm = graycomatrix(gray_8bit, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]

            # --- 1 technical flag ---
            flag = 1

            # append all 18 features
            features_list.append([
                cell_area, cell_diam, nuc_area, nuc_diam,
                solidity, eccentricity, circularity, nc_ratio,
                n_r, n_g, n_b, c_r, c_g, c_b,
                contrast, homogeneity, energy, flag
            ])
        else:
            # 18 zeros if detection fails
            features_list.append([0] * 18)

    x_features = np.array(features_list)
    print(f"extracted {x_features.shape[1]} features for {len(x_features)} images")

    # save to disk for fast loading next time
    np.save("pbc_features_18.npy", x_features)
    np.save("pbc_labels.npy", y_data)

# load saved 18 features and labels
x_features = np.load("pbc_features_18.npy")
y_data = np.load("pbc_labels.npy")

# split data: 80% train, 20% test
x_train, x_test, y_train, y_test = train_test_split(
    x_features, y_data, test_size=0.2, random_state=42
)

# train first working model on all 18 features
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# first classification on test data
y_pred = rf_model.predict(x_test)

# evaluation metrics
b_acc = balanced_accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f"balanced accuracy: {b_acc:.4f}")
print(f"macro-f1: {macro_f1:.4f}")

# rank all 18 features by importance
importances = rf_model.feature_importances_

feature_labels = [
    'cell_area', 'cell_diameter', 'nucleus_area', 'nucleus_diameter',
    'solidity', 'eccentricity', 'circularity', 'nc_ratio',
    'nucleus_r', 'nucleus_g', 'nucleus_b',
    'cytoplasm_r', 'cytoplasm_g', 'cytoplasm_b',
    'glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'technical_flag'
]

ranking = pd.DataFrame({'feature': feature_labels, 'importance': importances})
ranking = ranking.sort_values(by='importance', ascending=False)

print("\nfeature importance ranking (all 18 features):")
print(ranking.to_string(index=False))

# plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(ranking['feature'], ranking['importance'], color='steelblue')
plt.gca().invert_yaxis()  # put the most important feature at the top
plt.xlabel('importance')
plt.ylabel('feature')
plt.title('feature importance ranking')
plt.tight_layout()
plt.show()


# convert array to dataframe for easy feature selection
df = pd.DataFrame(x_features, columns=feature_labels)

# remove redundant diameters, zero-importance flag and weakest features
drop_cols = [
    'technical_flag',      # zero importance
    'nucleus_diameter',    # mathematical redundancy with nucleus_area
    'cell_diameter',       # mathematical redundancy with cell_area
    'glcm_energy',         # lowest importance on the plot
    'glcm_homogeneity'     # very low importance
]

x_selected = df.drop(columns=drop_cols).values

# split data with selected features
x_train_sel, x_test_sel, y_train, y_test = train_test_split(
    x_selected, y_data, test_size=0.2, random_state=42
)

# train model on reduced feature set (13 features instead of 18)
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(x_train_sel, y_train)

# test classification after selection
y_pred_sel = rf_selected.predict(x_test_sel)

b_acc_sel = balanced_accuracy_score(y_test, y_pred_sel)
macro_f1_sel = f1_score(y_test, y_pred_sel, average='macro')

print(f"\nselected features count: {x_selected.shape[1]}")
print(f"balanced accuracy after selection: {b_acc_sel:.4f}")
print(f"macro-f1 after selection: {macro_f1_sel:.4f}")


#svm


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. SVM z 13 cechami (po usunięciu redundancji i zerowej flagi)
x_train_sel, x_test_sel, y_train, y_test = train_test_split(
    x_selected, y_data, test_size=0.2, random_state=42
)

# SVM bardzo lubi przeskalowane dane (StandardScaler), szczególnie przy wielu cechach
svm_model = make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=42))
svm_model.fit(x_train_sel, y_train)
y_pred_svm = svm_model.predict(x_test_sel)

print("--- SVM (13 selected features) ---")
print(f"balanced accuracy: {balanced_accuracy_score(y_test, y_pred_svm):.4f}")
print(f"macro-f1: {f1_score(y_test, y_pred_svm, average='macro'):.4f}")

# 2. Sprawdźmy jeszcze mocniejsze ograniczenie: tylko TOP 8 najważniejszych cech z wykresu
# (nc_ratio, nucleus_area, nucleus_r, nucleus_diameter, nucleus_g, nucleus_b, cytoplasm_b, glcm_contrast)
x_top8 = x_selected[:, :8] # wybieramy pierwszych 8 kolumn (bo dataframe był posortowany malejąco)

x_train_8, x_test_8, y_train_8, y_test_8 = train_test_split(
    x_top8, y_data, test_size=0.2, random_state=42
)

svm_model_8 = make_pipeline(StandardScaler(), SVC(kernel='rbf', random_state=42))
svm_model_8.fit(x_train_8, y_train_8)
y_pred_8 = svm_model_8.predict(x_test_8)

print("\n--- SVM (TOP 8 features) ---")
print(f"balanced accuracy: {balanced_accuracy_score(y_test_8, y_pred_8):.4f}")
print(f"macro-f1: {f1_score(y_test_8, y_pred_8, average='macro'):.4f}")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# 1. Generujemy macierz pomyłek dla modelu SVM z 13 cechami
cm = confusion_matrix(y_test, y_pred_svm)

# 2. Definiujemy nazwy klas z Twojego zbioru PBC
class_names = [
    'basophil', 'eosinophil', 'erythroblast', 'ig',
    'lymphocyte', 'monocyte', 'neutrophil', 'platelet'
]

# 3. Rysujemy ładną macierz pomyłek
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title('Confusion Matrix - SVM (13 features)')
plt.tight_layout()
plt.show()