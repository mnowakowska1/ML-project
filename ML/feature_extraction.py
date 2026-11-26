import os
import numpy as np

from skimage import io, color, filters, measure
from skimage.feature import graycomatrix, graycoprops

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


def extract_features(img):
    # segmentation with otsu thresholding
    gray = color.rgb2gray(img)
    threshold = filters.threshold_otsu(gray)

    cell_mask = gray < threshold
    nucleus_mask = gray < 0.7 * threshold
    cytoplasm_mask = cell_mask & ~nucleus_mask

    cell_area = cell_mask.sum()
    nucleus_area = nucleus_mask.sum()
    cytoplasm_area = cytoplasm_mask.sum()

    # zero vector when the segmentation fails, so the image is not removed
    if cell_area == 0 or nucleus_area == 0:
        return [0.0] * len(feature_labels)

    # morphological features
    nucleus_props = measure.regionprops(nucleus_mask.astype(int))[0]

    cell_diameter = np.sqrt(4 * cell_area / np.pi)
    nucleus_diameter = np.sqrt(4 * nucleus_area / np.pi)
    solidity = nucleus_props.solidity
    eccentricity = nucleus_props.eccentricity

    perimeter = nucleus_props.perimeter
    circularity = 4 * np.pi * nucleus_area / perimeter ** 2 if perimeter > 0 else 0.0
    nc_ratio = nucleus_area / cytoplasm_area if cytoplasm_area > 0 else 0.0

    # color features (mean rgb)
    nucleus_r, nucleus_g, nucleus_b = img[nucleus_mask].mean(axis=0)
    if cytoplasm_area > 0:
        cytoplasm_r, cytoplasm_g, cytoplasm_b = img[cytoplasm_mask].mean(axis=0)
    else:
        cytoplasm_r, cytoplasm_g, cytoplasm_b = 0.0, 0.0, 0.0

    # texture features (glcm on the whole grayscale image)
    gray_uint8 = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_contrast = graycoprops(glcm, 'contrast')[0, 0]
    glcm_homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    glcm_energy = graycoprops(glcm, 'energy')[0, 0]

    # technical flag, equal to 1 when the segmentation worked
    technical_flag = 1

    return [
        cell_area, cell_diameter, nucleus_area, nucleus_diameter,
        solidity, eccentricity, circularity, nc_ratio,
        nucleus_r, nucleus_g, nucleus_b,
        cytoplasm_r, cytoplasm_g, cytoplasm_b,
        glcm_contrast, glcm_homogeneity, glcm_energy, technical_flag
    ]


def extract_dataset_features(dataset_path):
    features = []
    labels = []

    for label, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        print(f"extracting features for {class_name}...")

        # files are read in os.listdir order, which is not guaranteed (see README)
        for file_name in os.listdir(class_path):
            # skipping hidden files like .DS_Store
            if file_name.startswith('.') or not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img = io.imread(os.path.join(class_path, file_name))[:, :, :3]
            features.append(extract_features(img))
            labels.append(label)

    return np.array(features, dtype=float), np.array(labels)


def cached_data_is_valid(features, labels):
    if features.ndim != 2 or features.shape[1] != len(feature_labels):
        return False
    if labels.ndim != 1 or len(labels) != len(features) or len(labels) == 0:
        return False
    if not np.all(np.isfinite(features)):
        return False
    # all eight classes have to be present
    return set(np.unique(labels)) == set(range(len(classes)))


def load_or_extract_features(dataset_path, features_file, labels_file):
    if os.path.exists(features_file) and os.path.exists(labels_file):
        try:
            features = np.load(features_file)
            labels = np.load(labels_file)
            if cached_data_is_valid(features, labels):
                print("saved features and labels found")
                print("skipping feature extraction and loading existing data...")
                return features, labels
            print("saved files are not compatible, extracting features again...")
        except (OSError, ValueError):
            print("saved files could not be loaded, extracting features again...")

    print("loading data...")
    print("extracting 18 features...")
    features, labels = extract_dataset_features(dataset_path)

    np.save(features_file, features)
    np.save(labels_file, labels)
    print("features and labels saved")
    return features, labels
