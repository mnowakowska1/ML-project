import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from feature_extraction import classes, feature_labels, load_or_extract_features

# paths (change dataset_path if the images are stored somewhere else)
project_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(project_dir, "PBC_dataset_normal_DIB")
features_file = os.path.join(project_dir, "pbc_features_18.npy")
labels_file = os.path.join(project_dir, "pbc_labels.npy")
results_dir = os.path.join(project_dir, "results")
os.makedirs(results_dir, exist_ok=True)


def save_plot(file_name):
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, file_name), dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title, file_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=ax)
    plt.title(title)
    save_plot(file_name)


def print_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"accuracy: {accuracy:.4f}")
    print(f"balanced accuracy: {balanced_accuracy:.4f}")
    print(f"macro-f1: {macro_f1:.4f}")
    return accuracy, balanced_accuracy, macro_f1


# loading data and extracting features
x_features, y_data = load_or_extract_features(dataset_path, features_file, labels_file)

# analysing the dataset
data = pd.DataFrame(x_features, columns=feature_labels)
print("\nnumber of images:", data.shape[0])
print("number of features:", data.shape[1])
print("\nfirst five rows:")
print(data.head())

class_counts = pd.Series(y_data).value_counts().sort_index()
class_counts.index = classes
print("\nclass distribution:")
print(class_counts)

plt.figure(figsize=(9, 5))
plt.bar(class_counts.index, class_counts.values, color='steelblue')
plt.xlabel('cell type')
plt.ylabel('number of images')
plt.title('class distribution')
plt.xticks(rotation=45)
save_plot("class_distribution.png")

# correlation matrix (technical_flag is constant, so its row is empty)
correlation = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation, cmap='coolwarm', center=0, annot=True, fmt='.2f', annot_kws={'size': 7})
plt.title('feature correlation matrix')
save_plot("correlation_matrix.png")

print("\nhighly correlated feature pairs (|r| > 0.9):")
for i in range(len(feature_labels)):
    for j in range(i + 1, len(feature_labels)):
        r = correlation.iloc[i, j]
        if abs(r) > 0.9:
            print(f"{feature_labels[i]} - {feature_labels[j]}: {r:.3f}")

# splitting the dataset into test and train data (all 18 features)
X_train_18, X_test_18, y_train_18, y_test_18 = train_test_split(
    x_features, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# initial random forest model
print("\ntraining random forest (18 features)...")
rf_initial = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_initial.fit(X_train_18, y_train_18)
y_pred_rf_initial = rf_initial.predict(X_test_18)
print_scores(y_test_18, y_pred_rf_initial)
print(classification_report(y_test_18, y_pred_rf_initial, target_names=classes))
plot_confusion_matrix(y_test_18, y_pred_rf_initial,
                      'confusion matrix - initial random forest (18 features)', "cm_initial_random_forest.png")

# feature importance ranking
importance = pd.Series(rf_initial.feature_importances_, index=feature_labels).sort_values(ascending=False)
print("\nfeature importance:")
print(importance)

plt.figure(figsize=(9, 6))
plt.barh(importance.index, importance.values, color='steelblue')
plt.gca().invert_yaxis()
plt.xlabel('importance')
plt.title('random forest feature importance')
save_plot("feature_importance.png")

# feature selection (18 -> 13 features)
drop_cols = [
    'technical_flag', 'nucleus_diameter', 'cell_diameter',
    'glcm_energy', 'glcm_homogeneity'
]

data_selected = pd.DataFrame(x_features, columns=feature_labels).drop(columns=drop_cols)
print("\nremoved features:", drop_cols)
print("selected features:", list(data_selected.columns))

# initial svm model (original version: scaling before splitting)
# scaling before splitting causes data leakage
print("\ntraining initial svm (13 features, scaled before splitting)...")
scaler = StandardScaler()
x_scaled = scaler.fit_transform(data_selected.values)

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    x_scaled, y_data, test_size=0.2, random_state=42, stratify=y_data
)

svm_initial = SVC(kernel='rbf', random_state=42)
svm_initial.fit(X_train_scaled, y_train_scaled)
y_pred_svm_initial = svm_initial.predict(X_test_scaled)
print_scores(y_test_scaled, y_pred_svm_initial)
print(classification_report(y_test_scaled, y_pred_svm_initial, target_names=classes))
plot_confusion_matrix(y_test_scaled, y_pred_svm_initial,
                      'confusion matrix - initial svm (scaled before splitting)', "cm_initial_svm.png")

# splitting before scaling (no data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    data_selected.values, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# corrected initial svm (scaler fitted only on the training data)
print("\ntraining initial svm (corrected version, scaler inside pipeline)...")
svm_initial_corrected = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='rbf', random_state=42))
])
svm_initial_corrected.fit(X_train, y_train)
y_pred_svm_corrected = svm_initial_corrected.predict(X_test)
print_scores(y_test, y_pred_svm_corrected)

# comparing baseline models (5-fold stratified cv)
models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "SVM (RBF)": SVC(kernel='rbf', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("\ncomparing baseline models (5-fold cv)...")
for name, clf in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', clf)
    ])

    start = time.time()
    scores = cross_validate(
        pipe, X_train, y_train,
        cv=cv,
        scoring=['balanced_accuracy', 'f1_macro'],
        n_jobs=-1
    )
    elapsed = time.time() - start

    results.append({
        'model': name,
        'cv balanced accuracy': scores['test_balanced_accuracy'].mean(),
        'cv balanced accuracy std': scores['test_balanced_accuracy'].std(),
        'cv macro-f1': scores['test_f1_macro'].mean(),
        'cv macro-f1 std': scores['test_f1_macro'].std(),
        'time (s)': round(elapsed, 2)
    })
    print(f"{name:22s} | balanced accuracy: {scores['test_balanced_accuracy'].mean():.4f} "
          f"± {scores['test_balanced_accuracy'].std():.4f} | "
          f"macro-f1: {scores['test_f1_macro'].mean():.4f} | {elapsed:.1f}s")

results_df = pd.DataFrame(results).sort_values(by='cv macro-f1', ascending=False)
print("\nbaseline models:")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(results_dir, "baseline_results.csv"), index=False)

plt.figure(figsize=(10, 5))
plt.barh(results_df['model'], results_df['cv macro-f1'], xerr=results_df['cv macro-f1 std'],
         color='steelblue', capsize=4)
plt.gca().invert_yaxis()
plt.xlim(0.5, 1.0)
plt.xlabel('cv macro-f1')
plt.title('baseline models (5-fold stratified cv, training set)')
save_plot("baseline_comparison.png")

# tuning hyperparameters for selected models
param_grids = {
    "SVM (Tuned)": (
        SVC(random_state=42),
        {
            'model__C': [1, 10, 50],
            'model__gamma': ['scale', 0.05, 0.1],
            'model__class_weight': [None, 'balanced']
        }
    ),
    "HistGradientBoosting (Tuned)": (
        HistGradientBoostingClassifier(random_state=42),
        {
            'model__learning_rate': [0.05, 0.1],
            'model__max_iter': [200, 400],
            'model__max_depth': [6, 12, None],
            'model__l2_regularization': [0.0, 1.0]
        }
    ),
    "Random Forest (Tuned)": (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            'model__n_estimators': [200, 400],
            'model__max_depth': [15, 25, None],
            'model__min_samples_leaf': [1, 2],
            'model__class_weight': [None, 'balanced']
        }
    )
}

tuned_results = []
best_estimators = {}

print("\ntuning hyperparameters (gridsearchcv)...")
for name, (clf, grid) in param_grids.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', clf)
    ])

    search = GridSearchCV(
        pipe, param_grid=grid,
        cv=cv, scoring='f1_macro',
        n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_estimators[name] = best_model

    y_pred = best_model.predict(X_test)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    tuned_results.append({
        'model': name,
        'best cv macro-f1': search.best_score_,
        'holdout balanced accuracy': balanced_accuracy,
        'holdout macro-f1': macro_f1,
        'best params': str(search.best_params_)
    })
    print(f"\n{name}:")
    print("best parameters:", search.best_params_)
    print(f"best cv macro-f1: {search.best_score_:.4f}")
    print(f"holdout balanced accuracy: {balanced_accuracy:.4f} | holdout macro-f1: {macro_f1:.4f}")

# svm fine-tuning (expanding the grid around C=50 and gamma=0.05)
print("\nfine-tuning svm (expanded grid)...")
svm_fine_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='rbf', random_state=42))
])

svm_fine_grid = {
    'model__C': [50, 100, 200, 500],
    'model__gamma': [0.01, 0.02, 0.03, 0.05]
}

svm_fine_search = GridSearchCV(
    svm_fine_pipe, param_grid=svm_fine_grid,
    cv=cv, scoring='f1_macro',
    n_jobs=-1, verbose=1
)
svm_fine_search.fit(X_train, y_train)

best_svm_fine = svm_fine_search.best_estimator_
best_estimators["SVM (Fine-Tuned)"] = best_svm_fine

y_pred_fine = best_svm_fine.predict(X_test)
balanced_accuracy_fine = balanced_accuracy_score(y_test, y_pred_fine)
macro_f1_fine = f1_score(y_test, y_pred_fine, average='macro')

tuned_results.append({
    'model': 'SVM (Fine-Tuned)',
    'best cv macro-f1': svm_fine_search.best_score_,
    'holdout balanced accuracy': balanced_accuracy_fine,
    'holdout macro-f1': macro_f1_fine,
    'best params': str(svm_fine_search.best_params_)
})

print("\nSVM (Fine-Tuned):")
print("best parameters:", svm_fine_search.best_params_)
print(f"best cv macro-f1: {svm_fine_search.best_score_:.4f}")
print(f"holdout balanced accuracy: {balanced_accuracy_fine:.4f} | holdout macro-f1: {macro_f1_fine:.4f}")

# checking if ensemble models improve the results
print("\ntraining ensemble models (voting and stacking)...")

# hard voting (equal weights 1:1:1)
voting_clf = VotingClassifier(
    estimators=[
        ('svm', best_svm_fine),
        ('hgb', best_estimators["HistGradientBoosting (Tuned)"]),
        ('rf', best_estimators["Random Forest (Tuned)"])
    ],
    voting='hard',
    n_jobs=-1
)
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)

balanced_accuracy_voting = balanced_accuracy_score(y_test, y_pred_voting)
macro_f1_voting = f1_score(y_test, y_pred_voting, average='macro')

tuned_results.append({
    'model': 'Voting Ensemble (SVM+HGB+RF)',
    'best cv macro-f1': np.nan,
    'holdout balanced accuracy': balanced_accuracy_voting,
    'holdout macro-f1': macro_f1_voting,
    'best params': 'hard voting'
})
best_estimators['Voting Ensemble (SVM+HGB+RF)'] = voting_clf

# stacking (logistic regression learns how to combine the models)
meta_model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

stacking_clf = StackingClassifier(
    estimators=[
        ('svm', best_svm_fine),
        ('hgb', best_estimators["HistGradientBoosting (Tuned)"]),
        ('rf', best_estimators["Random Forest (Tuned)"])
    ],
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)
stacking_clf.fit(X_train, y_train)
y_pred_stack = stacking_clf.predict(X_test)

balanced_accuracy_stack = balanced_accuracy_score(y_test, y_pred_stack)
macro_f1_stack = f1_score(y_test, y_pred_stack, average='macro')

tuned_results.append({
    'model': 'Stacking Ensemble (SVM+HGB+RF)',
    'best cv macro-f1': np.nan,
    'holdout balanced accuracy': balanced_accuracy_stack,
    'holdout macro-f1': macro_f1_stack,
    'best params': 'meta: scaled logistic regression'
})
best_estimators['Stacking Ensemble (SVM+HGB+RF)'] = stacking_clf

# final comparison of tuned models and ensembles
tuned_df = pd.DataFrame(tuned_results).sort_values(by='holdout macro-f1', ascending=False)
print("\ntuned models and ensembles:")
print(tuned_df[['model', 'best cv macro-f1', 'holdout balanced accuracy', 'holdout macro-f1']].to_string(index=False))
tuned_df.to_csv(os.path.join(results_dir, "tuned_results.csv"), index=False)

plt.figure(figsize=(10, 5))
plt.barh(tuned_df['model'], tuned_df['holdout macro-f1'], color='darkorange')
plt.gca().invert_yaxis()
plt.xlim(0.85, 0.96)
plt.xlabel('holdout macro-f1')
plt.title('tuned models and ensembles (holdout set)')
save_plot("tuned_and_ensemble_comparison.png")

# final evaluation of the best model
# the best model is chosen using the holdout set, so its scores are not fully independent
winner_name = tuned_df.iloc[0]['model']
winner_model = best_estimators[winner_name]
y_pred_winner = winner_model.predict(X_test)

print(f"\nbest model: {winner_name}")
print_scores(y_test, y_pred_winner)
print(f"\nclassification report: {winner_name}")
print(classification_report(y_test, y_pred_winner, target_names=classes))

plot_confusion_matrix(y_test, y_pred_winner, f'confusion matrix - {winner_name}', "cm_final_model.png")

# class-level performance
report = classification_report(y_test, y_pred_winner, target_names=classes, output_dict=True)
class_scores = pd.DataFrame(report).T.loc[classes, ['precision', 'recall', 'f1-score']]

class_scores.plot(kind='bar', figsize=(10, 5), ylim=(0.5, 1.0))
plt.xlabel('cell type')
plt.ylabel('score')
plt.title(f'class-level performance - {winner_name}')
plt.xticks(rotation=45)
plt.legend(loc='lower right')
save_plot("class_performance.png")
