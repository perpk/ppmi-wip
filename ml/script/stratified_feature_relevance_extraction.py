from pathlib import Path
from typing import Final

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import scipy.sparse

PATH: Final = "/Users/kpax/Documents/aep/study/MSC/lab/PPMI_Project_133_RNASeq/classification/"


def fast_svm_importance(pipeline, X, y, n_repeats=5, n_jobs=-1):
    """
    Optimized permutation importance for SVM
    - Uses parallel processing
    - Sub-samples data for speed
    """
    # 1. Subsample the data
    n_samples = min(500, X.shape[0])
    sample_idx = np.random.choice(X.shape[0], n_samples, replace=False)

    # 2. Convert to numpy upfront to avoid indexing issues
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X_sample = X.iloc[sample_idx].values  # Convert DataFrame to numpy
    else:
        X_sample = X[sample_idx]  # Works for numpy arrays and sparse

    if isinstance(y, (pd.DataFrame, pd.Series)):
        y_sample = y.iloc[sample_idx].values
    else:
        y_sample = y[sample_idx]

    # 3. Ensure we have a numpy array (handle sparse matrices)
    if scipy.sparse.issparse(X_sample):
        X_sample = X_sample.toarray()

    # 4. Base score calculation
    def _calc_score(X_subset, y_subset):
        return pipeline.score(X_subset, y_subset)

    base_score = _calc_score(X_sample, y_sample)

    # 5. Parallel permutation with proper array handling
    def _permute_feature(feat_idx):
        rng = np.random.RandomState(42)
        perm_scores = []
        for _ in range(n_repeats):
            X_permuted = X_sample.copy()
            rng.shuffle(X_permuted[:, feat_idx])  # In-place permutation
            perm_scores.append(_calc_score(X_permuted, y_sample))
        return np.mean(perm_scores)

    # 6. Run in parallel (now safe with pure numpy)
    perm_results = Parallel(n_jobs=n_jobs)(
        delayed(_permute_feature)(i) for i in range(X_sample.shape[1])
    )

    return base_score - np.array(perm_results)


def get_svm_feature_importance(model_path, X_test, y_test, n_repeats=5, n_jobs=-1):
    """
       Robust feature importance for SVM models (works for all kernel types)
       Returns DataFrame with features and their importance scores

       Parameters:
       - model_path: path to saved joblib model
       - X_test: test features (DataFrame or array)
       - y_test: test labels
       - n_repeats: permutation repetitions (default: 5)
       - n_jobs: parallel jobs (default: -1 for all cores)
       """
    # 1. Load model and get feature names
    model_dict = joblib.load(model_path)
    pipeline = model_dict['model']
    feature_names = model_dict.get('features',
                                   pipeline[:-1].get_feature_names_out())

    # 2. Handle linear SVM case (use coefficients directly)
    svm = pipeline.steps[-1][1]
    if svm.kernel == 'linear':
        if svm.coef_.shape[0] == 1:  # Binary classification
            coef = svm.coef_[0]
        else:  # Multiclass
            coef = np.mean(np.abs(svm.coef_), axis=0)  # Average across classes

        return pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(coef),  # Use absolute value for importance
            'coefficient': coef  # Keep original coefficients
        }).sort_values('importance', ascending=False)

    # 3. For non-linear SVM, use optimized permutation importance
    print(f"Using permutation importance for {svm.kernel} kernel SVM...")

    # Convert to numpy arrays for faster processing
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_test, (pd.DataFrame, pd.Series)):
        y_test = y_test.values

    # Subsample for faster computation (adjust based on your dataset size)
    n_samples = min(1000, X_test.shape[0])
    sample_idx = np.random.choice(X_test.shape[0], n_samples, replace=False)
    X_sample = X_test[sample_idx]
    y_sample = y_test[sample_idx]

    # Base score calculation
    base_score = pipeline.score(X_sample, y_sample)

    # Parallel feature permutation
    def _permute_feature(i):
        rng = np.random.RandomState(42)
        perm_scores = []
        for _ in range(n_repeats):
            X_permuted = X_sample.copy()
            rng.shuffle(X_permuted[:, i])  # In-place permutation
            perm_scores.append(pipeline.score(X_permuted, y_sample))
        return np.mean(perm_scores)

    # Run in parallel
    perm_results = Parallel(n_jobs=n_jobs)(
        delayed(_permute_feature)(i) for i in range(X_sample.shape[1]))

    # Calculate importance (how much score drops when feature is permuted)
    importance = base_score - np.array(perm_results)

    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

def extract_feature_importance(model_path, model_type='auto'):
    model_dict = joblib.load(model_path)
    pipeline = model_dict['model']

    estimator = pipeline.steps[-1][1]

    try:
        feature_names = model_dict['features']  # From your saved dictionary
    except:
        feature_names = pipeline[:-1].get_feature_names_out()

    if model_type == 'lr':
        if estimator.coef_.shape[0] == 1:
            importance = estimator.coef_[0]
        else:
            importance = estimator.coef_.mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'abs_importance': np.abs(importance)
        })

    elif model_type == 'svm':
        # importance_df = get_permutation_importance(model_path, model_dict['X_test'], model_dict['y_test'], n_repeats=3)
        importance_df = get_svm_feature_importance(model_path, model_dict['X_test'], model_dict['y_test'])
        importance_df['abs_importance'] = np.abs(importance_df['importance'])

    elif model_type in ['rf', 'xgb']:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': estimator.feature_importances_
        })
        importance_df['abs_importance'] = importance_df['importance']

    importance_df = importance_df.sort_values('abs_importance', ascending=False)

    return importance_df, model_type


from sklearn.inspection import permutation_importance
import numpy as np


def get_permutation_importance(model_path, X_test, y_test, n_repeats=10):
    """
    Calculate permutation importance for any model
    Returns: DataFrame with features and their importance scores
    """
    # Load model
    model_dict = joblib.load(model_path)
    pipeline = model_dict['model']

    # Get feature names
    try:
        feature_names = model_dict['features']
    except:
        feature_names = pipeline[:-1].get_feature_names_out()

    # Calculate permutation importance
    result = permutation_importance(
        pipeline, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42
    )

    # Create results DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    }).sort_values('importance_mean', ascending=False)

    return importance_df

def plot_feature_importance(importance_df, output, model_type, top_n=20):
    plt.figure(figsize=(12, 6))

    # Different plot styles based on model type
    if model_type in ['lr', 'svm']:
        # Show direction with color
        colors = ['red' if x < 0 else 'green' for x in importance_df['importance'].head(top_n)]
        importance_df.head(top_n).plot.bar(
            x='feature',
            y='importance',
            color=colors,
            title=f'Feature Importance ({model_type.upper()}) - Direction Matters',
            ax=plt.gca()
        )
    else:  # rf, xgboost
        importance_df.head(top_n).plot.bar(
            x='feature',
            y='importance',
            title=f'Feature Importance ({model_type.upper()})',
            ax=plt.gca()
        )

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

def main():
    visits = ['BL', 'V02', 'V04', 'V06', 'V08']
    age_groups = ['30-50', '50-70', '70-80', '>80']
    genders = ['Male', 'Female']
    ml_paths = ['SVM2'] #('RF2', 'LR2', 'XGBOOST2', 'SVM2')
    model_ids = ['svm'] #('rf', 'lr', 'xgb', 'svm')

    for gender in genders:
        for age_group in age_groups:
            for visit in visits:
                for model_id, ml_path in zip(model_ids, ml_paths):
                    print(f"Processing {gender}, {age_group}, {visit}, {model_id}")
                    model_path = f"{PATH}{ml_path}/model_{gender}_{age_group}_{visit}.joblib"
                    if not Path(model_path).exists():
                        print(f"Model not found: {model_path}")
                        continue
                    importance_df, _ = extract_feature_importance(model_path, model_id)
                    importance_df.to_csv(f"{PATH}{ml_path}/feature_importance_data_{gender}_{age_group}_{visit}.csv")
                    plot_feature_importance(importance_df, f"{PATH}{ml_path}/feature_importance_plot_{gender}_{age_group}_{visit}.png", model_id)

if __name__ == '__main__':
    main()