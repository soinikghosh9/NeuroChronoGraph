"""
Statistical Significance Testing.

This module provides functions for permutation testing and
statistical significance assessment.
"""

import numpy as np
from typing import Callable, Dict, Tuple, Optional
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
import warnings


def permutation_test(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     metric_func: Callable = None,
                     n_permutations: int = 1000,
                     random_state: int = 42) -> Dict:
    """
    Perform permutation test to assess if classification is better than chance.
    
    Args:
        y_true: True labels (n_samples,)
        y_pred: Predicted labels (n_samples,)
        metric_func: Metric function(y_true, y_pred) -> float.
                    Default: macro F1-score
        n_permutations: Number of permutations
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with:
        - observed_score: Actual metric value
        - p_value: Probability of observing this score by chance
        - null_distribution: List of scores under null hypothesis
        - null_mean: Mean of null distribution
        - null_std: Standard deviation of null distribution
    """
    np.random.seed(random_state)
    
    if metric_func is None:
        metric_func = lambda y_t, y_p: f1_score(y_t, y_p, average='macro')
    
    # Observed score
    observed_score = metric_func(y_true, y_pred)
    
    # Generate null distribution
    null_distribution = []
    
    for _ in range(n_permutations):
        # Shuffle true labels
        y_shuffled = np.random.permutation(y_true)
        null_score = metric_func(y_shuffled, y_pred)
        null_distribution.append(null_score)
    
    null_distribution = np.array(null_distribution)
    
    # P-value: proportion of null scores >= observed
    p_value = (np.sum(null_distribution >= observed_score) + 1) / (n_permutations + 1)
    
    return {
        'observed_score': float(observed_score),
        'p_value': float(p_value),
        'null_distribution': null_distribution.tolist(),
        'null_mean': float(np.mean(null_distribution)),
        'null_std': float(np.std(null_distribution)),
        'significant': p_value < 0.05,
        'highly_significant': p_value < 0.001
    }


def bootstrap_confidence_interval(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   metric_func: Callable = None,
                                   n_bootstrap: int = 1000,
                                   confidence: float = 0.95,
                                   random_state: int = 42) -> Dict:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_func: Metric function
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed
        
    Returns:
        Dictionary with point estimate and confidence interval
    """
    np.random.seed(random_state)
    
    if metric_func is None:
        metric_func = lambda y_t, y_p: f1_score(y_t, y_p, average='macro')
    
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_true = y_true[indices]
        boot_pred = y_pred[indices]
        
        try:
            score = metric_func(boot_true, boot_pred)
            bootstrap_scores.append(score)
        except:
            continue
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, alpha / 2 * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)
    
    return {
        'point_estimate': float(metric_func(y_true, y_pred)),
        'ci_lower': float(lower),
        'ci_upper': float(upper),
        'confidence_level': confidence,
        'bootstrap_mean': float(np.mean(bootstrap_scores)),
        'bootstrap_std': float(np.std(bootstrap_scores))
    }


def mcnemar_test(y_true: np.ndarray,
                 y_pred_1: np.ndarray,
                 y_pred_2: np.ndarray) -> Dict:
    """
    McNemar's test to compare two classifiers.
    
    Tests whether two classifiers have significantly different error rates.
    
    Args:
        y_true: True labels
        y_pred_1: Predictions from classifier 1
        y_pred_2: Predictions from classifier 2
        
    Returns:
        Dictionary with test statistic and p-value
    """
    from scipy import stats
    
    # Build contingency table
    correct_1 = (y_pred_1 == y_true)
    correct_2 = (y_pred_2 == y_true)
    
    # b: Model 1 correct, Model 2 wrong
    b = np.sum(correct_1 & ~correct_2)
    # c: Model 1 wrong, Model 2 correct
    c = np.sum(~correct_1 & correct_2)
    
    # McNemar's test statistic (with continuity correction)
    if b + c == 0:
        return {
            'statistic': 0,
            'p_value': 1.0,
            'significant': False
        }
    
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'b': int(b),
        'c': int(c),
        'significant': p_value < 0.05
    }


def compute_effect_size(confusion_matrix: np.ndarray) -> Dict:
    """
    Compute effect size metrics for multi-class classification.
    
    Args:
        confusion_matrix: Confusion matrix (n_classes, n_classes)
        
    Returns:
        Dictionary with Cramér's V and Cohen's Kappa
    """
    from sklearn.metrics import cohen_kappa_score
    
    n_samples = confusion_matrix.sum()
    n_classes = confusion_matrix.shape[0]
    
    # Chi-squared
    row_sums = confusion_matrix.sum(axis=1)
    col_sums = confusion_matrix.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n_samples
    
    chi2 = np.sum((confusion_matrix - expected) ** 2 / (expected + 1e-10))
    
    # Cramér's V
    min_dim = min(n_classes - 1, n_classes - 1)
    cramers_v = np.sqrt(chi2 / (n_samples * min_dim)) if min_dim > 0 else 0
    
    # Interpret effect size
    if cramers_v < 0.1:
        interpretation = "Negligible"
    elif cramers_v < 0.2:
        interpretation = "Small"
    elif cramers_v < 0.4:
        interpretation = "Medium"
    elif cramers_v < 0.6:
        interpretation = "Large"
    else:
        interpretation = "Very Large"
    
    return {
        'cramers_v': float(cramers_v),
        'interpretation': interpretation,
        'chi_squared': float(chi2),
        'n_samples': int(n_samples)
    }


def run_significance_analysis(y_true: np.ndarray,
                               y_pred: np.ndarray,
                               n_permutations: int = 1000) -> Dict:
    """
    Run complete significance analysis.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_permutations: Number of permutations for permutation test
        
    Returns:
        Comprehensive significance analysis results
    """
    from sklearn.metrics import confusion_matrix as compute_cm
    
    results = {}
    
    # Permutation tests for different metrics
    metrics = {
        'accuracy': lambda y_t, y_p: accuracy_score(y_t, y_p),
        'f1_macro': lambda y_t, y_p: f1_score(y_t, y_p, average='macro'),
        'mcc': lambda y_t, y_p: matthews_corrcoef(y_t, y_p)
    }
    
    for metric_name, metric_func in metrics.items():
        perm_result = permutation_test(y_true, y_pred, metric_func, n_permutations)
        ci_result = bootstrap_confidence_interval(y_true, y_pred, metric_func)
        
        results[metric_name] = {
            'value': perm_result['observed_score'],
            'p_value': perm_result['p_value'],
            'significant': perm_result['significant'],
            'ci_lower': ci_result['ci_lower'],
            'ci_upper': ci_result['ci_upper']
        }
    
    # Effect size
    cm = compute_cm(y_true, y_pred)
    effect_size = compute_effect_size(cm)
    results['effect_size'] = effect_size
    
    return results
