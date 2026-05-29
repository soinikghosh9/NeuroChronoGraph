"""
Feature Analysis Module.

This module provides comprehensive tools for analyzing feature importance,
selecting top-performing features, and understanding which biomarkers
best differentiate AD, FTD, and healthy controls.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import (
        mutual_info_classif, f_classif, SelectKBest, RFE
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available. Some analysis functions will be limited.")


class FeatureAnalyzer:
    """
    Comprehensive feature analysis for biomarker discovery.
    
    Provides multiple methods for ranking and selecting features:
    - Statistical tests (ANOVA, Kruskal-Wallis)
    - Mutual information
    - Random Forest importance
    - Recursive Feature Elimination
    - Effect size analysis
    """
    
    def __init__(self, n_top_features: int = 20):
        """
        Initialize FeatureAnalyzer.
        
        Args:
            n_top_features: Number of top features to report
        """
        self.n_top_features = n_top_features
        self.feature_rankings = {}
        self.feature_importance = {}
        self.scaler = StandardScaler() if HAS_SKLEARN else None
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: List[str] = None) -> 'FeatureAnalyzer':
        """
        Fit analyzer to data and compute all importance measures.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            feature_names: Names of features
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        self.feature_names = feature_names
        self.X = X
        self.y = y
        self.n_classes = len(np.unique(y))
        
        # Compute all importance measures
        self._compute_anova_importance(X, y)
        self._compute_mutual_information(X, y)
        self._compute_effect_sizes(X, y)
        
        if HAS_SKLEARN:
            self._compute_rf_importance(X, y)
        
        # Combine rankings
        self._compute_combined_ranking()
        
        return self
    
    def _compute_anova_importance(self, X: np.ndarray, y: np.ndarray):
        """Compute ANOVA F-statistics and p-values."""
        f_stats = []
        p_values = []
        
        for i in range(X.shape[1]):
            groups = [X[y == c, i] for c in np.unique(y)]
            
            # Remove NaN values
            groups = [g[~np.isnan(g)] for g in groups]
            
            if all(len(g) > 0 for g in groups):
                try:
                    f, p = stats.f_oneway(*groups)
                    f_stats.append(f if not np.isnan(f) else 0)
                    p_values.append(p if not np.isnan(p) else 1)
                except:
                    f_stats.append(0)
                    p_values.append(1)
            else:
                f_stats.append(0)
                p_values.append(1)
        
        self.feature_importance['anova_f'] = np.array(f_stats)
        self.feature_importance['anova_p'] = np.array(p_values)
        
        # Ranking
        self.feature_rankings['anova'] = np.argsort(f_stats)[::-1]
    
    def _compute_mutual_information(self, X: np.ndarray, y: np.ndarray):
        """Compute mutual information between features and target."""
        if HAS_SKLEARN:
            # Handle NaN
            X_clean = np.nan_to_num(X, nan=0)
            mi = mutual_info_classif(X_clean, y, random_state=42)
            self.feature_importance['mutual_info'] = mi
            self.feature_rankings['mutual_info'] = np.argsort(mi)[::-1]
        else:
            self.feature_importance['mutual_info'] = np.zeros(X.shape[1])
            self.feature_rankings['mutual_info'] = np.arange(X.shape[1])
    
    def _compute_effect_sizes(self, X: np.ndarray, y: np.ndarray):
        """Compute effect sizes (eta-squared) for each feature."""
        eta_squared = []
        
        classes = np.unique(y)
        n_total = len(y)
        grand_mean = np.nanmean(X, axis=0)
        
        for i in range(X.shape[1]):
            # Between-group variance
            ss_between = 0
            ss_total = 0
            
            for c in classes:
                group = X[y == c, i]
                group = group[~np.isnan(group)]
                if len(group) > 0:
                    group_mean = np.mean(group)
                    ss_between += len(group) * (group_mean - grand_mean[i]) ** 2
            
            # Total variance
            values = X[:, i]
            values = values[~np.isnan(values)]
            if len(values) > 0:
                ss_total = np.sum((values - grand_mean[i]) ** 2)
            
            # Eta-squared
            eta = ss_between / ss_total if ss_total > 0 else 0
            eta_squared.append(eta)
        
        self.feature_importance['eta_squared'] = np.array(eta_squared)
        self.feature_rankings['effect_size'] = np.argsort(eta_squared)[::-1]
    
    def _compute_rf_importance(self, X: np.ndarray, y: np.ndarray):
        """Compute Random Forest feature importance."""
        X_clean = np.nan_to_num(X, nan=0)
        
        try:
            rf = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_clean, y)
            
            self.feature_importance['rf_importance'] = rf.feature_importances_
            self.feature_rankings['rf'] = np.argsort(rf.feature_importances_)[::-1]
        except Exception as e:
            warnings.warn(f"RF importance failed: {e}")
            self.feature_importance['rf_importance'] = np.zeros(X.shape[1])
            self.feature_rankings['rf'] = np.arange(X.shape[1])
    
    def _compute_combined_ranking(self):
        """Compute combined ranking from all methods."""
        n_features = len(self.feature_names)
        
        # Compute rank for each method
        rank_scores = np.zeros(n_features)
        n_methods = 0
        
        for method in ['anova', 'mutual_info', 'effect_size', 'rf']:
            if method in self.feature_rankings:
                ranking = self.feature_rankings[method]
                # Convert to rank scores
                for rank, idx in enumerate(ranking):
                    rank_scores[idx] += (n_features - rank)
                n_methods += 1
        
        if n_methods > 0:
            rank_scores /= n_methods
        
        self.feature_importance['combined_score'] = rank_scores
        self.feature_rankings['combined'] = np.argsort(rank_scores)[::-1]
    
    def get_top_features(self, method: str = 'combined', 
                         n: int = None) -> pd.DataFrame:
        """
        Get top features by specified method.
        
        Args:
            method: Ranking method ('combined', 'anova', 'rf', 'mutual_info', 'effect_size')
            n: Number of features to return
            
        Returns:
            DataFrame with top features
        """
        if n is None:
            n = self.n_top_features
        
        ranking = self.feature_rankings.get(method, self.feature_rankings['combined'])
        top_indices = ranking[:n]
        
        data = []
        for rank, idx in enumerate(top_indices, 1):
            row = {
                'Rank': rank,
                'Feature': self.feature_names[idx],
                'ANOVA_F': self.feature_importance.get('anova_f', np.zeros(len(self.feature_names)))[idx],
                'P-Value': self.feature_importance.get('anova_p', np.ones(len(self.feature_names)))[idx],
                'Eta²': self.feature_importance.get('eta_squared', np.zeros(len(self.feature_names)))[idx],
                'MI': self.feature_importance.get('mutual_info', np.zeros(len(self.feature_names)))[idx],
                'RF_Importance': self.feature_importance.get('rf_importance', np.zeros(len(self.feature_names)))[idx],
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary of all features with importance scores."""
        data = []
        
        for idx, name in enumerate(self.feature_names):
            row = {
                'Feature': name,
                'ANOVA_F': self.feature_importance.get('anova_f', np.zeros(len(self.feature_names)))[idx],
                'P-Value': self.feature_importance.get('anova_p', np.ones(len(self.feature_names)))[idx],
                'Eta²': self.feature_importance.get('eta_squared', np.zeros(len(self.feature_names)))[idx],
                'MI': self.feature_importance.get('mutual_info', np.zeros(len(self.feature_names)))[idx],
                'RF_Importance': self.feature_importance.get('rf_importance', np.zeros(len(self.feature_names)))[idx],
                'Combined_Score': self.feature_importance.get('combined_score', np.zeros(len(self.feature_names)))[idx],
                'Significant': self.feature_importance.get('anova_p', np.ones(len(self.feature_names)))[idx] < 0.05,
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('Combined_Score', ascending=False)
        
        return df
    
    def get_class_discriminative_features(self, 
                                           class_idx: int,
                                           n: int = 10) -> pd.DataFrame:
        """
        Get features that best discriminate a specific class from others.
        
        Args:
            class_idx: Class index (0=AD, 1=FTD, 2=CN)
            n: Number of features to return
            
        Returns:
            DataFrame with class-discriminative features
        """
        classes = np.unique(self.y)
        target_class = classes[class_idx]
        
        # Binary comparison: target vs rest
        y_binary = (self.y == target_class).astype(int)
        
        discriminative = []
        
        for idx, name in enumerate(self.feature_names):
            target_vals = self.X[self.y == target_class, idx]
            other_vals = self.X[self.y != target_class, idx]
            
            # Remove NaN
            target_vals = target_vals[~np.isnan(target_vals)]
            other_vals = other_vals[~np.isnan(other_vals)]
            
            if len(target_vals) > 0 and len(other_vals) > 0:
                # Mann-Whitney U test
                try:
                    stat, p = stats.mannwhitneyu(target_vals, other_vals, alternative='two-sided')
                    effect = (np.mean(target_vals) - np.mean(other_vals)) / (np.std(other_vals) + 1e-10)
                except:
                    p = 1
                    effect = 0
            else:
                p = 1
                effect = 0
            
            discriminative.append({
                'Feature': name,
                'P-Value': p,
                'Effect_Size': effect,
                'Target_Mean': np.mean(target_vals) if len(target_vals) > 0 else 0,
                'Other_Mean': np.mean(other_vals) if len(other_vals) > 0 else 0,
            })
        
        df = pd.DataFrame(discriminative)
        df = df.sort_values('P-Value').head(n)
        
        return df
    
    def identify_biomarkers(self, 
                            p_threshold: float = 0.05,
                            effect_threshold: float = 0.14) -> Dict[str, List[str]]:
        """
        Identify significant biomarkers based on statistical criteria.
        
        Args:
            p_threshold: P-value threshold for significance
            effect_threshold: Eta-squared threshold (0.01=small, 0.06=medium, 0.14=large)
            
        Returns:
            Dictionary with biomarker categories
        """
        p_values = self.feature_importance.get('anova_p', np.ones(len(self.feature_names)))
        eta_sq = self.feature_importance.get('eta_squared', np.zeros(len(self.feature_names)))
        
        biomarkers = {
            'highly_significant': [],  # p < 0.001 and large effect
            'significant': [],          # p < 0.05 and medium effect
            'trending': [],              # p < 0.1
            'not_significant': []
        }
        
        for idx, name in enumerate(self.feature_names):
            p = p_values[idx]
            eta = eta_sq[idx]
            
            if p < 0.001 and eta >= effect_threshold:
                biomarkers['highly_significant'].append(name)
            elif p < p_threshold and eta >= 0.06:
                biomarkers['significant'].append(name)
            elif p < 0.1:
                biomarkers['trending'].append(name)
            else:
                biomarkers['not_significant'].append(name)
        
        return biomarkers
    
    def cross_validate_features(self, 
                                 feature_indices: List[int] = None,
                                 n_features: int = 10,
                                 cv: int = 5) -> Dict[str, float]:
        """
        Cross-validate classification using selected features.
        
        Args:
            feature_indices: Indices of features to use (None = top N)
            n_features: Number of top features if indices not provided
            cv: Number of CV folds
            
        Returns:
            Dictionary with CV scores
        """
        if not HAS_SKLEARN:
            return {'accuracy': 0, 'f1': 0}
        
        if feature_indices is None:
            feature_indices = self.feature_rankings['combined'][:n_features]
        
        X_selected = self.X[:, feature_indices]
        X_clean = np.nan_to_num(X_selected, nan=0)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        try:
            scores = cross_val_score(clf, X_clean, self.y, cv=cv, scoring='accuracy')
            f1_scores = cross_val_score(clf, X_clean, self.y, cv=cv, scoring='f1_macro')
            
            return {
                'accuracy_mean': float(np.mean(scores)),
                'accuracy_std': float(np.std(scores)),
                'f1_mean': float(np.mean(f1_scores)),
                'f1_std': float(np.std(f1_scores)),
                'n_features': len(feature_indices)
            }
        except Exception as e:
            warnings.warn(f"CV failed: {e}")
            return {'accuracy_mean': 0, 'f1_mean': 0}
    
    def generate_report(self, save_path: Path = None) -> str:
        """
        Generate comprehensive feature analysis report.
        
        Args:
            save_path: Optional path to save report
            
        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("FEATURE ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append(f"\nTotal Features: {len(self.feature_names)}")
        lines.append(f"Samples: {len(self.y)}")
        lines.append(f"Classes: {self.n_classes}")
        
        # Biomarkers
        biomarkers = self.identify_biomarkers()
        lines.append(f"\n--- BIOMARKER SUMMARY ---")
        lines.append(f"Highly Significant: {len(biomarkers['highly_significant'])}")
        lines.append(f"Significant: {len(biomarkers['significant'])}")
        lines.append(f"Trending: {len(biomarkers['trending'])}")
        
        # Top features
        lines.append(f"\n--- TOP 15 FEATURES ---")
        top_df = self.get_top_features(n=15)
        lines.append(top_df.to_string(index=False))
        
        # Cross-validation
        cv_results = self.cross_validate_features(n_features=10)
        lines.append(f"\n--- CROSS-VALIDATION (Top 10 Features) ---")
        lines.append(f"Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results.get('accuracy_std', 0):.3f}")
        lines.append(f"F1 Macro: {cv_results['f1_mean']:.3f} ± {cv_results.get('f1_std', 0):.3f}")
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report


def analyze_feature_importance(features: Dict[str, np.ndarray],
                                labels: np.ndarray,
                                n_top: int = 20) -> pd.DataFrame:
    """
    Quick analysis of feature importance.
    
    Args:
        features: Dictionary of feature name to values
        labels: Class labels
        n_top: Number of top features to return
        
    Returns:
        DataFrame with feature rankings
    """
    # Convert to matrix
    feature_names = list(features.keys())
    X = np.column_stack([features[name] for name in feature_names])
    
    analyzer = FeatureAnalyzer(n_top_features=n_top)
    analyzer.fit(X, labels, feature_names)
    
    return analyzer.get_top_features()


def compare_groups_features(features: Dict[str, np.ndarray],
                             labels: np.ndarray,
                             group_names: List[str] = None) -> pd.DataFrame:
    """
    Compare feature values across groups.
    
    Args:
        features: Dictionary of feature name to values
        labels: Class labels
        group_names: Names of groups
        
    Returns:
        DataFrame with group comparisons
    """
    if group_names is None:
        group_names = ['AD', 'FTD', 'CN']
    
    results = []
    classes = np.unique(labels)
    
    for feat_name, values in features.items():
        row = {'Feature': feat_name}
        
        # Mean per group
        for i, c in enumerate(classes):
            group_vals = values[labels == c]
            group_vals = group_vals[~np.isnan(group_vals)]
            row[f'{group_names[i]}_mean'] = np.mean(group_vals) if len(group_vals) > 0 else 0
            row[f'{group_names[i]}_std'] = np.std(group_vals) if len(group_vals) > 0 else 0
        
        # ANOVA
        groups = [values[labels == c] for c in classes]
        groups = [g[~np.isnan(g)] for g in groups]
        
        if all(len(g) > 0 for g in groups):
            try:
                f, p = stats.f_oneway(*groups)
                row['F_stat'] = f
                row['P_value'] = p
            except:
                row['F_stat'] = 0
                row['P_value'] = 1
        else:
            row['F_stat'] = 0
            row['P_value'] = 1
        
        results.append(row)
    
    df = pd.DataFrame(results)
    df = df.sort_values('P_value')
    
    return df
