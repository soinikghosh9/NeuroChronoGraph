"""
Manuscript Tables Generation Module.

This module provides functions for generating publication-ready
tables for the research manuscript.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


def create_demographics_table(metadata: Dict[str, List]) -> pd.DataFrame:
    """
    Create demographics table (Table 1).
    
    Args:
        metadata: Dictionary with keys: group, age, sex, mmse
        
    Returns:
        Formatted DataFrame
    """
    groups = ['AD', 'FTD', 'CN']
    
    data = []
    for group in groups:
        mask = np.array(metadata['group']) == group
        
        n = np.sum(mask)
        ages = np.array(metadata['age'])[mask]
        sexes = np.array(metadata['sex'])[mask]
        mmse = np.array(metadata['mmse'])[mask]
        
        # Sex distribution
        n_male = np.sum(sexes == 1)
        n_female = n - n_male
        
        row = {
            'Group': group,
            'N': n,
            'Age (years)': f"{np.mean(ages):.1f} ± {np.std(ages):.1f}",
            'Age Range': f"{np.min(ages):.0f} - {np.max(ages):.0f}",
            'Sex (M/F)': f"{n_male}/{n_female}",
            'MMSE': f"{np.mean(mmse):.1f} ± {np.std(mmse):.1f}",
            'MMSE Range': f"{np.min(mmse):.0f} - {np.max(mmse):.0f}"
        }
        data.append(row)
    
    # Add total row
    n_total = len(metadata['age'])
    ages_all = np.array(metadata['age'])
    sexes_all = np.array(metadata['sex'])
    mmse_all = np.array(metadata['mmse'])
    
    total_row = {
        'Group': 'Total',
        'N': n_total,
        'Age (years)': f"{np.mean(ages_all):.1f} ± {np.std(ages_all):.1f}",
        'Age Range': f"{np.min(ages_all):.0f} - {np.max(ages_all):.0f}",
        'Sex (M/F)': f"{np.sum(sexes_all == 1)}/{n_total - np.sum(sexes_all == 1)}",
        'MMSE': f"{np.nanmean(mmse_all):.1f} ± {np.nanstd(mmse_all):.1f}",
        'MMSE Range': f"{np.nanmin(mmse_all):.0f} - {np.nanmax(mmse_all):.0f}"
    }
    data.append(total_row)
    
    df = pd.DataFrame(data)
    return df


def create_classification_results_table(results: Dict) -> pd.DataFrame:
    """
    Create classification results table (Table 2).
    
    Args:
        results: Classification results dictionary
        
    Returns:
        Formatted DataFrame
    """
    class_names = ['AD', 'FTD', 'CN']
    
    # Overall metrics
    overall_data = {
        'Metric': ['Accuracy', 'Macro F1', 'Weighted F1', 'MCC', 'AUC (macro)'],
        'Value': [
            f"{results.get('accuracy', 0):.4f}",
            f"{results.get('f1_macro', 0):.4f}",
            f"{results.get('f1_weighted', 0):.4f}",
            f"{results.get('mcc', 0):.4f}",
            f"{results.get('auc_macro', 0):.4f}" if results.get('auc_macro') else 'N/A'
        ],
        '95% CI': [
            f"[{results.get('accuracy_ci_lower', 0):.3f}, {results.get('accuracy_ci_upper', 0):.3f}]",
            f"[{results.get('f1_ci_lower', 0):.3f}, {results.get('f1_ci_upper', 0):.3f}]",
            '-',
            f"[{results.get('mcc_ci_lower', 0):.3f}, {results.get('mcc_ci_upper', 0):.3f}]",
            '-'
        ],
        'p-value': [
            f"{results.get('accuracy_p', 0):.4f}",
            f"{results.get('f1_p', 0):.4f}",
            '-',
            f"{results.get('mcc_p', 0):.4f}",
            '-'
        ]
    }
    
    df_overall = pd.DataFrame(overall_data)
    return df_overall


def create_per_class_metrics_table(results: Dict) -> pd.DataFrame:
    """
    Create per-class metrics table (Table 3).
    
    Args:
        results: Classification results dictionary
        
    Returns:
        Formatted DataFrame
    """
    class_names = ['AD', 'FTD', 'CN']
    
    data = []
    for i, class_name in enumerate(class_names):
        f1 = results['f1_per_class'][i] if 'f1_per_class' in results else 0
        sens = results.get(f'{class_name}_sensitivity', 0)
        spec = results.get(f'{class_name}_specificity', 0)
        prec = results['precision_per_class'][i] if 'precision_per_class' in results else 0
        
        row = {
            'Class': class_name,
            'Sensitivity': f"{sens:.4f}",
            'Specificity': f"{spec:.4f}",
            'Precision': f"{prec:.4f}",
            'F1-Score': f"{f1:.4f}",
            'Support': results.get(f'{class_name}_support', 'N/A')
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def create_feature_importance_table(importance: Dict[str, Dict[str, float]],
                                     top_n: int = 10) -> pd.DataFrame:
    """
    Create feature importance table (Table 4).
    
    Args:
        importance: {class: {feature: importance}} nested dict
        top_n: Number of top features to include
        
    Returns:
        Formatted DataFrame
    """
    data = []
    
    for class_name, class_imp in importance.items():
        sorted_features = sorted(class_imp.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (feature, imp) in enumerate(sorted_features[:top_n], 1):
            data.append({
                'Class': class_name,
                'Rank': rank,
                'Region/Feature': feature,
                'Importance': f"{imp:.4f}"
            })
    
    df = pd.DataFrame(data)
    return df


def create_statistical_comparison_table(comparisons: Dict) -> pd.DataFrame:
    """
    Create statistical comparison table (Table 5).
    
    Args:
        comparisons: Dictionary of statistical comparisons
        
    Returns:
        Formatted DataFrame
    """
    data = []
    
    for comparison, results in comparisons.items():
        row = {
            'Comparison': comparison,
            'Test Statistic': f"{results.get('statistic', 0):.3f}",
            'p-value': f"{results.get('p_value', 0):.4f}",
            'Effect Size': f"{results.get('effect_size', 0):.3f}",
            'Interpretation': results.get('interpretation', '-'),
            'Significant': '***' if results.get('p_value', 1) < 0.001 else
                          '**' if results.get('p_value', 1) < 0.01 else
                          '*' if results.get('p_value', 1) < 0.05 else 'n.s.'
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def create_model_comparison_table(models: Dict) -> pd.DataFrame:
    """
    Create model comparison table (Table 6).
    
    Args:
        models: Dictionary of model results
        
    Returns:
        Formatted DataFrame
    """
    data = []
    
    for model_name, results in models.items():
        row = {
            'Model': model_name,
            'Accuracy': f"{results.get('accuracy', 0):.4f}",
            'Macro F1': f"{results.get('f1_macro', 0):.4f}",
            'MCC': f"{results.get('mcc', 0):.4f}",
            'Training Time': f"{results.get('training_time', 0):.1f}s",
            'Parameters': f"{results.get('n_params', 0):,}"
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def create_confusion_matrix_table(cm: np.ndarray,
                                   class_names: List[str] = None) -> pd.DataFrame:
    """
    Create confusion matrix as table.
    
    Args:
        cm: Confusion matrix array
        class_names: Class names
        
    Returns:
        Formatted DataFrame
    """
    if class_names is None:
        class_names = ['AD', 'FTD', 'CN']
    
    # Create DataFrame
    df = pd.DataFrame(cm, 
                     index=[f'True {c}' for c in class_names],
                     columns=[f'Pred {c}' for c in class_names])
    
    return df


def create_graph_metrics_table(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create graph metrics comparison table.
    
    Args:
        metrics: {group: {metric: value}} nested dict
        
    Returns:
        Formatted DataFrame
    """
    # Get all metric names
    all_metrics = set()
    for group_metrics in metrics.values():
        all_metrics.update(group_metrics.keys())
    
    data = []
    for metric in sorted(all_metrics):
        row = {'Metric': metric.replace('_', ' ').title()}
        for group in ['AD', 'FTD', 'CN']:
            if group in metrics and metric in metrics[group]:
                row[group] = f"{metrics[group][metric]:.4f}"
            else:
                row[group] = '-'
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def create_spectral_features_table(features: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create spectral features comparison table.
    
    Args:
        features: {group: {band: power}} nested dict
        
    Returns:
        Formatted DataFrame
    """
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    
    data = []
    for band in bands:
        row = {'Band': band}
        for group in ['AD', 'FTD', 'CN']:
            if group in features:
                val = features[group].get(band.lower(), 0)
                row[f'{group} (μV²)'] = f"{val:.4f}"
        data.append(row)
    
    df = pd.DataFrame(data)
    return df


def save_table_as_latex(df: pd.DataFrame, 
                        filepath: Path,
                        caption: str = '',
                        label: str = '') -> str:
    """
    Save table as LaTeX format.
    
    Args:
        df: DataFrame to save
        filepath: Output path
        caption: Table caption
        label: Table label for referencing
        
    Returns:
        LaTeX string
    """
    latex = df.to_latex(index=False, escape=False)
    
    # Add caption and label if provided
    if caption or label:
        latex_lines = latex.split('\n')
        insert_idx = 1  # After \begin{tabular}
        
        if caption:
            latex_lines.insert(insert_idx, f'\\caption{{{caption}}}')
            insert_idx += 1
        if label:
            latex_lines.insert(insert_idx, f'\\label{{{label}}}')
        
        latex = '\n'.join(latex_lines)
    
    # Save to file
    with open(filepath, 'w') as f:
        f.write(latex)
    
    return latex


def save_table_as_csv(df: pd.DataFrame, filepath: Path):
    """Save table as CSV."""
    df.to_csv(filepath, index=False)


def save_table_as_excel(tables: Dict[str, pd.DataFrame], filepath: Path):
    """Save multiple tables to Excel workbook."""
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)  # Excel sheet name limit


def generate_all_tables(results: Dict,
                        output_dir: Path,
                        format: str = 'all') -> Dict[str, pd.DataFrame]:
    """
    Generate all manuscript tables.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        format: 'csv', 'latex', 'excel', or 'all'
        
    Returns:
        Dictionary of generated tables
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tables = {}
    
    # Table 2: Classification Results
    if 'accuracy' in results:
        tables['classification_results'] = create_classification_results_table(results)
    
    # Table 3: Per-Class Metrics
    if 'f1_per_class' in results:
        tables['per_class_metrics'] = create_per_class_metrics_table(results)
    
    # Table 4: Confusion Matrix
    if 'confusion_matrix' in results:
        cm = np.array(results['confusion_matrix'])
        tables['confusion_matrix'] = create_confusion_matrix_table(cm)
    
    # Save in requested format
    if format in ['csv', 'all']:
        for name, df in tables.items():
            save_table_as_csv(df, output_dir / f'{name}.csv')
    
    if format in ['latex', 'all']:
        for name, df in tables.items():
            save_table_as_latex(df, output_dir / f'{name}.tex',
                               caption=name.replace('_', ' ').title())
    
    if format in ['excel', 'all']:
        save_table_as_excel(tables, output_dir / 'manuscript_tables.xlsx')
    
    print(f"Generated {len(tables)} tables in {output_dir}")
    
    return tables
