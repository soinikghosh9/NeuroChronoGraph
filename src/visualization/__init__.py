"""Visualization module initialization - Publication Ready."""

# Import style configuration first
from .style_config import (
    # Color palettes
    PALETTE_PRIMARY,
    PALETTE_EXTENDED,
    PALETTE_BANDS,
    PALETTE_NEUTRAL,
    PALETTE_COLORBLIND,
    CLASS_COLORS,
    CLASS_NAMES,
    BAND_COLORS,
    BAND_NAMES,
    # Colormaps
    CMAP_CONNECTIVITY,
    CMAP_IMPORTANCE,
    CMAP_DIVERGING,
    # Style functions
    set_publication_style,
    set_nature_style,
    set_science_style,
    set_neurology_style,
    # Figure sizes
    FIGURE_SIZES,
    get_figure_size,
    # Helper functions
    add_panel_label,
    add_significance_annotation,
    format_axis,
    despine,
    create_colorbar
)

# EEG visualization
from .eeg_plots import (
    plot_eeg_segment,
    plot_psd_comparison,
    plot_topography,
    plot_multi_topography,
    plot_erp
)

# Connectivity visualization
from .connectivity_plots import (
    plot_connectivity_matrix,
    plot_multiband_connectivity,
    plot_group_connectivity_comparison,
    plot_brain_network,
    plot_graph_metrics_comparison,
    plot_circular_connectome
)

# Model performance visualization
from .performance_plots import (
    plot_confusion_matrix,
    plot_multiclass_roc,
    plot_precision_recall_curves,
    plot_metrics_summary,
    plot_cross_validation_results,
    plot_probability_calibration,
    create_metrics_table
)

# Explainability visualization
from .explainability_plots import (
    plot_node_importance,
    plot_importance_by_class,
    plot_importance_topography,
    plot_edge_importance,
    plot_attention_weights,
    plot_class_differential_importance,
    plot_feature_contributions,
    plot_brain_schematic
)

# Statistical visualization
from .statistical_plots import (
    plot_permutation_test,
    plot_bootstrap_ci,
    plot_effect_size_bar,
    plot_group_comparison_boxplot,
    plot_multi_feature_comparison,
    plot_significance_summary_table,
    plot_clinical_correlation
)

# Manuscript figures
from .manuscript_figures import (
    generate_figure_1_overview,
    generate_figure_2_spectral,
    generate_figure_3_connectivity,
    generate_figure_4_model_performance,
    generate_figure_5_explainability,
    generate_all_manuscript_figures
)

# Manuscript tables
from .manuscript_tables import (
    create_demographics_table,
    create_classification_results_table,
    create_per_class_metrics_table,
    create_feature_importance_table,
    create_statistical_comparison_table,
    create_model_comparison_table,
    create_confusion_matrix_table,
    create_graph_metrics_table,
    create_spectral_features_table,
    save_table_as_latex,
    save_table_as_csv,
    save_table_as_excel,
    generate_all_tables
)

# Convenience: Apply publication style on import
set_publication_style()

# Define public API
__all__ = [
    # Style
    'PALETTE_PRIMARY', 'PALETTE_BANDS', 'PALETTE_NEUTRAL',
    'CLASS_COLORS', 'CLASS_NAMES', 'BAND_COLORS', 'BAND_NAMES',
    'set_publication_style', 'set_nature_style', 'set_science_style',
    'FIGURE_SIZES', 'get_figure_size',
    'add_panel_label', 'add_significance_annotation', 'format_axis', 'despine',
    # EEG
    'plot_eeg_segment', 'plot_psd_comparison', 'plot_topography', 
    'plot_multi_topography', 'plot_erp',
    # Connectivity
    'plot_connectivity_matrix', 'plot_multiband_connectivity',
    'plot_group_connectivity_comparison', 'plot_brain_network',
    'plot_graph_metrics_comparison', 'plot_circular_connectome',
    # Performance
    'plot_confusion_matrix', 'plot_multiclass_roc', 'plot_precision_recall_curves',
    'plot_metrics_summary', 'plot_cross_validation_results', 
    'plot_probability_calibration', 'create_metrics_table',
    # Explainability
    'plot_node_importance', 'plot_importance_by_class', 'plot_edge_importance',
    'plot_brain_schematic', 'plot_feature_contributions',
    # Statistics
    'plot_permutation_test', 'plot_bootstrap_ci', 'plot_group_comparison_boxplot',
    'plot_significance_summary_table',
    # Manuscript
    'generate_all_manuscript_figures', 'generate_all_tables',
]
