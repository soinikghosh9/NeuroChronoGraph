"""
GNN Explainer for Brain Network Interpretation.

This module provides explainability tools for understanding which
brain regions and connections drive the classification decisions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import warnings

try:
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.data import Data
    HAS_EXPLAINER = True
except ImportError:
    HAS_EXPLAINER = False
    warnings.warn("PyTorch Geometric Explainer not available.")

from ..config.config import CHANNEL_NAMES, ROI_NAMES_8


class BrainNetworkExplainer:
    """
    Explainer for brain network classification models.
    
    Uses GNNExplainer to identify important nodes (brain regions)
    and edges (connections) for classification.
    """
    
    def __init__(self,
                 model: nn.Module,
                 node_names: List[str] = None,
                 device: str = 'cuda'):
        """
        Initialize the explainer.
        
        Args:
            model: Trained GNN model
            node_names: Names of graph nodes (channels or ROIs)
            device: Device for computation
        """
        self.model = model
        self.node_names = node_names or CHANNEL_NAMES
        self.device = device
        
        self.model.eval()
        
        if HAS_EXPLAINER:
            self.explainer = Explainer(
                model=model,
                algorithm=GNNExplainer(epochs=200),
                explanation_type='model',
                node_mask_type='object',
                edge_mask_type='object',
                model_config=dict(
                    mode='multiclass_classification',
                    task_level='graph',
                    return_type='probs'
                )
            )
        else:
            self.explainer = None
    
    def explain_prediction(self,
                           data: 'Data',
                           target_class: Optional[int] = None) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            data: PyG Data object with graph
            target_class: Class to explain. If None, uses predicted class.
            
        Returns:
            Dictionary with node and edge importance scores
        """
        if not HAS_EXPLAINER:
            return self._fallback_explanation(data)
        
        data = data.to(self.device)
        
        # Get prediction if target not specified
        if target_class is None:
            with torch.no_grad():
                out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
                target_class = out.argmax(dim=1).item()
        
        # Get explanation
        explanation = self.explainer(
            data.x, data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
            index=target_class
        )
        
        # Extract masks
        node_mask = explanation.node_mask.cpu().numpy()
        edge_mask = explanation.edge_mask.cpu().numpy()
        
        # Map to node names
        node_importance = {}
        for i, name in enumerate(self.node_names[:len(node_mask)]):
            node_importance[name] = float(node_mask[i].mean())
        
        return {
            'node_importance': node_importance,
            'edge_mask': edge_mask,
            'target_class': target_class,
            'node_mask_raw': node_mask,
            'edge_mask_raw': edge_mask
        }
    
    def _fallback_explanation(self, data: 'Data') -> Dict:
        """Fallback when GNNExplainer is not available."""
        # Use gradient-based importance as fallback
        
        data = data.to(self.device)
        data.x.requires_grad = True
        
        out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
        target_class = out.argmax(dim=1).item()
        
        # Compute gradients
        out[0, target_class].backward()
        
        # Node importance from gradients
        node_importance = data.x.grad.abs().mean(dim=1).cpu().numpy()
        
        result = {}
        for i, name in enumerate(self.node_names[:len(node_importance)]):
            result[name] = float(node_importance[i])
        
        return {
            'node_importance': result,
            'edge_mask': None,
            'target_class': target_class,
            'method': 'gradient'
        }
    
    def explain_dataset(self,
                        data_list: List['Data'],
                        labels: List[int]) -> Dict:
        """
        Explain predictions for multiple samples.
        
        Args:
            data_list: List of PyG Data objects
            labels: True labels for each sample
            
        Returns:
            Aggregated explanations by class
        """
        class_explanations = {0: [], 1: [], 2: []}  # AD, FTD, CN
        
        for data, label in zip(data_list, labels):
            try:
                exp = self.explain_prediction(data, target_class=label)
                class_explanations[label].append(exp['node_importance'])
            except Exception as e:
                warnings.warn(f"Explanation failed: {e}")
        
        # Aggregate per class
        aggregated = {}
        class_names = {0: 'AD', 1: 'FTD', 2: 'CN'}
        
        for class_idx, explanations in class_explanations.items():
            if len(explanations) == 0:
                continue
            
            class_name = class_names[class_idx]
            
            # Average importance across samples
            avg_importance = {}
            for node in self.node_names:
                values = [exp.get(node, 0) for exp in explanations]
                avg_importance[node] = np.mean(values)
            
            # Sort by importance
            sorted_nodes = sorted(avg_importance.items(), 
                                  key=lambda x: x[1], reverse=True)
            
            aggregated[class_name] = {
                'importance': dict(sorted_nodes),
                'top_5_nodes': [n for n, _ in sorted_nodes[:5]],
                'n_samples': len(explanations)
            }
        
        return aggregated
    
    def get_discriminative_regions(self,
                                    class_explanations: Dict) -> Dict:
        """
        Identify brain regions that discriminate between classes.
        
        Args:
            class_explanations: Output from explain_dataset
            
        Returns:
            Regions that are more important for each class
        """
        discriminative = {}
        
        class_names = list(class_explanations.keys())
        
        for class_name in class_names:
            if class_name not in class_explanations:
                continue
            
            importance = class_explanations[class_name]['importance']
            
            # Compare to other classes
            other_classes = [c for c in class_names if c != class_name]
            
            discriminating_nodes = {}
            for node, imp in importance.items():
                other_imps = []
                for other in other_classes:
                    if other in class_explanations:
                        other_imps.append(
                            class_explanations[other]['importance'].get(node, 0)
                        )
                
                if other_imps:
                    ratio = imp / (np.mean(other_imps) + 1e-6)
                    discriminating_nodes[node] = ratio
            
            # Sort by discrimination ratio
            sorted_nodes = sorted(discriminating_nodes.items(),
                                  key=lambda x: x[1], reverse=True)
            
            discriminative[class_name] = {
                'most_discriminative': [n for n, _ in sorted_nodes[:5]],
                'discrimination_ratios': dict(sorted_nodes)
            }
        
        return discriminative


def compute_attention_importance(model: nn.Module,
                                  data: 'Data') -> Dict:
    """
    Extract attention weights from GAT layers as importance scores.
    
    Args:
        model: Model with GAT layers
        data: Input graph data
        
    Returns:
        Dictionary with attention-based importance
    """
    model.eval()
    
    # Extract attention weights (implementation depends on model structure)
    # This is a placeholder for the actual implementation
    
    return {
        'method': 'attention',
        'note': 'Implement based on specific model structure'
    }


def map_importance_to_brain_regions(node_importance: Dict,
                                     level: str = 'sensor') -> Dict:
    """
    Map node importance to anatomical brain regions.
    
    Args:
        node_importance: Dictionary of node -> importance
        level: 'sensor' for 19-channel, 'source' for 8-ROI
        
    Returns:
        Importance mapped to anatomical regions
    """
    if level == 'sensor':
        # Map sensors to lobes
        lobe_mapping = {
            'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
            'Central': ['C3', 'C4', 'Cz'],
            'Temporal': ['T3', 'T4', 'T5', 'T6'],
            'Parietal': ['P3', 'P4', 'Pz'],
            'Occipital': ['O1', 'O2']
        }
    else:
        # Source-level ROIs already represent lobes
        lobe_mapping = {
            'Frontal': ['L_Frontal', 'R_Frontal'],
            'Temporal': ['L_Temporal', 'R_Temporal'],
            'Parietal': ['L_Parietal', 'R_Parietal'],
            'Occipital': ['L_Occipital', 'R_Occipital']
        }
    
    lobe_importance = {}
    for lobe, channels in lobe_mapping.items():
        values = [node_importance.get(ch, 0) for ch in channels if ch in node_importance]
        if values:
            lobe_importance[lobe] = np.mean(values)
        else:
            lobe_importance[lobe] = 0
    
    return lobe_importance
