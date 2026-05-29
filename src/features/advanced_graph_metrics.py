"""
Advanced Graph Metrics Module.

This module provides additional graph-theoretic measures including
Hub Disruption Index and Rich-Club Coefficient.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

from ..config.config import FEATURE_CONFIG


def compute_hub_disruption_index(adjacency: np.ndarray,
                                  reference_adjacency: np.ndarray = None,
                                  threshold: float = None,
                                  hub_percentile: float = 80) -> Dict[str, float]:
    """
    Compute Hub Disruption Index.
    
    This metric quantifies the shift in hub structure, which is
    a key biomarker for AD (posterior hub disruption) and FTD
    (frontal hub disruption).
    
    Args:
        adjacency: Patient adjacency matrix
        reference_adjacency: Control/reference adjacency matrix (optional)
        threshold: Binarization threshold
        hub_percentile: Percentile for defining hubs
        
    Returns:
        Dictionary with hub metrics
    """
    if threshold is None:
        threshold = FEATURE_CONFIG.get('graph_threshold', 0.3)
    
    # Compute node strengths
    strengths = np.sum(adjacency, axis=0)
    n_nodes = len(strengths)
    
    # Identify hubs (top percentile by strength)
    hub_threshold = np.percentile(strengths, hub_percentile)
    hub_mask = strengths >= hub_threshold
    
    metrics = {
        'n_hubs': int(np.sum(hub_mask)),
        'hub_strength_mean': float(np.mean(strengths[hub_mask])) if np.any(hub_mask) else 0,
        'hub_strength_std': float(np.std(strengths[hub_mask])) if np.any(hub_mask) else 0,
        'non_hub_strength_mean': float(np.mean(strengths[~hub_mask])) if np.any(~hub_mask) else 0,
        'hub_concentration': float(np.sum(strengths[hub_mask]) / np.sum(strengths)) if np.sum(strengths) > 0 else 0,
    }
    
    # Hub vulnerability (coefficient of variation)
    if np.mean(strengths[hub_mask]) > 0:
        metrics['hub_vulnerability'] = float(np.std(strengths[hub_mask]) / np.mean(strengths[hub_mask]))
    else:
        metrics['hub_vulnerability'] = 0
    
    # Hub disruption relative to reference (if provided)
    if reference_adjacency is not None:
        ref_strengths = np.sum(reference_adjacency, axis=0)
        
        # Hub Disruption Index: correlation between strength change and original strength
        # Negative correlation = hubs are preferentially affected
        strength_change = strengths - ref_strengths
        
        if np.std(ref_strengths) > 0 and np.std(strength_change) > 0:
            corr = np.corrcoef(ref_strengths, strength_change)[0, 1]
            metrics['hub_disruption_index'] = float(corr) if not np.isnan(corr) else 0
        else:
            metrics['hub_disruption_index'] = 0
        
        # Relative hub loss
        ref_hub_mask = ref_strengths >= np.percentile(ref_strengths, hub_percentile)
        hub_strength_change = np.mean(strengths[ref_hub_mask]) - np.mean(ref_strengths[ref_hub_mask])
        metrics['hub_strength_loss'] = float(hub_strength_change)
    
    # Identify hub locations (indices)
    hub_indices = np.where(hub_mask)[0]
    metrics['hub_indices'] = hub_indices.tolist()
    
    return metrics


def compute_rich_club_coefficient(adjacency: np.ndarray,
                                   k_levels: int = 10,
                                   normalize: bool = True,
                                   n_random: int = 10) -> Dict[str, float]:
    """
    Compute Rich-Club Coefficient.
    
    Measures the tendency of well-connected nodes (hubs) to form
    tightly interconnected groups. This is often disrupted in AD.
    
    Phi(k) = E_>k / (N_>k * (N_>k - 1) / 2)
    
    Args:
        adjacency: Adjacency matrix
        k_levels: Number of degree levels to test
        normalize: Whether to normalize by random network
        n_random: Number of random networks for normalization
        
    Returns:
        Dictionary with rich-club metrics
    """
    adj_bin = (adjacency > 0).astype(float)
    degrees = np.sum(adj_bin, axis=0).astype(int)
    n_nodes = len(degrees)
    
    # Degree range
    k_min = 1
    k_max = max(int(np.max(degrees)), 2)
    k_values = np.unique(np.linspace(k_min, int(k_max * 0.8), k_levels).astype(int))
    k_values = k_values[k_values > 0]
    
    rich_club = {}
    phi_values = []
    
    for k in k_values:
        # Nodes with degree > k
        rich_nodes = np.where(degrees > k)[0]
        n_rich = len(rich_nodes)
        
        if n_rich < 2:
            continue
        
        # Count edges among rich nodes
        subgraph = adj_bin[np.ix_(rich_nodes, rich_nodes)]
        n_edges = np.sum(subgraph) / 2
        max_edges = n_rich * (n_rich - 1) / 2
        
        # Rich-club coefficient
        phi_k = n_edges / max_edges if max_edges > 0 else 0
        
        # Normalize by random networks
        if normalize and n_random > 0:
            phi_random = []
            for _ in range(n_random):
                adj_rand = _create_random_network(adj_bin)
                rand_rich_nodes = np.where(np.sum(adj_rand, axis=0) > k)[0]
                n_rand_rich = len(rand_rich_nodes)
                
                if n_rand_rich >= 2:
                    rand_subgraph = adj_rand[np.ix_(rand_rich_nodes, rand_rich_nodes)]
                    rand_edges = np.sum(rand_subgraph) / 2
                    rand_max = n_rand_rich * (n_rand_rich - 1) / 2
                    phi_random.append(rand_edges / rand_max if rand_max > 0 else 0)
            
            phi_random_mean = np.mean(phi_random) if phi_random else 1
            phi_normalized = phi_k / phi_random_mean if phi_random_mean > 0 else phi_k
            rich_club[f'phi_k{k}_norm'] = float(phi_normalized)
        
        rich_club[f'phi_k{k}'] = float(phi_k)
        phi_values.append(phi_k)
    
    # Summary statistics
    if phi_values:
        rich_club['rich_club_mean'] = float(np.mean(phi_values))
        rich_club['rich_club_max'] = float(np.max(phi_values))
        _trapezoid = getattr(np, "trapezoid", np.trapz)
        rich_club['rich_club_curve_auc'] = float(_trapezoid(phi_values))
    else:
        rich_club['rich_club_mean'] = 0
        rich_club['rich_club_max'] = 0
        rich_club['rich_club_curve_auc'] = 0
    
    return rich_club


def _create_random_network(adj_bin: np.ndarray) -> np.ndarray:
    """Create random network preserving edge count."""
    n = adj_bin.shape[0]
    n_edges = int(np.sum(adj_bin) / 2)
    
    adj_random = np.zeros_like(adj_bin)
    placed = 0
    attempts = 0
    
    while placed < n_edges and attempts < n_edges * 100:
        i, j = np.random.randint(0, n, 2)
        if i != j and adj_random[i, j] == 0:
            adj_random[i, j] = 1
            adj_random[j, i] = 1
            placed += 1
        attempts += 1
    
    return adj_random


def compute_network_resilience(adjacency: np.ndarray,
                                attack_type: str = 'targeted',
                                attack_fraction: float = 0.2) -> Dict[str, float]:
    """
    Compute network resilience to node attacks.
    
    Simulates removal of nodes (targeted at hubs or random)
    to measure network fragility.
    
    Args:
        adjacency: Adjacency matrix
        attack_type: 'targeted' (hubs first) or 'random'
        attack_fraction: Fraction of nodes to remove
        
    Returns:
        Dictionary with resilience metrics
    """
    n_nodes = adjacency.shape[0]
    n_remove = max(1, int(n_nodes * attack_fraction))
    
    # Get node strengths
    strengths = np.sum(adjacency, axis=0)
    
    # Original efficiency
    adj_bin = (adjacency > 0).astype(float)
    eff_original = _compute_efficiency(adj_bin)
    
    # Select nodes to remove
    if attack_type == 'targeted':
        # Remove hubs (highest strength)
        nodes_to_remove = np.argsort(strengths)[::-1][:n_remove]
    else:
        # Random removal
        nodes_to_remove = np.random.choice(n_nodes, n_remove, replace=False)
    
    # Create damaged network
    remaining_nodes = np.setdiff1d(np.arange(n_nodes), nodes_to_remove)
    adj_damaged = adjacency[np.ix_(remaining_nodes, remaining_nodes)]
    adj_damaged_bin = (adj_damaged > 0).astype(float)
    
    eff_after = _compute_efficiency(adj_damaged_bin) if len(remaining_nodes) > 1 else 0
    
    # Compute largest connected component
    lcc_size = _largest_component_size(adj_damaged_bin) / len(remaining_nodes) if len(remaining_nodes) > 0 else 0
    
    return {
        f'{attack_type}_attack_resilience': float(eff_after / eff_original) if eff_original > 0 else 0,
        f'{attack_type}_efficiency_loss': float(eff_original - eff_after),
        f'{attack_type}_lcc_fraction': float(lcc_size),
        'efficiency_original': float(eff_original),
        'efficiency_after_attack': float(eff_after),
    }


def _compute_efficiency(adj_bin: np.ndarray) -> float:
    """Compute global efficiency."""
    n = adj_bin.shape[0]
    if n < 2:
        return 0
    
    total = 0
    for i in range(n):
        distances = _bfs_distances(adj_bin, i)
        for j in range(n):
            if i != j and distances[j] > 0:
                total += 1.0 / distances[j]
    
    return total / (n * (n - 1))


def _bfs_distances(adj: np.ndarray, start: int) -> np.ndarray:
    """BFS to compute distances from start node."""
    n = adj.shape[0]
    distances = -np.ones(n)
    distances[start] = 0
    
    queue = [start]
    while queue:
        current = queue.pop(0)
        neighbors = np.where(adj[current] > 0)[0]
        for neighbor in neighbors:
            if distances[neighbor] < 0:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    
    distances[distances < 0] = 0
    return distances


def _largest_component_size(adj: np.ndarray) -> int:
    """Find size of largest connected component."""
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    max_size = 0
    
    for start in range(n):
        if visited[start]:
            continue
        
        # BFS from this node
        component = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            neighbors = np.where(adj[node] > 0)[0]
            queue.extend([n for n in neighbors if not visited[n]])
        
        max_size = max(max_size, len(component))
    
    return max_size


def compute_all_advanced_graph_metrics(adjacency: np.ndarray,
                                        reference: np.ndarray = None) -> Dict[str, float]:
    """
    Compute all advanced graph metrics.
    
    Args:
        adjacency: Adjacency matrix
        reference: Reference (control) adjacency matrix
        
    Returns:
        Dictionary of all advanced metrics
    """
    metrics = {}
    
    # Hub disruption
    hub_metrics = compute_hub_disruption_index(adjacency, reference)
    for k, v in hub_metrics.items():
        if not isinstance(v, (list, np.ndarray)):
            metrics[f'hub_{k}'] = v
    
    # Rich-club
    rc_metrics = compute_rich_club_coefficient(adjacency, normalize=False)
    for k, v in rc_metrics.items():
        metrics[k] = v
    
    # Resilience
    resilience = compute_network_resilience(adjacency, 'targeted', 0.1)
    for k, v in resilience.items():
        metrics[k] = v
    
    return metrics
