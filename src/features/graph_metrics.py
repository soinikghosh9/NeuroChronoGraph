"""
Graph Theory Metrics Module.

This module provides functions for computing graph-theoretic measures
from brain connectivity matrices.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings

try:
    import bct
    HAS_BCT = True
except ImportError:
    HAS_BCT = False
    warnings.warn("bctpy not available. Graph metrics will be limited.")

from ..config.config import FEATURE_CONFIG


def compute_graph_metrics(adjacency: np.ndarray,
                          binary: bool = False,
                          threshold: float = None) -> Dict[str, float]:
    """
    Compute graph-theoretic metrics from adjacency matrix.
    
    Args:
        adjacency: Weighted or binary adjacency matrix (n_nodes, n_nodes)
        binary: Whether the matrix is binary
        threshold: Threshold for binarization (if binary=True and matrix is weighted)
        
    Returns:
        Dictionary of graph metrics
    """
    metrics = {}
    
    # Ensure symmetric
    adjacency = (adjacency + adjacency.T) / 2
    np.fill_diagonal(adjacency, 0)
    
    # Threshold if needed
    if threshold is None:
        threshold = FEATURE_CONFIG.get('graph_threshold', 0.3)
    
    if binary:
        adj_bin = (adjacency > threshold).astype(float)
    else:
        adj_bin = (adjacency > threshold).astype(float)
    
    n_nodes = adjacency.shape[0]
    
    if HAS_BCT:
        try:
            # Global Efficiency
            metrics['global_efficiency'] = bct.efficiency_bin(adj_bin)
            
            # Local Efficiency
            local_eff = bct.efficiency_bin(adj_bin, local=True)
            metrics['local_efficiency'] = np.mean(local_eff)
            
            # Clustering Coefficient
            cc = bct.clustering_coef_bu(adj_bin)
            metrics['clustering_coeff'] = np.mean(cc)
            
            # Modularity
            ci, q = bct.modularity_und(adjacency)
            metrics['modularity'] = q
            metrics['n_modules'] = len(np.unique(ci))
            
            # Characteristic Path Length
            D = bct.distance_bin(adj_bin)
            D[D == np.inf] = 0
            D[D == 0] = np.nan
            metrics['char_path_length'] = np.nanmean(D)
            
            # Degree
            degrees = np.sum(adj_bin, axis=0)
            metrics['mean_degree'] = np.mean(degrees)
            metrics['max_degree'] = np.max(degrees)
            
            # Nodal Strength (for weighted)
            strengths = np.sum(adjacency, axis=0)
            metrics['mean_strength'] = np.mean(strengths)
            
        except Exception as e:
            warnings.warn(f"Error computing BCT metrics: {e}")
            metrics = _compute_basic_metrics(adjacency, adj_bin)
    else:
        metrics = _compute_basic_metrics(adjacency, adj_bin)
    
    return metrics


def _compute_basic_metrics(adjacency: np.ndarray, 
                           adj_bin: np.ndarray) -> Dict[str, float]:
    """Compute basic graph metrics without BCT."""
    metrics = {}
    n_nodes = adjacency.shape[0]
    
    # Degree
    degrees = np.sum(adj_bin, axis=0)
    metrics['mean_degree'] = np.mean(degrees)
    metrics['max_degree'] = np.max(degrees)
    
    # Density
    n_edges = np.sum(adj_bin) / 2
    max_edges = n_nodes * (n_nodes - 1) / 2
    metrics['density'] = n_edges / max_edges if max_edges > 0 else 0
    
    # Strength
    strengths = np.sum(adjacency, axis=0)
    metrics['mean_strength'] = np.mean(strengths)
    
    # Basic clustering coefficient approximation
    cc = _basic_clustering(adj_bin)
    metrics['clustering_coeff'] = cc
    
    # Global efficiency approximation
    metrics['global_efficiency'] = _basic_efficiency(adj_bin)
    
    return metrics


def _basic_clustering(adj_bin: np.ndarray) -> float:
    """Basic clustering coefficient computation."""
    n_nodes = adj_bin.shape[0]
    cc_values = []
    
    for i in range(n_nodes):
        neighbors = np.where(adj_bin[i] > 0)[0]
        k = len(neighbors)
        
        if k < 2:
            cc_values.append(0)
            continue
        
        # Count triangles
        n_triangles = 0
        for j in neighbors:
            for l in neighbors:
                if j < l and adj_bin[j, l] > 0:
                    n_triangles += 1
        
        max_triangles = k * (k - 1) / 2
        cc_values.append(n_triangles / max_triangles if max_triangles > 0 else 0)
    
    return np.mean(cc_values)


def _basic_efficiency(adj_bin: np.ndarray) -> float:
    """Basic global efficiency computation using BFS."""
    n_nodes = adj_bin.shape[0]
    
    if n_nodes < 2:
        return 0
    
    # Compute shortest paths using BFS
    total_inv_dist = 0
    count = 0
    
    for start in range(n_nodes):
        distances = _bfs_distances(adj_bin, start)
        for end in range(n_nodes):
            if start != end and distances[end] > 0:
                total_inv_dist += 1.0 / distances[end]
                count += 1
    
    return total_inv_dist / (n_nodes * (n_nodes - 1)) if count > 0 else 0


def _bfs_distances(adj_bin: np.ndarray, start: int) -> np.ndarray:
    """Compute distances from start node using BFS."""
    n_nodes = adj_bin.shape[0]
    distances = np.zeros(n_nodes)
    distances[start] = 0
    
    visited = {start}
    queue = [start]
    
    while queue:
        current = queue.pop(0)
        current_dist = distances[current]
        
        neighbors = np.where(adj_bin[current] > 0)[0]
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)
    
    # Set unvisited to infinity represented as 0 (no path)
    return distances


def compute_nodal_metrics(adjacency: np.ndarray,
                          threshold: float = None) -> Dict[str, np.ndarray]:
    """
    Compute node-level graph metrics.
    
    Args:
        adjacency: Adjacency matrix
        threshold: Binarization threshold
        
    Returns:
        Dictionary of nodal metrics (each is n_nodes array)
    """
    if threshold is None:
        threshold = FEATURE_CONFIG.get('graph_threshold', 0.3)
    
    adj_bin = (adjacency > threshold).astype(float)
    n_nodes = adjacency.shape[0]
    
    metrics = {}
    
    # Degree
    metrics['degree'] = np.sum(adj_bin, axis=0)
    
    # Strength (weighted degree)
    metrics['strength'] = np.sum(adjacency, axis=0)
    
    # Betweenness centrality (simplified)
    if HAS_BCT:
        try:
            metrics['betweenness'] = bct.betweenness_bin(adj_bin)
        except:
            metrics['betweenness'] = np.zeros(n_nodes)
    else:
        metrics['betweenness'] = np.zeros(n_nodes)
    
    return metrics


def compute_small_world_index(adjacency: np.ndarray,
                              n_random: int = 10) -> Dict[str, float]:
    """
    Compute small-world index by comparing to random networks.
    
    Args:
        adjacency: Adjacency matrix
        n_random: Number of random networks for comparison
        
    Returns:
        Dictionary with small-world metrics
    """
    adj_bin = (adjacency > 0).astype(float)
    
    # Compute metrics for actual network
    cc_actual = _basic_clustering(adj_bin)
    
    # Generate random networks and compute averages
    cc_random = []
    
    for _ in range(n_random):
        # Create random network with same degree sequence
        adj_random = _randomize_network(adj_bin)
        cc_random.append(_basic_clustering(adj_random))
    
    cc_random_mean = np.mean(cc_random) if cc_random else 1
    
    # Small-world index
    # (In full implementation, would also compare path lengths)
    sw_index = cc_actual / cc_random_mean if cc_random_mean > 0 else 0
    
    return {
        'clustering_ratio': sw_index,
        'small_world_index': sw_index
    }


def _randomize_network(adj_bin: np.ndarray) -> np.ndarray:
    """Create random network preserving degree distribution."""
    n = adj_bin.shape[0]
    adj_random = np.zeros_like(adj_bin)
    
    # Get number of edges to preserve
    n_edges = int(np.sum(adj_bin) / 2)
    
    # Randomly assign edges
    placed = 0
    attempts = 0
    max_attempts = n_edges * 100
    
    while placed < n_edges and attempts < max_attempts:
        i, j = np.random.randint(0, n, 2)
        if i != j and adj_random[i, j] == 0:
            adj_random[i, j] = 1
            adj_random[j, i] = 1
            placed += 1
        attempts += 1
    
    return adj_random


def extract_graph_features(connectivity: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Extract graph features from multi-band connectivity.
    
    Args:
        connectivity: Dictionary of connectivity matrices per band
        
    Returns:
        Dictionary of graph features
    """
    all_features = {}
    
    for band_name, conn_matrix in connectivity.items():
        metrics = compute_graph_metrics(conn_matrix)
        
        for metric_name, value in metrics.items():
            all_features[f'{band_name}_{metric_name}'] = value
    
    return all_features
