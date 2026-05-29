"""
Hierarchical Clinical Conditioning Module.

This module implements advanced FiLM conditioning with:
- Layer-wise conditioning (different clinical factors at different depths)
- Uncertainty estimation from clinical data
- Disease stage embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation Layer.
    
    Modulates features using learned scaling (gamma) and shifting (beta).
    """
    
    def __init__(self,
                 feature_dim: int,
                 condition_dim: int,
                 hidden_dim: int = 64):
        super().__init__()
        
        self.gamma_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.beta_net = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Initialize to identity modulation
        nn.init.zeros_(self.gamma_net[-1].weight)
        nn.init.ones_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].weight)
        nn.init.zeros_(self.beta_net[-1].bias)
        
    def forward(self,
                features: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.
        
        Args:
            features: Input features [batch, ..., feature_dim]
            condition: Conditioning vector [batch, condition_dim]
            
        Returns:
            Modulated features
        """
        gamma = self.gamma_net(condition)  # [B, feature_dim]
        beta = self.beta_net(condition)    # [B, feature_dim]
        
        # Expand for broadcasting
        while gamma.dim() < features.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        
        return gamma * features + beta


class HierarchicalFiLMConditioner(nn.Module):
    """
    Hierarchical Clinical Conditioner.
    
    Different clinical factors affect different network depths:
    - Early layers: Age effects (general brain aging patterns)
    - Mid layers: Disease severity (MMSE-based)
    - Late layers: Disease type specificity
    
    Also includes uncertainty estimation.
    """
    
    def __init__(self,
                 feature_dim: int,
                 n_layers: int,
                 age_dim: int = 1,
                 mmse_dim: int = 1,
                 sex_dim: int = 1,
                 hidden_dim: int = 64,
                 use_uncertainty: bool = True,
                 combined_dim: int = None):
        """
        Initialize HierarchicalFiLMConditioner.
        
        Args:
            feature_dim: Dimension of features to modulate
            n_layers: Total number of layers in the model
            age_dim: Dimension of age encoding
            mmse_dim: Dimension of MMSE encoding
            sex_dim: Dimension of sex encoding
            hidden_dim: Hidden dimension for conditioning networks
            use_uncertainty: Whether to estimate uncertainty
            combined_dim: Dimension of features for the late stage (defaults to feature_dim)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.n_layers = n_layers
        self.use_uncertainty = use_uncertainty
        self.clinical_dropout = 0.2  # Default clinical dropout rate

        
        # Use combined_dim if provided, else use feature_dim
        self.combined_dim = combined_dim if combined_dim is not None else feature_dim
        
        # Divide layers into stages
        self.stage_size = n_layers // 3
        
        # Clinical embeddings
        self.age_embedder = nn.Sequential(
            nn.Linear(age_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.mmse_embedder = nn.Sequential(
            nn.Linear(mmse_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.sex_embedder = nn.Embedding(2, hidden_dim)
        
        # Disease stage embeddings (learned based on MMSE ranges)
        self.stage_embedder = nn.Embedding(4, hidden_dim)  # Normal, MCI, Mild, Moderate+
        
        # Stage-specific FiLM layers
        # Early: Age conditioning
        self.age_film_layers = nn.ModuleList([
            FiLMLayer(feature_dim, hidden_dim, hidden_dim)
            for _ in range(self.stage_size)
        ])
        
        # Mid: MMSE conditioning
        self.mmse_film_layers = nn.ModuleList([
            FiLMLayer(feature_dim, hidden_dim, hidden_dim)
            for _ in range(self.stage_size)
        ])
        
        # Late: Combined clinical conditioning
        self.combined_film_layers = nn.ModuleList([
            FiLMLayer(self.combined_dim, hidden_dim * 3, hidden_dim)
            for _ in range(n_layers - 2 * self.stage_size)
        ])
        
        # Uncertainty estimation
        if use_uncertainty:
            self.uncertainty_net = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)  # mean, log_var
            )
        
        # Domain embedding (for potential multi-site adaptation)
        self.domain_embedding = nn.Embedding(10, hidden_dim)  # Up to 10 sites
        
    def get_disease_stage(self, mmse: torch.Tensor) -> torch.Tensor:
        """Convert MMSE score to disease stage index."""
        # MMSE ranges: 30=normal, 25-29=MCI, 20-24=Mild, <20=Moderate+
        stage = torch.zeros_like(mmse, dtype=torch.long)
        stage[mmse < 30] = 1  # MCI
        stage[mmse < 25] = 2  # Mild
        stage[mmse < 20] = 3  # Moderate+
        return stage
        
    def forward(self,
                features: torch.Tensor,
                layer_idx: int,
                clinical_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply hierarchical conditioning at specified layer.
        
        Args:
            features: Input features [batch, ..., feature_dim]
            layer_idx: Index of current layer
            clinical_data: Dict with 'age', 'mmse', 'sex'
            
        Returns:
            Modulated features and uncertainty (if applicable)
        """
        batch_size = features.shape[0]
        device = features.device
        
        # Embed clinical data - ensuring consistent tensor shapes
        age = clinical_data.get('age', torch.zeros(batch_size, 1, device=device))
        mmse = clinical_data.get('mmse', torch.ones(batch_size, 1, device=device) * 30)
        sex = clinical_data.get('sex', torch.zeros(batch_size, dtype=torch.long, device=device))
        
        # Ensure age and mmse are 2D [batch, 1]
        if age.dim() == 1:
            age = age.unsqueeze(-1)
        if mmse.dim() == 1:
            mmse = mmse.unsqueeze(-1)
        
        # Ensure sex is 1D for embedding lookup
        if sex.dim() > 1:
            sex = sex.squeeze(-1)
        sex = sex.long()
        
        # Missing MMSE Masking (Standardized -1.0)
        # If MMSE is < 0, it means it's missing (Mixed datasets)
        # We zero out the embedding to signal "neutral/unknown"
        mmse_missing_mask = (mmse >= 0).float()
        
        # Clinical Dropout (Training only)
        if self.training and self.clinical_dropout > 0:
            dropout_mask = (torch.rand(batch_size, 1, device=device) > self.clinical_dropout).float()
            # Combine missingness with dropout
            # If missing, it stays 0. If present, it might become 0 due to dropout.
            final_mmse_mask = mmse_missing_mask * dropout_mask
            final_age_mask = dropout_mask # Age usually dummy 70 if missing but we treat same
        else:
            final_mmse_mask = mmse_missing_mask
            final_age_mask = torch.ones_like(age)
            
        age_embed = self.age_embedder(age) * final_age_mask
        mmse_embed = self.mmse_embedder(mmse) * final_mmse_mask
        sex_embed = self.sex_embedder(sex) * final_age_mask # Dropout sex too if we dropout age
        
        # Disease stage
        mmse_for_stage = mmse.squeeze(-1) if mmse.dim() > 1 else mmse
        stage = self.get_disease_stage(mmse_for_stage)
        stage_embed = self.stage_embedder(stage) * final_mmse_mask # Mask stage if MMSE missing
        
        # Combined embedding - all should be [B, hidden]
        combined = torch.cat([age_embed, mmse_embed + stage_embed, sex_embed], dim=-1)
        
        # Determine which stage this layer belongs to
        if layer_idx < self.stage_size:
            # Early layers: Age conditioning
            idx = layer_idx
            modulated = self.age_film_layers[idx](features, age_embed)
            
        elif layer_idx < 2 * self.stage_size:
            # Mid layers: MMSE conditioning
            idx = layer_idx - self.stage_size
            modulated = self.mmse_film_layers[idx](features, mmse_embed + stage_embed)
            
        else:
            # Late layers: Combined conditioning
            idx = layer_idx - 2 * self.stage_size
            idx = min(idx, len(self.combined_film_layers) - 1)
            modulated = self.combined_film_layers[idx](features, combined)
        
        # Compute uncertainty if enabled
        uncertainty = None
        if self.use_uncertainty:
            unc_params = self.uncertainty_net(combined)
            mean, log_var = unc_params[:, 0], unc_params[:, 1]
            uncertainty = torch.exp(0.5 * log_var)
            
            # Scale modulation by confidence (inverse uncertainty)
            confidence = 1.0 / (1.0 + uncertainty)
            while confidence.dim() < modulated.dim():
                confidence = confidence.unsqueeze(-1)
            
            # Blend: high confidence = more modulation, low = closer to original
            modulated = confidence * modulated + (1 - confidence) * features
        
        return modulated, uncertainty


class DiseaseStageEncoder(nn.Module):
    """
    Encode disease severity/stage from clinical indicators.
    """
    
    def __init__(self,
                 hidden_dim: int = 64,
                 output_dim: int = 64):
        super().__init__()
        
        self.mmse_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.age_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Interaction modeling
        self.interaction = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Stage classifier (auxiliary task)
        self.stage_classifier = nn.Linear(output_dim, 4)
        
    def forward(self,
                age: torch.Tensor,
                mmse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode disease stage.
        
        Returns:
            stage_embedding: Continuous stage representation
            stage_logits: Stage classification logits
        """
        age_embed = self.age_encoder(age)
        mmse_embed = self.mmse_encoder(mmse)
        
        combined = torch.cat([age_embed, mmse_embed], dim=-1)
        stage_embed = self.interaction(combined)
        
        stage_logits = self.stage_classifier(stage_embed)
        
        return stage_embed, stage_logits


class UncertaintyHead(nn.Module):
    """
    Estimate prediction uncertainty for clinical validity.
    
    Uses evidential deep learning for uncertainty quantification.
    """
    
    def __init__(self,
                 input_dim: int,
                 n_classes: int = 3):
        super().__init__()
        
        self.n_classes = n_classes
        
        # Evidence predictor (Dirichlet distribution parameters)
        self.evidence_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, n_classes),
            nn.Softplus()  # Evidence must be positive
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute predictions with uncertainty.
        
        Returns:
            Dict with 'prob', 'uncertainty', 'alpha'
        """
        # Evidence (alpha - 1 in Dirichlet)
        evidence = self.evidence_net(features)
        alpha = evidence + 1  # Dirichlet parameters
        
        # Total evidence
        S = alpha.sum(dim=-1, keepdim=True)
        
        # Probability (expected value of Dirichlet)
        prob = alpha / S
        
        # Uncertainty (inverse of total evidence)
        uncertainty = self.n_classes / S.squeeze(-1)
        
        return {
            'prob': prob,
            'uncertainty': uncertainty,
            'alpha': alpha,
            'evidence': evidence
        }
    
    def compute_loss(self,
                     outputs: Dict[str, torch.Tensor],
                     targets: torch.Tensor) -> torch.Tensor:
        """
        Compute evidential loss for uncertainty-aware training.
        """
        alpha = outputs['alpha']
        S = alpha.sum(dim=-1, keepdim=True)
        
        # One-hot targets
        targets_oh = F.one_hot(targets, self.n_classes).float()
        
        # Expected log likelihood
        A = torch.sum(targets_oh * (torch.digamma(S) - torch.digamma(alpha)), dim=-1)
        
        # KL divergence from uniform prior
        alpha0 = torch.ones_like(alpha)
        kl = self._kl_divergence(alpha, alpha0)
        
        # Annealing coefficient (increases over training)
        lambda_t = 0.1  # Should increase from 0 to 1
        
        return A.mean() + lambda_t * kl.mean()
    
    def _kl_divergence(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """KL divergence between two Dirichlet distributions."""
        S_alpha = alpha.sum(dim=-1, keepdim=True)
        S_beta = beta.sum(dim=-1, keepdim=True)
        
        kl = (torch.lgamma(S_alpha) - torch.lgamma(S_beta) - 
              torch.lgamma(alpha).sum(dim=-1, keepdim=True) + 
              torch.lgamma(beta).sum(dim=-1, keepdim=True))
        
        kl = kl + torch.sum((alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha)), dim=-1, keepdim=True)
        
        return kl.squeeze(-1)
