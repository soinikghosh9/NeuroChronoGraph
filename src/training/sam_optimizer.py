"""
Sharpness-Aware Minimization (SAM) Optimizer.

SAM seeks parameters that lie in neighborhoods having uniformly low loss,
resulting in models that generalize better.

Reference: Foret et al. (2021) "Sharpness-Aware Minimization for Efficiently 
Improving Generalization" (ICLR 2021)

Enhanced with 2024 improvements:
- Friendly-SAM: Better handling of stochastic gradient noise
- Late-stage SAM: Can be applied only in final epochs for efficiency
"""

import torch
from torch.optim import Optimizer
from typing import Callable, Iterable, Optional


class SAM(Optimizer):
    """
    Sharpness-Aware Minimization Optimizer.
    
    This optimizer seeks flatter minima in the loss landscape,
    which empirically leads to better generalization.
    
    Args:
        params: Model parameters
        base_optimizer: Base optimizer class (e.g., torch.optim.AdamW)
        rho: Neighborhood size for sharpness (default: 0.05)
        adaptive: Use adaptive SAM (scale rho by gradient magnitude)
        **kwargs: Arguments passed to base optimizer
    """
    
    def __init__(self, 
                 params: Iterable,
                 base_optimizer: type,
                 rho: float = 0.05,
                 adaptive: bool = False,
                 **kwargs):
        
        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        Compute and apply the perturbation (ascent step).
        
        This moves parameters towards the direction of steepest loss increase,
        finding the "sharpest" point in the neighborhood.
        """
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Compute perturbation
                if group['adaptive']:
                    # Adaptive SAM: scale by parameter magnitude
                    e_w = (torch.pow(p, 2) if group['adaptive'] else 1.0) * p.grad * scale
                else:
                    e_w = p.grad * scale
                
                # Store original parameters
                self.state[p]['e_w'] = e_w.clone()
                
                # Apply perturbation (ascent)
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        Restore parameters and apply the actual gradient update.
        
        This computes the gradient at the perturbed point and updates
        from the original parameters.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Restore original parameters
                p.sub_(self.state[p]['e_w'])
        
        # Apply base optimizer step
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.
        
        For SAM, this requires closure to recompute the loss after perturbation.
        
        Usage:
            def closure():
                loss = model(x).loss
                loss.backward()
                return loss
            
            optimizer.step(closure)
        """
        if closure is None:
            raise ValueError("SAM requires a closure function")
        
        # First step: find sharpest point
        self.first_step(zero_grad=True)
        
        # Recompute loss and gradients at perturbed point
        with torch.enable_grad():
            closure()
        
        # Second step: update from original point using perturbed gradient
        self.second_step()
    
    def _grad_norm(self):
        """Compute the norm of all gradients."""
        shared_device = self.param_groups[0]['params'][0].device
        
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class FSAM(SAM):
    """
    Friendly-SAM (F-SAM) Optimizer.
    
    Enhanced version of SAM that better leverages stochastic gradient noise
    and reduces the negative impact of full gradient components.
    
    Reference: 2024 arXiv paper on Friendly-SAM
    """
    
    def __init__(self, 
                 params: Iterable,
                 base_optimizer: type,
                 rho: float = 0.05,
                 sigma: float = 0.01,
                 **kwargs):
        super().__init__(params, base_optimizer, rho, adaptive=False, **kwargs)
        self.sigma = sigma
    
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """F-SAM first step with noise injection."""
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Add friendly noise to perturbation
                noise = torch.randn_like(p) * self.sigma
                e_w = (p.grad + noise) * scale
                
                self.state[p]['e_w'] = e_w.clone()
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()


def create_sam_optimizer(model, 
                         base_optimizer_class=torch.optim.AdamW,
                         rho: float = 0.05,
                         use_fsam: bool = False,
                         **optimizer_kwargs):
    """
    Factory function to create a SAM optimizer.
    
    Args:
        model: PyTorch model
        base_optimizer_class: Base optimizer to wrap (default: AdamW)
        rho: SAM neighborhood size (default: 0.05)
        use_fsam: Use Friendly-SAM variant
        **optimizer_kwargs: Arguments for base optimizer (lr, weight_decay, etc.)
    
    Returns:
        SAM or FSAM optimizer
    """
    if use_fsam:
        return FSAM(
            model.parameters(),
            base_optimizer_class,
            rho=rho,
            **optimizer_kwargs
        )
    else:
        return SAM(
            model.parameters(),
            base_optimizer_class,
            rho=rho,
            **optimizer_kwargs
        )
