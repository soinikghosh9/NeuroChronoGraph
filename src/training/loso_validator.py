"""
Leave-One-Subject-Out (LOSO) Cross-Validation.

This module implements LOSO cross-validation for rigorous
subject-independent evaluation of the classification model.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)
import time
from pathlib import Path
import json

from ..config.config import VALIDATION_CONFIG, DEVICE, RANDOM_SEED


class LOSOValidator:
    """
    Leave-One-Subject-Out Cross-Validation for EEG classification.
    
    Implements rigorous subject-independent validation where each subject
    is used as the test set exactly once.
    """
    
    def __init__(self,
                 model_class,
                 model_kwargs: Dict,
                 loss_fn: torch.nn.Module,
                 device: str = None,
                 verbose: bool = True):
        """
        Initialize LOSO validator.
        
        Args:
            model_class: Model class to instantiate
            model_kwargs: Keyword arguments for model initialization
            loss_fn: Loss function module
            device: Device to use ('cuda' or 'cpu')
            verbose: Whether to print progress
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.loss_fn = loss_fn
        self.device = device or str(DEVICE)
        self.verbose = verbose
        
        self.results_history = []
        
    def run(self,
            dataset,
            n_epochs: int = 100,
            patience: int = 15,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.01,
            save_models: bool = False,
            output_dir: Optional[Path] = None) -> Dict:
        """
        Run complete LOSO cross-validation.
        
        Args:
            dataset: Dataset object with get_subject_data method
            n_epochs: Maximum training epochs per fold
            patience: Early stopping patience
            learning_rate: Initial learning rate
            weight_decay: Weight decay (L2 regularization)
            save_models: Whether to save model checkpoints
            output_dir: Directory to save results
            
        Returns:
            Dictionary with aggregated results and per-subject predictions
        """
        subject_ids = dataset.get_subject_ids()
        n_subjects = len(subject_ids)
        
        all_results = []
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        start_time = time.time()
        
        for fold_idx, test_subject in enumerate(subject_ids):
            fold_start = time.time()
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Fold {fold_idx + 1}/{n_subjects}: Testing on {test_subject}")
                print(f"{'='*60}")
            
            # Get train/test split
            train_subjects = [s for s in subject_ids if s != test_subject]
            
            # Get data loaders
            train_loader = dataset.get_loader(train_subjects, shuffle=True)
            test_loader = dataset.get_loader([test_subject], shuffle=False)
            
            # Initialize fresh model
            model = self.model_class(**self.model_kwargs).to(self.device)
            
            # Optimizer and scheduler
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=n_epochs,
                eta_min=1e-6
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            for epoch in range(n_epochs):
                # Train
                train_loss = self._train_epoch(model, train_loader, optimizer)
                
                # Validate on subset of training data
                val_loss = self._validate(model, train_loader)
                
                scheduler.step()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if self.verbose:
                            print(f"  Early stopping at epoch {epoch + 1}")
                        break
                
                if self.verbose and (epoch + 1) % 20 == 0:
                    print(f"  Epoch {epoch + 1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
            
            # Load best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                model.to(self.device)
            
            # Test on held-out subject
            result = self._test_subject(model, test_loader, test_subject)
            all_results.append(result)
            
            all_predictions.append(result['pred'])
            all_true_labels.append(result['true'])
            all_probabilities.append(result['probs'])
            
            fold_time = time.time() - fold_start
            
            if self.verbose:
                print(f"  Result: True={result['true']}, Pred={result['pred']} "
                      f"(Time: {fold_time:.1f}s)")
            
            # Save model if requested
            if save_models and output_dir is not None:
                model_path = output_dir / f"model_fold{fold_idx+1}_{test_subject}.pt"
                torch.save(best_model_state, model_path)
        
        total_time = time.time() - start_time
        
        # Aggregate results
        aggregated = self._aggregate_results(
            np.array(all_true_labels),
            np.array(all_predictions),
            np.array(all_probabilities),
            all_results
        )
        
        aggregated['total_time_seconds'] = total_time
        aggregated['time_per_fold'] = total_time / n_subjects
        
        if self.verbose:
            self._print_summary(aggregated)
        
        # Save results
        if output_dir is not None:
            self._save_results(aggregated, output_dir)
        
        return aggregated
    
    def _train_epoch(self,
                     model: torch.nn.Module,
                     loader: DataLoader,
                     optimizer: torch.optim.Optimizer) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in loader:
            optimizer.zero_grad()
            
            # Move data to device
            if hasattr(batch, 'to'):
                batch = batch.to(self.device)
            
            # Forward pass
            if hasattr(batch, 'x'):  # PyG Data object
                logits = model(batch.x, batch.edge_index, batch.edge_attr, 
                              batch.batch, batch.metadata if hasattr(batch, 'metadata') else None)
                labels = batch.y
            else:  # Dictionary batch
                logits = model(**{k: v.to(self.device) if torch.is_tensor(v) else v 
                                  for k, v in batch.items() if k != 'labels'})
                labels = batch['labels'].to(self.device)
            
            loss = self.loss_fn(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def _validate(self,
                  model: torch.nn.Module,
                  loader: DataLoader) -> float:
        """Validate model."""
        model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                if hasattr(batch, 'to'):
                    batch = batch.to(self.device)
                
                if hasattr(batch, 'x'):
                    logits = model(batch.x, batch.edge_index, batch.edge_attr,
                                  batch.batch, batch.metadata if hasattr(batch, 'metadata') else None)
                    labels = batch.y
                else:
                    logits = model(**{k: v.to(self.device) if torch.is_tensor(v) else v 
                                      for k, v in batch.items() if k != 'labels'})
                    labels = batch['labels'].to(self.device)
                
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def _test_subject(self,
                      model: torch.nn.Module,
                      loader: DataLoader,
                      subject_id: str) -> Dict:
        """Test model on a single subject."""
        model.eval()
        all_preds = []
        all_probs = []
        true_label = None
        
        with torch.no_grad():
            for batch in loader:
                if hasattr(batch, 'to'):
                    batch = batch.to(self.device)
                
                if hasattr(batch, 'x'):
                    logits = model(batch.x, batch.edge_index, batch.edge_attr,
                                  batch.batch, batch.metadata if hasattr(batch, 'metadata') else None)
                    labels = batch.y
                else:
                    logits = model(**{k: v.to(self.device) if torch.is_tensor(v) else v 
                                      for k, v in batch.items() if k != 'labels'})
                    labels = batch['labels'].to(self.device)
                
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                
                if true_label is None:
                    true_label = labels[0].cpu().item()
        
        # Aggregate epoch predictions via majority voting
        from scipy import stats
        subject_pred = int(stats.mode(all_preds, keepdims=False)[0])
        mean_probs = np.concatenate(all_probs, axis=0).mean(axis=0)
        
        return {
            'subject': subject_id,
            'true': true_label,
            'pred': subject_pred,
            'probs': mean_probs,
            'epoch_preds': all_preds,
            'correct': true_label == subject_pred
        }
    
    def _aggregate_results(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_probs: np.ndarray,
                           subject_results: List[Dict]) -> Dict:
        """Aggregate fold results into final metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'f1_per_class': f1_score(y_true, y_pred, average=None).tolist(),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, 
                                                           target_names=['AD', 'FTD', 'CN']),
        }
        
        # Per-class metrics
        cm = confusion_matrix(y_true, y_pred)
        for i, class_name in enumerate(['AD', 'FTD', 'CN']):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics[f'{class_name}_sensitivity'] = sensitivity
            metrics[f'{class_name}_specificity'] = specificity
        
        metrics['y_true'] = y_true.tolist()
        metrics['y_pred'] = y_pred.tolist()
        metrics['y_probs'] = y_probs.tolist() if y_probs is not None else None
        metrics['subject_results'] = subject_results
        
        return metrics
    
    def _print_summary(self, results: Dict):
        """Print results summary."""
        print(f"\n{'='*60}")
        print("LOSO Cross-Validation Results")
        print(f"{'='*60}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1: {results['f1_macro']:.4f}")
        print(f"Weighted F1: {results['f1_weighted']:.4f}")
        print(f"MCC: {results['mcc']:.4f}")
        print(f"\nPer-class F1: AD={results['f1_per_class'][0]:.4f}, "
              f"FTD={results['f1_per_class'][1]:.4f}, CN={results['f1_per_class'][2]:.4f}")
        print(f"\nConfusion Matrix:")
        cm = np.array(results['confusion_matrix'])
        print(f"         Pred: AD  FTD   CN")
        print(f"True AD:     {cm[0,0]:3d}  {cm[0,1]:3d}  {cm[0,2]:3d}")
        print(f"True FTD:    {cm[1,0]:3d}  {cm[1,1]:3d}  {cm[1,2]:3d}")
        print(f"True CN:     {cm[2,0]:3d}  {cm[2,1]:3d}  {cm[2,2]:3d}")
        print(f"\nTotal time: {results['total_time_seconds']/60:.1f} minutes")
        print(f"{'='*60}")
    
    def _save_results(self, results: Dict, output_dir: Path):
        """Save results to file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON-serializable results
        save_results = {k: v for k, v in results.items() 
                        if k not in ['subject_results']}
        
        with open(output_dir / 'loso_results.json', 'w') as f:
            json.dump(save_results, f, indent=2)
        
        # Save detailed subject results
        subject_df_data = []
        for r in results['subject_results']:
            subject_df_data.append({
                'subject': r['subject'],
                'true': r['true'],
                'pred': r['pred'],
                'correct': r['correct'],
                'prob_AD': r['probs'][0],
                'prob_FTD': r['probs'][1],
                'prob_CN': r['probs'][2]
            })
        
        import pandas as pd
        df = pd.DataFrame(subject_df_data)
        df.to_csv(output_dir / 'subject_predictions.csv', index=False)
