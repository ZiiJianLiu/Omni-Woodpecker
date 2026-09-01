"""
Alternative Implementations for Visual Subspace Construction
Used for quick testing and comparing different approaches
"""
from __future__ import annotations
import torch
from torch import Tensor
from typing import Optional
from hulluedit.steer import HullueditSteerer, HullueditConfig


class AlternativeSteerer(HullueditSteerer):
    """Extended Hulluedit Steerer supporting multiple visual subspace construction methods"""
    
    def __init__(self, config: HullueditConfig, method: str = "cosine"):
        """
        Args:
            config: Hulluedit configuration
            method: Visual subspace construction method
                - "cosine": current method (cosine similarity weighted)
                - "standard": scheme 1 - standard PCA
                - "variance": scheme 2 - variance weighted
                - "multi_text": scheme 3 - multiple text tokens
                - "hybrid": scheme 4 - hybrid weights
                - "adaptive": scheme 6 - adaptive rank
        """
        super().__init__(config)
        self.method = method
    
    def compute_evidence_subspace(
        self, 
        vis_states: Tensor, 
        last_txt: Tensor,
        txt_states: Optional[Tensor] = None
    ) -> Tensor:
        """Select different construction schemes based on method"""
        if self.method == "cosine":
            return self._compute_cosine_weighted(vis_states, last_txt)
        elif self.method == "standard":
            return self._compute_standard_pca(vis_states, last_txt)
        elif self.method == "variance":
            return self._compute_variance_weighted(vis_states, last_txt)
        elif self.method == "multi_text":
            if txt_states is None:
                # Fallback to single text
                return self._compute_cosine_weighted(vis_states, last_txt)
            return self._compute_multi_text(vis_states, txt_states)
        elif self.method == "hybrid":
            return self._compute_hybrid(vis_states, last_txt)
        elif self.method == "adaptive":
            return self._compute_adaptive_rank(vis_states, last_txt)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _compute_cosine_weighted(
        self, vis_states: Tensor, last_txt: Tensor
    ) -> Tensor:
        """Scheme 0: Current method - cosine similarity weighted"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        with torch.no_grad():
            v_norm = torch.linalg.norm(vis_states, dim=1) + self.cfg.eps
            t_norm = torch.linalg.norm(last_txt) + self.cfg.eps
            cos = (vis_states @ last_txt) / (v_norm * t_norm)
            temp = self.cfg.weight_temp if self.cfg.weight_temp > 0 else 1.0
            w = torch.softmax(cos / temp, dim=0)
            U = self._weighted_svd(vis_states, w, self.cfg.rank_evidence)
        return U
    
    def _compute_standard_pca(
        self, vis_states: Tensor, last_txt: Tensor
    ) -> Tensor:
        """Scheme 1: Standard PCA, no weighting"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        # Direct SVD, no weighting
        U = self._weighted_svd(vis_states, w=None, k=self.cfg.rank_evidence)
        return U
    
    def _compute_variance_weighted(
        self, vis_states: Tensor, last_txt: Tensor
    ) -> Tensor:
        """Scheme 2: Variance-weighted PCA"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        with torch.no_grad():
            # Compute variance for each visual token
            vis_mean = vis_states.mean(dim=0, keepdim=True)
            vis_var = ((vis_states - vis_mean) ** 2).sum(dim=1)  # [n_v]
            
            # Normalize to weights
            w = vis_var / (vis_var.sum() + self.cfg.eps)
            
            U = self._weighted_svd(vis_states, w, self.cfg.rank_evidence)
        return U
    
    def _compute_multi_text(
        self, vis_states: Tensor, txt_states: Tensor
    ) -> Tensor:
        """Scheme 3: Use similarity with multiple text tokens"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = txt_states.shape[-1] if txt_states.numel() > 0 else 4096
            return txt_states.new_zeros((d, 0))
        
        with torch.no_grad():
            # Use last N text tokens
            n_text = min(5, txt_states.shape[0])
            txt_avg = txt_states[-n_text:].mean(dim=0)  # [d]
            
            # Compute average similarity
            v_norm = torch.linalg.norm(vis_states, dim=1) + self.cfg.eps
            t_norm = torch.linalg.norm(txt_avg) + self.cfg.eps
            cos = (vis_states @ txt_avg) / (v_norm * t_norm)
            
            # Alternatively: compute similarity with each text token, then average
            # t_norms = torch.linalg.norm(txt_states[-n_text:], dim=1) + self.cfg.eps
            # cos_all = (vis_states @ txt_states[-n_text:].T) / (v_norm[:, None] * t_norms[None, :])
            # cos = cos_all.mean(dim=1)
            
            temp = self.cfg.weight_temp if self.cfg.weight_temp > 0 else 1.0
            w = torch.softmax(cos / temp, dim=0)
            
            U = self._weighted_svd(vis_states, w, self.cfg.rank_evidence)
        return U
    
    def _compute_hybrid(
        self, vis_states: Tensor, last_txt: Tensor, alpha: float = 0.7
    ) -> Tensor:
        """Scheme 4: Hybrid weights (similarity + variance)"""
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        with torch.no_grad():
            # 1. Cosine similarity weights
            v_norm = torch.linalg.norm(vis_states, dim=1) + self.cfg.eps
            t_norm = torch.linalg.norm(last_txt) + self.cfg.eps
            cos = (vis_states @ last_txt) / (v_norm * t_norm)
            temp = self.cfg.weight_temp if self.cfg.weight_temp > 0 else 1.0
            w_sim = torch.softmax(cos / temp, dim=0)
            
            # 2. Variance weights
            vis_mean = vis_states.mean(dim=0, keepdim=True)
            vis_var = ((vis_states - vis_mean) ** 2).sum(dim=1)
            w_var = vis_var / (vis_var.sum() + self.cfg.eps)
            
            # 3. Hybrid
            w = alpha * w_sim + (1 - alpha) * w_var
            w = w / (w.sum() + self.cfg.eps)  # Re-normalize
            
            U = self._weighted_svd(vis_states, w, self.cfg.rank_evidence)
        return U
    
    def _compute_adaptive_rank(
        self, vis_states: Tensor, last_txt: Tensor
    ) -> Tensor:
        """Scheme 6: Adaptive rank selection"""
        if vis_states.numel() == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        with torch.no_grad():
            # First do SVD to inspect singular values
            vis_centered = vis_states - vis_states.mean(dim=0, keepdim=True)
            
            if vis_centered.abs().max() < 1e-10:
                # If close to zero after centering, use fixed rank
                rank_adaptive = self.cfg.rank_evidence
            else:
                try:
                    _, s, _ = torch.linalg.svd(vis_centered.float(), full_matrices=False)
                    
                    # Method 1: retain principal components with cumulative energy > 0.95
                    cumsum = torch.cumsum(s, dim=0)
                    cumsum_norm = cumsum / (cumsum[-1] + 1e-12)
                    rank_adaptive = (cumsum_norm < 0.95).sum().item() + 1
                    rank_adaptive = min(rank_adaptive, len(s), self.cfg.rank_evidence * 2)
                    
                    # Clamp to reasonable range
                    rank_adaptive = max(1, min(rank_adaptive, len(s), 32))
                except Exception:
                    # If SVD fails, use fixed rank
                    rank_adaptive = self.cfg.rank_evidence
            
            if rank_adaptive == 0:
                d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
                return last_txt.new_zeros((d, 0))
            
            # Recompute with selected rank (can add weighting)
            v_norm = torch.linalg.norm(vis_states, dim=1) + self.cfg.eps
            t_norm = torch.linalg.norm(last_txt) + self.cfg.eps
            cos = (vis_states @ last_txt) / (v_norm * t_norm)
            temp = self.cfg.weight_temp if self.cfg.weight_temp > 0 else 1.0
            w = torch.softmax(cos / temp, dim=0)
            
            U = self._weighted_svd(vis_states, w, rank_adaptive)
        return U


# For compatibility, rewrite compute_evidence_subspace method
def patch_steerer_method(steerer: HullueditSteerer, method: str, alpha: float = 0.7):
    """
    Dynamically replace steerer's compute_evidence_subspace method
    
    Args:
        steerer: HullueditSteerer instance
        method: method name
        alpha: hybrid weight parameter (only used for hybrid method)
    """
    if method == "cosine":
        # Use original method
        return
    
    alt_steerer = AlternativeSteerer(steerer.cfg, method)
    
    if method == "standard":
        def new_method(self, vis_states, last_txt):
            return alt_steerer._compute_standard_pca(vis_states, last_txt)
    elif method == "variance":
        def new_method(self, vis_states, last_txt):
            return alt_steerer._compute_variance_weighted(vis_states, last_txt)
    elif method == "multi_text":
        # Need to modify call to pass txt_states
        # Not implemented here yet, requires engine code modification
        raise NotImplementedError("multi_text method requires engine code modification to pass txt_states")
    elif method == "hybrid":
        def new_method(self, vis_states, last_txt):
            return alt_steerer._compute_hybrid(vis_states, last_txt, alpha)
    elif method == "adaptive":
        def new_method(self, vis_states, last_txt):
            return alt_steerer._compute_adaptive_rank(vis_states, last_txt)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Replace method
    steerer.compute_evidence_subspace = new_method.__get__(steerer, type(steerer))
