"""
Hulluedit Core Module: Evidence Subspace Editor
Implements online subspace estimation, trust domain gating, and evidence certification
"""
from __future__ import annotations
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional


@dataclass
class HullueditConfig:
    """Hulluedit Configuration"""
    rank_evidence: int = 6      # Evidence subspace rank r
    rank_prior: int = 4          # Anti-prior subspace rank q
    kappa: float = 0.6           # Trust domain gating coefficient
    lambda_prior: float = 0.3    # Prior suppression strength
    eps: float = 1e-6            # Numerical stability parameter
    # Robustness hyperparameters for closed-form editing
    lambda_n_max: float = 4.0     # Upper bound for lambda_n, prevents over-contraction
    lambda_p_max: float = 4.0     # Upper bound for lambda_p, prevents over-contraction
    vcr_floor: float = 0.05       # Lower bound for VCR, suppresses lambda_n explosion
    pcr_ceiling: float = 0.95     # Upper bound for PCR, suppresses lambda_p explosion
    pcr_threshold: float = 0.02   # Low conflict threshold, prior not suppressed below this
    blend_tau: float = 0.7        # Blend ratio with original h: h_out = (1-tau)h + tau h_closed
    norm_preserve: bool = True    # Whether to preserve norm scale
    norm_beta: float = 0.5        # Norm restoration exponent: s^beta, s=||h||/||h'||
    weight_temp: float = 1.5      # Visual weight temperature: softmax(cos/temperature)
    # Ablation and variant controls
    uniform_svd: bool = False     # Do not use weighting for evidence subspace extraction (uniform SVD)
    no_complement: bool = False   # Anti-prior subspace not projected to visual complement (disable orthogonality)
    no_gating: bool = False       # Disable certificate-based gating (lambda_n uses fixed value)
    use_fixed_strengths: bool = False  # Fixed editing strength (overrides adaptive)
    fixed_lambda_n: float = 0.0   # Fixed residual contraction strength
    fixed_lambda_p: float = 0.0   # Fixed anti-prior contraction strength
    only_residual: bool = False   # Only reduce residual component (disable anti-prior)
    only_anti_prior: bool = False # Only reduce anti-prior component (disable residual reduction)


@dataclass
class HullueditReturn:
    """Hulluedit Edit Return Values"""
    h_edited: Tensor      # Edited hidden state [d]
    gate: Tensor          # Gating value (scalar)
    vcr: Tensor           # Vision Coverage Ratio
    pcr: Tensor           # Prior Conflict Ratio
    U: Tensor             # Evidence subspace basis [d, r]
    P: Tensor             # Anti-prior subspace basis [d, q]


class HullueditSteerer:
    """Hulluedit Steerer: Online Subspace Estimation and Minimum Norm Editing"""
    
    def __init__(self, config: HullueditConfig):
        self.cfg = config

    @staticmethod
    def _weighted_svd(X: Tensor, w: Optional[Tensor], k: int) -> Tensor:
        """
        Extract top k principal component directions using weighted SVD (suitable for n << d)
        
        When the number of samples is much smaller than the dimension, perform SVD directly
        on the data matrix to avoid constructing a large covariance matrix.
        This is numerically more stable and efficient.
        
        Args:
            X: [n, d] sample matrix, typically n << d (e.g., visual tokens: 576 << 4096)
            w: [n] weight vector (optional), used for weighted PCA
            k: number of principal components
        Returns:
            V: [d, k_actual] top k principal component directions (sorted by singular values descending, actual may be less than k)
        """
        if k <= 0 or X.numel() == 0:
            d = X.shape[-1] if X.numel() > 0 else 4096
            return X.new_zeros((d, 0))
        
        n, d = X.shape
        if n == 0 or d == 0:
            return X.new_zeros((d, 0))
        
        # Centering
        if w is None:
            Xc = X - X.mean(dim=0, keepdim=True)
        else:
            # Weighted centering
            if w.shape[0] != n:
                raise ValueError(f"Weight dimension {w.shape[0]} does not match sample count {n}")
            w = w / (w.sum() + 1e-12)
            mu = (w[:, None] * X).sum(dim=0, keepdim=True)
            Xc = X - mu
        
        # Determine actual extractable k (cannot exceed rank)
        max_rank = min(n, d)
        k_actual = min(k, max_rank)
        if k_actual == 0:
            return Xc.new_zeros((d, 0))
        
        # Apply weights (weighted SVD: equivalent to SVD on weighted data)
        if w is not None:
            # Weighting: X_weighted = diag(sqrt(w)) @ Xc
            # Ensure weights are positive and numerically stable
            w_sqrt = torch.sqrt(torch.clamp(w, min=0.0))
            Xc = Xc * w_sqrt[:, None]
        
        # Perform SVD on [n, d] matrix
        # Xc = U_n @ Sigma @ V^T, where:
        # - U_n: [n, min(n,d)] left singular vectors
        # - Sigma: [min(n,d)] singular values (descending)
        # - Vt: [min(n,d), d] right singular vectors (transposed), Vt[i] is the i-th principal direction
        orig_dtype = Xc.dtype
        Xc_float = Xc.float() if Xc.dtype == torch.float16 else Xc
        
        try:
            # full_matrices=False: only compute min(n,d) singular vectors, more efficient
            # Check numerical stability: if matrix is too small or all zeros, return zero matrix
            if Xc_float.abs().max() < 1e-10:
                return Xc.new_zeros((d, k_actual))
            
            _, _, Vt = torch.linalg.svd(Xc_float, full_matrices=False)
            # Vt: [min(n,d), d], each row is a principal component direction (sorted by singular values descending)
            # Ensure k_actual does not exceed Vt's first dimension (theoretically k_actual <= Vt.shape[0], but check for safety)
            k_extract = min(k_actual, Vt.shape[0])
            if k_extract > 0:
                V = Vt[:k_extract, :].T.to(orig_dtype)  # [d, k_extract]
            else:
                V = Xc.new_zeros((d, 0))
        except RuntimeError as e:
            # If SVD fails (theoretically shouldn't), fallback to original method
            print(f"[WARN] SVD failed, fallback to covariance method: {e}")
            # Fallback: recompute centered data, use covariance method with stronger regularization
            # Note: need to recompute here because Xc may have been modified (weighted)
            if w is not None:
                w_norm = w / (w.sum() + 1e-12)
                mu = (w_norm[:, None] * X).sum(dim=0, keepdim=True)
                Xc_fallback = X - mu
                # Weighted covariance: C = Xc^T @ diag(w) @ Xc
                C = Xc_fallback.T @ (w_norm[:, None] * Xc_fallback)
            else:
                Xc_fallback = X - X.mean(dim=0, keepdim=True)
                # Standard covariance
                C = Xc_fallback.T @ Xc_fallback / max(1, X.shape[0] - 1)
            # Use stronger regularization
            C_reg = C.float() + 1e-3 * torch.eye(C.shape[0], device=C.device, dtype=torch.float32)
            evals, evecs = torch.linalg.eigh(C_reg)
            # Ensure k does not exceed number of eigenvectors (k_actual in fallback path should match normal path)
            # Use the same k_actual calculation as normal path here
            max_rank_fallback = min(n, d)  # Recompute max_rank (n and d are defined at function start)
            k_actual_fallback = min(k, max_rank_fallback)
            k_extract = min(k_actual_fallback, evecs.shape[1])
            if k_extract > 0:
                V = evecs[:, -k_extract:].to(orig_dtype)
            else:
                V = Xc.new_zeros((d, 0))
        
        return V

    @staticmethod
    def _weighted_cov(X: Tensor, w: Optional[Tensor]) -> Tensor:
        """
        Compute weighted covariance matrix (deprecated, kept for backward compatibility)
        Args:
            X: [n, d] sample matrix
            w: [n] weight vector (optional)
        Returns:
            C: [d, d] covariance matrix
        """
        if w is None:
            # No weights: standard covariance
            Xc = X - X.mean(dim=0, keepdim=True)
            return Xc.T @ Xc / max(1, X.shape[0] - 1)
        # Weighted covariance
        w = w / (w.sum() + 1e-12)
        mu = (w[:, None] * X).sum(dim=0, keepdim=True)
        Xc = X - mu
        return (Xc.T * w) @ Xc

    @staticmethod
    def _top_eigvecs(C: Tensor, k: int, reg: float = 1e-5) -> Tensor:
        """
        Extract top k principal eigenvectors of covariance matrix (deprecated, kept for backward compatibility)
        Args:
            C: [d, d] symmetric positive definite matrix
            k: number of eigenvectors
            reg: regularization coefficient, used to stabilize ill-conditioned matrices
        Returns:
            V: [d, k] top k eigenvectors (sorted by eigenvalues descending)
        """
        if k <= 0:
            return C.new_zeros((C.shape[0], 0))
        # linalg.eigh does not support Half on CUDA, need to convert to float32
        orig_dtype = C.dtype
        C_float = C.float() if C.dtype == torch.float16 else C
        
        # Add regularization to prevent ill-conditioning
        C_reg = C_float + reg * torch.eye(C_float.shape[0], device=C_float.device, dtype=C_float.dtype)
        
        try:
            evals, evecs = torch.linalg.eigh(C_reg)  # Ascending order
        except RuntimeError as e:
            # If still fails, use stronger regularization
            print(f"[WARN] Eigendecomposition failed, using stronger regularization: {e}")
            C_reg = C_float + 1e-3 * torch.eye(C_float.shape[0], device=C_float.device, dtype=torch.float32)
            evals, evecs = torch.linalg.eigh(C_reg)
        
        evecs = evecs.to(orig_dtype)  # Convert back to original precision
        return evecs[:, -k:]  # Take largest k

    def compute_evidence_subspace(
        self, 
        vis_states: Tensor, 
        last_txt: Tensor
    ) -> Tensor:
        """
        Compute evidence subspace U (based on visual-text similarity weighted SVD)
        
        Use SVD method to directly extract principal components, avoiding construction
        of large covariance matrix. This is more stable and efficient when the number
        of samples is much smaller than the dimension (n_v << d).
        
        Args:
            vis_states: [n_v, d] visual token hidden states (n_v typically hundreds, d=4096)
            last_txt: [d] current text terminal hidden state
        Returns:
            U: [d, r] evidence subspace basis vectors
        """
        if vis_states.numel() == 0 or self.cfg.rank_evidence == 0:
            d = last_txt.shape[0] if last_txt.numel() > 0 else 4096
            return last_txt.new_zeros((d, 0))
        
        with torch.no_grad():
            # Compute cosine similarity weights (can switch to uniform SVD)
            if self.cfg.uniform_svd:
                w = None
            else:
                v_norm = torch.linalg.norm(vis_states, dim=1) + self.cfg.eps
                t_norm = torch.linalg.norm(last_txt) + self.cfg.eps
                cos = (vis_states @ last_txt) / (v_norm * t_norm)
                # Use temperature smoothing, temp>1 is more uniform; <=0 degrades to no temperature
                temp = self.cfg.weight_temp if self.cfg.weight_temp > 0 else 1.0
                w = torch.softmax(cos / temp, dim=0)
            # Use (weighted/uniform) SVD to directly extract principal components (avoid constructing 4096x4096 covariance matrix)
            U = self._weighted_svd(vis_states, w, self.cfg.rank_evidence)
        return U

    def compute_anti_prior_subspace(self, nonvis_txt_states: Tensor, U: Tensor) -> Tensor:
        """
        Compute anti-prior subspace P (non-visual text principal components in visual complement space)
        
        According to methodology: first project non-visual text states to visual complement space (I - UU^T),
        then extract principal components in that space to ensure U^T P = 0, avoiding conflict with visual evidence.
        
        Args:
            nonvis_txt_states: [n_t, d] non-visual text token hidden states
            U: [d, r] evidence subspace basis
        Returns:
            P: [d, q] anti-prior subspace basis vectors (in visual complement space)
        """
        if nonvis_txt_states.numel() == 0 or self.cfg.rank_prior == 0:
            d = nonvis_txt_states.shape[-1] if nonvis_txt_states.numel() > 0 else (U.shape[0] if U.numel() > 0 else 4096)
            return nonvis_txt_states.new_zeros((d, 0))
        
        with torch.no_grad():
            # First project to visual complement space: T_perp = (I - UU^T) T (can be disabled)
            if (not self.cfg.no_complement) and U.numel() > 0:
                # Use equivalent efficient form to avoid explicit construction of I: T - (T U) U^T
                TU = nonvis_txt_states @ U            # [n_t, r]
                txt_in_complement = nonvis_txt_states - (TU @ U.T)  # [n_t, d]
            else:
                txt_in_complement = nonvis_txt_states
            
            # Extract principal components in complement space
            P = self._weighted_svd(txt_in_complement, None, self.cfg.rank_prior)
        return P

    def trust_gate(self, h: Tensor, U: Tensor) -> Tensor:
        """
        Compute trust domain gating value (based on evidence coverage)
        Args:
            h: [..., d] hidden state
            U: [d, r] evidence subspace
        Returns:
            gate: [...] gating value (0=high evidence, 1=low evidence)
        """
        if U.numel() == 0:
            return torch.zeros(h.shape[:-1], device=h.device, dtype=h.dtype)
        
        Uh = h @ U  # [..., r] project to evidence subspace
        num = (Uh * Uh).sum(dim=-1)
        den = (h * h).sum(dim=-1) + self.cfg.eps
        evidence_score = (num / den).clamp(0.0, 1.0)  # Evidence coverage ratio
        gate = self.cfg.kappa * (1.0 - evidence_score)  # Reverse gating
        return gate

    def edit_text_hidden(
        self, 
        h_last_txt: Tensor, 
        U: Tensor, 
        P: Tensor
    ) -> HullueditReturn:
        """
        Perform Hulluedit editing on text terminal hidden state
        Args:
            h_last_txt: [d] hidden state of current generation token
            U: [d, r] evidence subspace
            P: [d, q] anti-prior subspace
        Returns:
            HullueditReturn: edit result + certificate
        """
        # Decompose h to U, P and residual subspace
        if U.numel() > 0:
            h_U = U @ (U.T @ h_last_txt)
        else:
            h_U = torch.zeros_like(h_last_txt)

        if P.numel() > 0:
            h_P = P @ (P.T @ h_last_txt)
        else:
            h_P = torch.zeros_like(h_last_txt)

        h_R = h_last_txt - h_U - h_P

        # Compute VCR/PCR (with robust clipping)
        h_norm_sq = (h_last_txt * h_last_txt).sum() + self.cfg.eps
        vcr_raw = (h_U * h_U).sum() / h_norm_sq
        pcr_raw = (h_P * h_P).sum() / h_norm_sq
        vcr = torch.clamp(vcr_raw, min=self.cfg.vcr_floor, max=1.0)
        pcr = torch.clamp(pcr_raw, min=0.0, max=self.cfg.pcr_ceiling)

        # Compute edit strength
        if self.cfg.use_fixed_strengths:
            # Fixed strength (overrides all adaptive mechanisms)
            lambda_n = torch.tensor(float(max(0.0, self.cfg.fixed_lambda_n)),
                                    device=h_last_txt.device, dtype=h_last_txt.dtype)
            lambda_p = torch.tensor(float(max(0.0, self.cfg.fixed_lambda_p)),
                                    device=h_last_txt.device, dtype=h_last_txt.dtype)
        else:
            # Adaptive strength (closed-form minimum norm editing)
            if self.cfg.only_anti_prior:
                lambda_n = torch.zeros((), device=h_last_txt.device, dtype=h_last_txt.dtype)
            elif self.cfg.no_gating:
                # Disable gating: use constant lambda_n, independent of evidence coverage
                lambda_n = torch.tensor(float(max(0.0, self.cfg.kappa)),
                                        device=h_last_txt.device, dtype=h_last_txt.dtype)
            else:
                lambda_n = self.cfg.kappa * (1.0 - vcr) / (vcr + self.cfg.eps)
            if self.cfg.only_residual:
                lambda_p = torch.zeros((), device=h_last_txt.device, dtype=h_last_txt.dtype)
            elif pcr < self.cfg.pcr_threshold:
                lambda_p = torch.zeros((), device=h_last_txt.device, dtype=h_last_txt.dtype)
            else:
                lambda_p = self.cfg.lambda_prior * pcr / (1.0 - pcr + self.cfg.eps)

        # Clipping, prevents over-contraction
        lambda_n = torch.clamp(lambda_n, min=0.0, max=self.cfg.lambda_n_max)
        lambda_p = torch.clamp(lambda_p, min=0.0, max=self.cfg.lambda_p_max)

        # Closed-form scaling
        h_edited = h_U + h_P / (1.0 + lambda_n + lambda_p) + h_R / (1.0 + lambda_n)

        # Optional: preserve norm scale, avoid logits shrinking too much
        if self.cfg.norm_preserve:
            h_prime_norm = torch.sqrt((h_edited * h_edited).sum() + self.cfg.eps)
            h_orig_norm = torch.sqrt((h_last_txt * h_last_txt).sum() + self.cfg.eps)
            scale = (h_orig_norm / h_prime_norm).pow(self.cfg.norm_beta)
            h_edited = h_edited * scale

        # Blend with original state for improved robustness
        tau = float(self.cfg.blend_tau)
        tau = max(0.0, min(1.0, tau))
        if tau < 1.0:
            h_edited = (1.0 - tau) * h_last_txt + tau * h_edited

        # Record metrics; gate characterized by equivalent scaling strength
        gate = lambda_n / (1.0 + lambda_n)

        return HullueditReturn(
            h_edited=h_edited,
            gate=gate,
            vcr=vcr,
            pcr=pcr,
            U=U,
            P=P
        )

    def clean_visual_tokens(self, vis_states: Tensor, U: Tensor) -> Tensor:
        """
        Visual token cleaning: project to evidence subspace v <- UU^T v
        Args:
            vis_states: [n_v, d] visual token hidden states
            U: [d, r] evidence subspace
        Returns:
            vis_cleaned: [n_v, d] cleaned visual states
        """
        if U.numel() == 0 or vis_states.numel() == 0:
            return vis_states
        return (vis_states @ U) @ U.T
