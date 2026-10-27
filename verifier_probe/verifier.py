from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time
import numpy as np
import torch

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


DEFAULT_CONFIG = {
    "seed": 0,
    "device": "cpu",
    "dtype": "float32",          # precision the verifier runs in ("float32"/"float64")
    "method": "CROWN",           # "CROWN" or "CROWN-Optimized" (alpha-CROWN)
    "conv_mode": "matrix",
    "safe_tol": 0.0,             # verdict is SAFE iff margin_lb >= safe_tol
    # --- attacker knobs (read by Attacker) ---
    "attack_budget": 4000,       # hill-climb forward evaluations
    "attack_batch": 128,
    "attack_seed": 0,
}


def _dtype(cfg) -> torch.dtype:
    return torch.float64 if str(cfg.get("dtype", "float32")) == "float64" else torch.float32


def build_margin_C(true_label: int, n_class: int, dtype=torch.float32) -> torch.Tensor:
    """C with one row per competitor class j: row = e_true - e_j.

    CROWN's lower bound on (C y) is then a lower bound on every margin
    logit_true - logit_j; the property holds iff all rows are >= 0.
    Shape [1, n_class-1, n_class].
    """
    rows = [[1.0 if k == true_label else (-1.0 if k == j else 0.0) for k in range(n_class)]
            for j in range(n_class) if j != true_label]
    return torch.tensor([rows], dtype=dtype)


@dataclass
class Certificate:
    """Result of certify(). `verdict` is the only thing a user must trust."""
    verdict: str                         # "safe" | "unsafe-or-unknown"
    margin_lb: float                     # min over competitor classes of the CROWN lower bound
    per_spec_lb: List[float]             # CROWN lower bound per competitor class
    method: str
    dtype: str
    n_class: int
    true_label: int
    elapsed_s: float
    # read-only handles for the Prober (not part of the trusted verdict):
    bounded_module: Optional[Any] = field(default=None, repr=False)
    C: Optional[Any] = field(default=None, repr=False)

    @property
    def is_safe(self) -> bool:
        """True if the verdict is "safe"."""
        return self.verdict == "safe"


class ABCrownVerifier:
    """Thin adapter: certify(model, X) -> Certificate, plus internal-state access."""

    def __init__(self, config: Optional[dict] = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.device = self.config["device"]

    # -- internal: build a BoundedModule for (model, region) at the configured dtype --
    def build_module(self, model, X, dtype=None):
        """Wrap model in an auto_LiRPA BoundedModule for region X; returns (module, bounded_input)."""
        dt = dtype or _dtype(self.config)
        torch.set_default_dtype(dt)
        model = model.to(dt)
        center = X.center.to(dt)
        bm = BoundedModule(model, center,
                           bound_opts={"conv_mode": self.config["conv_mode"]},
                           device=self.device)
        bm.eval()
        ptb = PerturbationLpNorm(norm=np.inf, eps=X.eps)
        bx = BoundedTensor(center, ptb)
        return bm, bx

    def _n_class(self, model, X, dt) -> int:
        with torch.no_grad():
            return int(model.to(dt)(X.center.to(dt)).shape[-1])

    def certify(self, model, X) -> Certificate:
        """Run CROWN (or alpha-CROWN) and return the robustness certificate."""
        dt = _dtype(self.config)
        t0 = time.time()
        n_class = self._n_class(model, X, dt)
        C = build_margin_C(X.true_label, n_class, dtype=dt)
        bm, bx = self.build_module(model, X, dtype=dt)
        lb, _ = bm.compute_bounds(x=(bx,), method=self.config["method"], C=C)
        per_spec = [float(v) for v in lb.reshape(-1).tolist()]
        margin_lb = min(per_spec) if per_spec else float("nan")
        verdict = "safe" if margin_lb >= self.config["safe_tol"] else "unsafe-or-unknown"
        return Certificate(verdict=verdict, margin_lb=margin_lb, per_spec_lb=per_spec,
                           method=self.config["method"], dtype=str(dt).split(".")[-1],
                           n_class=n_class, true_label=X.true_label,
                           elapsed_s=time.time() - t0, bounded_module=bm, C=C)


def compute_crown_margin_lb(model, center, eps, true_label=0, dtype=torch.float32,
                            method="CROWN", conv_mode="matrix", device="cpu") -> float:
    """Standalone helper (used for model calibration): CROWN lower bound on the
    worst-case margin over the L-inf box. Returns a single float."""
    torch.set_default_dtype(dtype)
    model = model.to(dtype)
    center = center.to(dtype)
    with torch.no_grad():
        n_class = int(model(center).shape[-1])
    C = build_margin_C(true_label, n_class, dtype=dtype)
    bm = BoundedModule(model, center, bound_opts={"conv_mode": conv_mode}, device=device)
    bm.eval()
    bx = BoundedTensor(center, PerturbationLpNorm(norm=np.inf, eps=eps))
    lb, _ = bm.compute_bounds(x=(bx,), method=method, C=C)
    return float(lb.min())
