from typing import Tuple, Dict, Any
import numpy as np
import torch


class Attacker:
    """Searches an InputRegion for a float32 soundness violation."""
    def __init__(self, model, verifier, config: dict = None):
        self.model = model
        self.verifier = verifier
        self.config = {**verifier.config, **(config or {})}

    def _margin_fn(self, true_label):
        def margin(x):  # x: [B, *shape] -> [B]  margin = logit_true - max_{j!=true}
            y = self.model(x)
            tl = y[:, true_label]
            other = y.clone(); other[:, true_label] = -float("inf")
            return tl - other.max(dim=1).values
        return margin

    def _margin_single(self, x1, true_label):
        # Authoritative "real execution": evaluate ONE input with a single-input
        # forward pass. Batched-GEMM rounding differs from single-input rounding
        # by several ulp at large logit scale, so every accept decision and the
        # final violation check must use this, not the batched probe value.
        with torch.no_grad():
            y = self.model(x1.reshape(1, *x1.shape[1:]))
            tl = float(y[0, true_label])
            oth = y.clone(); oth[0, true_label] = -float("inf")
            return tl - float(oth.max())

    def run(self, X) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Certify X, start at the CROWN minimizing corner, then hill-climb in float32.
        Returns (adv_x, atk_logs). adv_x is the violating input, or the closest
        near-miss if no violation is found within the budget.
        """
        cfg = self.config
        torch.set_default_dtype(torch.float32)
        model32 = self.model.to(torch.float32)
        center = X.center.to(torch.float32)
        eps = X.eps
        budget = int(cfg["attack_budget"])
        batch = int(cfg["attack_batch"])
        gen = torch.Generator().manual_seed(int(cfg["attack_seed"]))
        margin = self._margin_fn(X.true_label)

        # 1. certificate
        cert = self.verifier.certify(model32, X)

        # 2. grounded start: CROWN minimizing corner
        x = center.clone().requires_grad_(True)
        margin(x).sum().backward()
        corner = X.clamp((center - eps * torch.sign(x.grad.detach())).detach())

        with torch.no_grad():
            best_m = self._margin_single(corner, X.true_label)   # single-input authoritative
            best_x = corner.clone()
            traj = [("corner", best_m)]
            # 3. hill-climb: shrinking-radius random probes around the best point.
            #    Batched probe PROPOSES; a single-input forward DECIDES (so the
            #    accepted margin is a real single-input value, not a GEMM artifact).
            evals = batch
            step = 0
            while evals < budget:
                frac = max(0.05, 1.0 - evals / budget)          # anneal radius
                noise = (torch.rand(batch, *center.shape[1:], generator=gen) * 2 - 1) * eps * frac
                cand = X.clamp(best_x + noise)
                mv = margin(cand)                                # batched proposal
                j = int(torch.argmin(mv))
                cand_j = cand[j:j+1]
                m_single = self._margin_single(cand_j, X.true_label)  # authoritative
                if m_single < best_m:
                    best_m = m_single; best_x = cand_j.clone()
                    traj.append((f"step{step}", best_m))
                evals += batch; step += 1

        # 4. float64 control at the found point
        torch.set_default_dtype(torch.float64)
        m64 = self.model.to(torch.float64)
        with torch.no_grad():
            y64 = m64(best_x.double())
            tl = y64[:, X.true_label]
            oth = y64.clone(); oth[:, X.true_label] = -float("inf")
            margin_f64 = float((tl - oth.max(dim=1).values).item())
        torch.set_default_dtype(torch.float32)
        self.model.to(torch.float32)

        with torch.no_grad():
            adv_logits = model32(best_x).reshape(-1).tolist()
        argmax32 = int(np.argmax(adv_logits))
        genuine_misprediction = (argmax32 != X.true_label)   # class actually flips
        violation = bool(cert.is_safe and best_m < 0.0 and genuine_misprediction)

        atk_logs = {
            "verifier_claim": {
                "verdict": cert.verdict,
                "margin_lb": cert.margin_lb,
                "per_spec_lb": cert.per_spec_lb,
                "method": cert.method,
                "dtype": cert.dtype,
            },
            "model_real_output": {
                "logits_float32": adv_logits,
                "margin_float32": best_m,          # single-input authoritative
                "margin_float64": margin_f64,
                "argmax_float32": argmax32,
                "genuine_misprediction": genuine_misprediction,
            },
            "search": {
                "strategy": "crown-corner + annealed float32 hill-climb",
                "budget_evals": budget,
                "trajectory": traj,          # (label, best real margin) over the search
            },
            "soundness_violation": violation,
            "explanation": (
                "verifier certified SAFE (margin_lb>=0) but the float32 forward "
                "margin at adv_x is <0 => the float32 certificate is unsound"
                if violation else
                "no float32 soundness violation within budget; best real margin "
                f"= {best_m:+.3e} (closest near-miss returned as adv_x)"
            ),
            "float64_control": (
                "violation vanishes in float64 (margin_float64>=0) => pure "
                "floating-point gap, not a real counterexample"
                if violation and margin_f64 >= 0 else
                "n/a"
            ),
        }
        return best_x, atk_logs
