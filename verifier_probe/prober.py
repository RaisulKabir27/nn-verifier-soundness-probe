
from typing import Any, Dict
import numpy as np
import torch

from auto_LiRPA import BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from .verifier import build_margin_C


# ---- documented schema: every field, its meaning and shape --------------------
TRACE_SCHEMA = {
    "schema_version": "string; version of this trace format.",
    "verifier": "string; which verifier/engine produced the trace.",
    "method": "string; bound method used (CROWN / CROWN-Optimized).",
    "dtype": "string; float32 or float64 the bounds were computed in.",
    "input": {
        "shape": "list[int]; shape of the introspected input x.",
        "values": "nested list; the concrete input x (float).",
        "eps": "float; L-inf radius of the box used for bound propagation.",
        "true_label": "int; class whose margin is certified.",
    },
    "layers": ("ordered list of per-node intermediate bounds. Each item: "
               "{node: str id, kind: op type, shape: list[int], "
               "lower: nested list (elementwise pre-activation lower bound), "
               "upper: nested list (elementwise pre-activation upper bound)}. "
               "For a ReLU node, lower<0<upper marks an UNSTABLE neuron (the "
               "source of relaxation). Bounds are in the network's activation "
               "units."),
    "output_linear_bounds": (
        "the output's linear relaxation in the INPUT, from CROWN's backward "
        "pass, one row per competitor-class margin spec. Fields: "
        "lA [n_spec, n_in] lower coefficient matrix, uA [n_spec, n_in] upper "
        "coefficient matrix, lbias [n_spec] lower bias, ubias [n_spec] upper "
        "bias. The certified margin lower bound is "
        "lbias + sum_i (lA_i>0 ? lA_i*x_min_i : lA_i*x_max_i)."),
    "certified_margin_lb": "list[float]; per-spec CROWN lower bound (the verdict basis).",
    "alpha_slopes": ("per-ReLU optimizable lower-relaxation slopes alpha in [0,1] "
                     "(present only for method=CROWN-Optimized; {} otherwise). "
                     "{relu_node: {start_node: shape}}."),
    "notes": "free-form provenance / instrumentation notes.",
}

# auto_LiRPA op types that carry activation bounds worth dumping (skip BoundParams)
_ACTIVATION_KINDS = ("BoundInput", "BoundLinear", "BoundConv", "BoundRelu",
                     "BoundAdd", "BoundFlatten", "BoundReshape", "BoundBatchNorm")


def _to_list(t, max_elems=20000):
    if t is None:
        return None
    t = t.detach().cpu()
    if t.numel() > max_elems:            # keep big tensors JSON-sane
        flat = t.reshape(-1)
        return {"_truncated": True, "numel": int(t.numel()),
                "min": float(flat.min()), "max": float(flat.max()),
                "head": flat[:64].tolist()}
    return t.tolist()


class Prober:
    """Dumps per-layer bounds, output linear coefficients and alpha slopes."""
    def __init__(self, verifier):
        self.verifier = verifier

    def _nodes(self, bm):
        try:
            return list(bm.nodes())
        except Exception:
            return list(bm._modules.values())

    def dump(self, model, x, eps: float = None, true_label: int = 0) -> Dict[str, Any]:
        """Introspect the verifier on a single input x. `eps` defaults to a tiny
        box so intermediate bounds are well-defined around x."""
        from .verifier import _dtype
        cfg = self.verifier.config
        dt = _dtype(cfg)
        torch.set_default_dtype(dt)
        model = model.to(dt)
        x = x.to(dt)
        eps = cfg.get("probe_eps", 1e-4) if eps is None else eps
        with torch.no_grad():
            n_class = int(model(x).shape[-1])
        C = build_margin_C(true_label, n_class, dtype=dt)

        bm = self.verifier.build_module(model, _Region(x, eps, true_label), dtype=dt)[0]
        bx = BoundedTensor(x, PerturbationLpNorm(norm=np.inf, eps=eps))

        # (a) per-layer intermediate pre-activation bounds (no C: fills all nodes)
        bm.compute_bounds(x=(bx,), method="CROWN")
        layers = []
        for n in self._nodes(bm):
            kind = type(n).__name__
            lo = getattr(n, "lower", None); up = getattr(n, "upper", None)
            if kind == "BoundParams" or (lo is None and up is None):
                continue
            if kind not in _ACTIVATION_KINDS:
                # still dump if it carries bounds, but tag it
                pass
            ref = lo if lo is not None else up
            layers.append({
                "node": getattr(n, "name", "?"),
                "kind": kind,
                "shape": list(ref.shape),
                "lower": _to_list(lo),
                "upper": _to_list(up),
            })

        # (b) output linear relaxation coefficients + certified margin lb (with C)
        lb, _, A = bm.compute_bounds(x=(bx,), method=cfg["method"], C=C,
                                     return_A=True,
                                     needed_A_dict={bm.output_name[0]: [bm.input_name[0]]})
        ak = A[bm.output_name[0]][bm.input_name[0]]
        output_linear = {k: _to_list(ak[k]) for k in ("lA", "uA", "lbias", "ubias")}

        # (c) alpha slopes if alpha-CROWN was used
        alpha = {}
        for n in self._nodes(bm):
            if type(n).__name__ == "BoundRelu" and isinstance(getattr(n, "alpha", None), dict):
                alpha[getattr(n, "name", "?")] = {k: list(v.shape) for k, v in n.alpha.items()}

        return {
            "schema_version": "1.0",
            "verifier": "alpha-beta-CROWN (auto_LiRPA CROWN core)",
            "method": cfg["method"],
            "dtype": str(dt).split(".")[-1],
            "input": {"shape": list(x.shape), "values": _to_list(x),
                      "eps": eps, "true_label": true_label},
            "layers": layers,
            "output_linear_bounds": output_linear,
            "certified_margin_lb": [float(v) for v in lb.reshape(-1).tolist()],
            "alpha_slopes": alpha,
            "notes": ("Read-only via compute_bounds / return_A / node.lower-upper; "
                      "no auto_LiRPA internals mutated."),
        }


class _Region:
    """Minimal region shim for build_module (center+eps)."""
    def __init__(self, center, eps, true_label):
        self.center = center; self.eps = eps; self.true_label = true_label
