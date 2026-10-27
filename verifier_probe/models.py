from typing import Tuple, Sequence
import numpy as np
import torch
import torch.nn as nn

from .region import InputRegion
from .verifier import compute_crown_margin_lb
from .calibration import exact_preact_bounds, calibrate_output_bias


def _seq(n_in, width, out, W1, b1, Wout, bout) -> nn.Sequential:
    m = nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Linear(width, out))
    with torch.no_grad():
        m[0].weight.copy_(torch.as_tensor(W1, dtype=torch.float32)); m[0].bias.copy_(torch.as_tensor(b1, dtype=torch.float32))
        m[2].weight.copy_(torch.as_tensor(Wout, dtype=torch.float32)); m[2].bias.copy_(torch.as_tensor(bout, dtype=torch.float32))
    return m.eval()


def build_fp_adversarial_net(seed: int = 0, n_in: int = 50, width: int = 64,
                             eps: float = 0.05, big: float = 12500.0,
                             hbias: float = 10.0,
                             big_fallbacks: Sequence[float] = (5000.0, 17500.0, 10000.0,
                                                               7500.0, 15000.0, 20000.0),
                             ) -> Tuple[nn.Sequential, InputRegion, dict]:

    """Build the calibrated float32-unsound instance.

    Proves the hidden ReLU is always active, then calibrates the output bias so
    float32 CROWN certifies SAFE while the float32 forward margin at the corner
    is negative. Tries `big` first, then `big_fallbacks`. Returns (model, X, meta).
    """

    rng = np.random.default_rng(seed)
    W1 = (rng.standard_normal((width, n_in)) / np.sqrt(n_in)).astype(np.float32)
    b1 = np.full(width, hbias, np.float32)
    a = (rng.standard_normal(width) * 50).astype(np.float32)
    d = (rng.standard_normal(width) / np.sqrt(width)).astype(np.float32)
    center = torch.zeros(1, n_in)
    torch.set_default_dtype(torch.float32)

    # --- 1. EXACT pre-ReLU minimum analysis (proof the net is affine on the box) -
    #     (depends only on the hidden layer, so it does not change with `big`)
    preact = exact_preact_bounds(W1, b1, center, eps)
    assert preact["all_active"], (
        f"hidden ReLU NOT provably active: global min pre-activation "
        f"{preact['global_min_preact']:.4e} at neuron {preact['global_min_neuron']} "
        f"<= 0. Raise `hbias` or shrink `eps`.")

    # --- 2. proper calibration: boundary-search in output-bias space ------------
    def calibrate_for(big_v):
        Wout = np.stack([big_v * a + d / 2, big_v * a - d / 2]).astype(np.float32)

        def crown_lb_of(ob):     # float32 CROWN lower bound on the margin at bias ob
            return compute_crown_margin_lb(_seq(n_in, width, 2, W1, b1, Wout, [ob, 0.0]),
                                           center, eps, true_label=0, dtype=torch.float32)

        def fwd_of(ob):          # float32 forward margin at the CROWN corner at bias ob
            torch.set_default_dtype(torch.float32)
            m = _seq(n_in, width, 2, W1, b1, Wout, [ob, 0.0])
            x = center.clone().requires_grad_(True)
            y = m(x); (y[:, 0] - y[:, 1]).sum().backward()
            corner = (center - eps * torch.sign(x.grad.detach())).detach()
            with torch.no_grad():
                yc = m(corner.reshape(1, -1))                 # single-input (authoritative)
                return float((yc[0, 0] - yc[0, 1]).item())

        return Wout, calibrate_output_bias(crown_lb_of, fwd_of, tau_tol=1e-7)

    big_tried = []
    for big_v in [big] + [b for b in big_fallbacks if b != big]:
        Wout, calib = calibrate_for(big_v)
        big_tried.append({"big": big_v,
                          "fp_gap": calib["fp_gap_diagnostic"]["value"],
                          "straddle_verified": calib["straddle_verified"]})
        if calib["straddle_verified"]:
            break
    assert calib["straddle_verified"], (
        f"calibration failed to find a float32 straddle for every `big` tried: "
        f"{big_tried}. Try other `big` values.")
    out_bias = calib["calibrated_out_bias"]

    # --- 3. build the calibrated model (handed to the Attacker next) ------------
    model = _seq(n_in, width, 2, W1, b1, Wout, [out_bias, 0.0])
    X = InputRegion(center=center, eps=eps, true_label=0)
    meta = {"seed": seed, "n_in": n_in, "width": width, "eps": eps, "big": big_v,
            "hbias": hbias, "calibrated_out_bias": out_bias,
            "preact_analysis": preact, "calibration": calib,
            "big_tried": big_tried}
    return model, X, meta


def build_natural_net(seed: int = 1, n_in: int = 50, width: int = 64,
                      eps: float = 0.02) -> Tuple[nn.Sequential, InputRegion, dict]:
    """An ordinary small net: modest weights, small biases, no cancelling logits.
    Used as the honest-negative control -- CROWN's relaxation over-approximation
    is far larger than any float32 error, so the attack should NOT find a
    soundness violation."""
    rng = np.random.default_rng(seed)
    W1 = (rng.standard_normal((width, n_in)) / np.sqrt(n_in)).astype(np.float32)
    b1 = (rng.standard_normal(width) * 0.1 + 0.5).astype(np.float32)
    Wout = (rng.standard_normal((2, width)) / np.sqrt(width)).astype(np.float32)
    # give class 0 a comfortable margin so CROWN genuinely certifies SAFE; the
    # point of this control is that even a real SAFE certificate resists the
    # float32 attack (relaxation margin >> rounding error).
    bout = np.array([2.0, 0.0], np.float32)
    model = _seq(n_in, width, 2, W1, b1, Wout, bout)
    X = InputRegion(center=torch.zeros(1, n_in), eps=eps, true_label=0)
    return model, X, {"seed": seed, "n_in": n_in, "width": width, "eps": eps}


def build_toy_safe() -> Tuple[nn.Sequential, InputRegion, dict]:
    """2-in, 2-out net robust on a small box: class 0 wins by a wide margin."""
    W1 = np.eye(2, dtype=np.float32)
    b1 = np.array([1.0, 1.0], np.float32)       # ReLU active
    Wout = np.array([[1.0, 0.0], [0.0, 1.0]], np.float32)
    bout = np.array([5.0, 0.0], np.float32)     # logit0 >> logit1 everywhere
    model = _seq(2, 2, 2, W1, b1, Wout, bout)
    X = InputRegion(center=torch.zeros(1, 2), eps=0.1, true_label=0)
    return model, X, {"expected": "safe"}


def build_toy_unsafe() -> Tuple[nn.Sequential, InputRegion, dict]:
    """Same shape, but a real adversarial example exists in the box: the input
    can drive logit1 above logit0, so the property is genuinely false."""
    W1 = np.eye(2, dtype=np.float32)
    b1 = np.array([1.0, 1.0], np.float32)
    Wout = np.array([[1.0, 0.0], [0.0, 5.0]], np.float32)   # class 1 very input-sensitive
    bout = np.array([0.2, 0.0], np.float32)
    model = _seq(2, 2, 2, W1, b1, Wout, bout)
    X = InputRegion(center=torch.zeros(1, 2), eps=1.0, true_label=0)
    return model, X, {"expected": "unsafe-or-unknown"}
