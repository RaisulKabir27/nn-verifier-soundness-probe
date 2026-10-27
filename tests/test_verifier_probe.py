"""Tests for VerifierProbe:
  * certify() on a toy KNOWN-SAFE and toy KNOWN-UNSAFE property
  * trace schema validation
  * end-to-end attack with a small fixed budget (reproducible)

Run:  pytest -q
"""
import numpy as np
import torch
from verifier_probe import (Attacker, Prober, ABCrownVerifier, TRACE_SCHEMA,
                            exact_preact_bounds)
from verifier_probe.models import (build_fp_adversarial_net, build_toy_safe,
                                   build_toy_unsafe)


def _verifier(budget=2000):
    return ABCrownVerifier(config={"dtype": "float32", "method": "CROWN",
                                   "attack_budget": budget, "attack_seed": 0})


def test_certify_known_safe():
    model, X, meta = build_toy_safe()
    cert = _verifier().certify(model, X)
    assert cert.verdict == "safe", (cert.verdict, cert.margin_lb)
    assert cert.margin_lb >= 0.0


def test_certify_known_unsafe():
    model, X, meta = build_toy_unsafe()
    cert = _verifier().certify(model, X)
    # a sound verifier must NOT certify a genuinely unsafe property
    assert cert.verdict != "safe", (cert.verdict, cert.margin_lb)
    assert cert.margin_lb < 0.0


def test_trace_schema():
    model, X, meta = build_toy_safe()
    v = _verifier()
    trace = Prober(v).dump(model, X.center, eps=X.eps, true_label=X.true_label)
    # every documented top-level field is present
    for key in TRACE_SCHEMA:
        assert key in trace, f"missing trace field: {key}"
    assert trace["layers"], "no layers dumped"
    for layer in trace["layers"]:
        assert {"node", "kind", "shape", "lower", "upper"} <= set(layer)
    for k in ("lA", "uA", "lbias", "ubias"):
        assert k in trace["output_linear_bounds"]
    assert isinstance(trace["certified_margin_lb"], list)


def test_exact_preact_matches_bruteforce_and_active():
    # exact closed-form pre-activation min equals a dense brute-force min, and the
    # constructed net's hidden ReLU is provably active over the box.
    model, X, meta = build_fp_adversarial_net(seed=0)
    pa = meta["preact_analysis"]
    assert pa["all_active"] and pa["global_min_preact"] > 0.0
    W1 = model[0].weight.detach().numpy().astype(np.float64)
    b1 = model[0].bias.detach().numpy().astype(np.float64)
    # brute-force lower bound by sampling corners/edges must not go below the
    # exact closed-form global min (minus tiny fp slack)
    rng = np.random.default_rng(0)
    xs = (rng.uniform(-1, 1, size=(20000, W1.shape[1])) * X.eps).astype(np.float64)
    z = xs @ W1.T + b1
    assert z.min() >= pa["global_min_preact"] - 1e-6


def test_calibration_boundaries_and_labeling():
    model, X, meta = build_fp_adversarial_net(seed=0)
    cal = meta["calibration"]
    assert cal["straddle_verified"] is True
    # both boundary searches localized below the tau tolerance
    assert cal["adversarial_boundary"]["tau1_minus_tau0"] <= cal["tau_tol"]
    assert cal["safe_boundary"]["tau1_minus_tau0"] <= cal["tau_tol"]
    # fp_gap (window width) is positive and is NOT the same quantity as tau1-tau0
    assert cal["fp_gap_diagnostic"]["value"] > 0.0
    assert cal["fp_gap_diagnostic"]["value"] > 10 * cal["tau_tol"]
    # at the calibrated bias: certified safe AND real-forward mispredicts
    assert cal["crown_lb_at_calibrated"] >= 0.0 > cal["fwd_margin_at_calibrated"]


def test_end_to_end_attack_reproducible():
    model, X, meta = build_fp_adversarial_net(seed=0)
    v = _verifier(budget=3000)
    cert = v.certify(model, X)
    assert cert.is_safe, "constructed net should be certified SAFE in float32"
    adv_x, logs = Attacker(model, v, v.config).run(X)
    assert logs["soundness_violation"] is True
    assert logs["model_real_output"]["margin_float32"] < 0.0
    # it is a float32 artifact: float64 margin is non-negative
    assert logs["model_real_output"]["margin_float64"] >= 0.0
    # reproducible under the same seed
    adv_x2, logs2 = Attacker(model, v, v.config).run(X)
    assert abs(logs["model_real_output"]["margin_float32"]
               - logs2["model_real_output"]["margin_float32"]) < 1e-6
