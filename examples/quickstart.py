"""Quickstart: both capabilities on one constructed model + alpha-beta-CROWN.

    python -m examples.quickstart        (from the project root)
"""
import json
import torch
from verifier_probe import Attacker, Prober, ABCrownVerifier
from verifier_probe.models import build_fp_adversarial_net


def main():
    # 1. a small model on which CROWN's float32 certificate is unsound.
    #    build_fp_adversarial_net runs, in order:
    #      (a) EXACT pre-ReLU minimum analysis  -> proves the net is affine on X
    #      (b) boundary-search calibration loop -> picks the output bias
    model, X, meta = build_fp_adversarial_net(seed=0)
    pa = meta["preact_analysis"]; cal = meta["calibration"]
    print("(a) pre-ReLU analysis: global_min_preact "
          f"{pa['global_min_preact']:+.4f} at neuron {pa['global_min_neuron']} "
          f"-> all_active={pa['all_active']} (net is affine on X)")
    print(f"(b) calibration [{cal['status']}]: "
          f"adv_boundary tau1-tau0={cal['adversarial_boundary']['tau1_minus_tau0']:.2e} "
          f"(<{cal['tau_tol']:.0e}), safe_boundary tau1-tau0="
          f"{cal['safe_boundary']['tau1_minus_tau0']:.2e}")
    print(f"    fp_gap (straddle-window width, NOT tau1-tau0) = "
          f"{cal['fp_gap_diagnostic']['value']:+.4f}  -> out_bias={meta['calibrated_out_bias']:+.5f}")
    print(f"    at calibrated bias: CROWN_lb={cal['crown_lb_at_calibrated']:+.4e} (>=0), "
          f"fwd_margin={cal['fwd_margin_at_calibrated']:+.4e} (<0)")

    # 2. the verifier adapter (float32 == the precision it ships in by default)
    verifier = ABCrownVerifier(config={"dtype": "float32", "method": "CROWN",
                                        "attack_budget": 4000, "attack_seed": 0})

    # 3. certify: the verifier claims the box is SAFE
    cert = verifier.certify(model, X)
    print(f"\n[certify] verdict={cert.verdict}  margin_lb={cert.margin_lb:+.6e}  "
          f"({cert.method}, {cert.dtype})")

    # 4. attack: find an in-box input the model actually mispredicts
    atk = Attacker(model, verifier, verifier.config)
    adv_x, atk_logs = atk.run(X)
    print("\n[attack] logs:")
    print(json.dumps({k: v for k, v in atk_logs.items() if k != "search"}, indent=2))
    print("search trajectory (first/last):",
          atk_logs["search"]["trajectory"][0], atk_logs["search"]["trajectory"][-1])
    print(f"\nSOUNDNESS VIOLATION FOUND: {atk_logs['soundness_violation']}")

    # 5. introspect: dump the verifier's internal state on the adversarial input
    prober = Prober(verifier)
    trace = prober.dump(model, adv_x, eps=X.eps, true_label=X.true_label)
    print("\n[prober] trace keys:", list(trace.keys()))
    print("  layers dumped:", [(l["node"], l["kind"], l["shape"]) for l in trace["layers"]])
    print("  output_linear_bounds.lA shape:",
          _shape(trace["output_linear_bounds"]["lA"]))
    print("  certified_margin_lb:", trace["certified_margin_lb"])
    with open("quickstart_trace.json", "w") as f:
        json.dump(trace, f, indent=2)
    print("  full trace -> quickstart_trace.json")


def _shape(x):
    s = []
    while isinstance(x, list):
        s.append(len(x)); x = x[0] if x else None
    return s


if __name__ == "__main__":
    main()
