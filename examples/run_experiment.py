"""Full experiment: the positive result (constructed net) and the honest-negative
control (ordinary net), side by side.

    python examples/run_experiment.py
"""
import json
import torch
from verifier_probe import Attacker, ABCrownVerifier
from verifier_probe.models import build_fp_adversarial_net, build_natural_net


def run_case(name, builder):
    model, X, meta = builder
    verifier = ABCrownVerifier(config={"dtype": "float32", "method": "CROWN",
                                        "attack_budget": 6000, "attack_seed": 0})
    cert = verifier.certify(model, X)
    adv_x, logs = Attacker(model, verifier, verifier.config).run(X)
    print(f"\n==================== {name} ====================")
    print(f"meta: {meta}")
    print(f"certify: {cert.verdict}  margin_lb={cert.margin_lb:+.6e}")
    print(f"real float32 margin at adv: {logs['model_real_output']['margin_float32']:+.6e}")
    print(f"real float64 margin at adv: {logs['model_real_output']['margin_float64']:+.6e}")
    print(f"SOUNDNESS VIOLATION: {logs['soundness_violation']}")
    print(f"-> {logs['explanation']}")
    return logs["soundness_violation"]


def main():
    pos = run_case("CONSTRUCTED (expect violation)", build_fp_adversarial_net(seed=0))
    neg = run_case("NATURAL (expect NO violation)", build_natural_net(seed=1))
    print("\n======================= SUMMARY =======================")
    print(f"constructed net  -> soundness violation: {pos}  (expected True)")
    print(f"natural net      -> soundness violation: {neg}  (expected False)")
    print("\nInterpretation: the float32 CROWN certificate is unsound precisely when\n"
          "the network is engineered so the decision margin lives inside the\n"
          "per-logit rounding noise. On an ordinary net the relaxation\n"
          "over-approximation dwarfs float32 error, so the certificate holds and\n"
          "the attack honestly fails (closest near-miss reported).")


if __name__ == "__main__":
    main()
