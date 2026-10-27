# VerifierProbe

Adversarial probing + intermediate-state dumping for the **α,β-CROWN** DNN
verifier (its `auto_LiRPA` CROWN bound-propagation core).

**Report:** see [`VerifierProbe_Report.pdf`](VerifierProbe_Report.pdf) for the full write-up (method, results, references).

It does two things against the verifier:

1. **Attack** — given `(model, verifier, X)` the verifier certifies SAFE, find a
   concrete input in `X` that the model **mispredicts under real float32
   execution** (a floating-point soundness violation), grounded in the Jia &
   Rinard mechanism.
2. **Introspect** — dump the verifier's internal computation on one input
   (per-layer bounds, the output linear-relaxation coefficients, α slopes) into a
   documented, JSON-serializable trace.

```python
from verifier_probe import Attacker, Prober, ABCrownVerifier
from verifier_probe.models import build_fp_adversarial_net

model, X, meta = build_fp_adversarial_net(seed=0)
verifier = ABCrownVerifier(config={"dtype": "float32", "method": "CROWN"})

cert = verifier.certify(model, X)          # -> verdict "safe"
atk = Attacker(model, verifier, verifier.config)
adv_x, atk_logs = atk.run(X)               # -> input the model actually mispredicts
prober = Prober(verifier)
trace = prober.dump(model, adv_x)          # -> documented internal-state trace
```

---

## Install & run

This project was validated with Python 3.11.9 and the following pinned
dependencies:

- `torch==2.11.0`
- `numpy==2.4.6`
- `appdirs==1.4.4`
- `tqdm==4.70.1`
- `packaging==26.3`
- `graphviz==0.21`
- `pytest==9.1.1`
- `auto_LiRPA==0.8.0`

`auto_LiRPA` is used from the pinned α,β-CROWN checkout rather than from
PyPI. Set `AUTO_LIRPA_PATH` to that checkout's `auto_LiRPA` directory.

```bash
# 0. Use Python 3.11
python --version

# 1. Set the path to the pinned α,β-CROWN checkout
export AUTO_LIRPA_PATH=/path/to/alpha-beta-CROWN/auto_LiRPA

# 2. Install this package and its pinned dependencies
pip install -e ".[test]"

# 3. Install the local auto_LiRPA checkout
pip install -e "$AUTO_LIRPA_PATH"

# 4. End-to-end quickstart
python -m examples.quickstart

# 5. Full experiment: constructed + natural control
python -m examples.run_experiment

# 6. Tests
pytest -q
```

`build_fp_adversarial_net` uses `big=12500` by default. Whether a given `big`
produces the float32 straddle depends on the machine's float32 math library, so
if 12500 does not straddle it tries the values in `big_fallbacks` in order; the
value used is reported in `meta["big"]` and every attempt in `meta["big_tried"]`.

### What you should see
- `quickstart.py`: `certify -> safe`, then `SOUNDNESS VIOLATION FOUND: True` with
  float32 margin `< 0` (class flips) but float64 margin `>= 0`; writes
  `quickstart_trace.json`.
- `run_experiment.py`: constructed net → violation **True**; natural net (CROWN
  certifies SAFE) → violation **False** (closest near-miss reported).
- `pytest`: 6 passed.

---

## Method write-up

### Which verifier, and why
**α,β-CROWN** — the repeated VNN-COMP winner, Python/PyTorch, and introspectable
through its `auto_LiRPA` engine. Its complete verifier is CROWN linear-relaxation
bound propagation + α (optimizable ReLU slopes) + β (branch-and-bound). We target
the **CROWN bound-propagation core**, which is exactly the engine α,β-CROWN
evaluates at every branch-and-bound node; a numerical soundness gap in this bound-propagation computation can propagate
into the complete verification procedure (BaB tightens bounds but does not
change the underlying float32 arithmetic model).

### Sound vs. complete (for α,β-CROWN)
- **Sound**: if it certifies "no adversarial example in `X`", there is none —
  *in real arithmetic*. CROWN's bound is a provable lower bound on the true
  worst-case margin.
- **Complete**: with branch-and-bound it will (given time) resolve every case, so
  it does not answer "unknown" indefinitely on decidable ReLU properties.
- The soundness guarantee assumes **real-valued arithmetic**. The implementation
  runs in IEEE-754 **float32** by default (`double_fp: false` in the real tool),
  and that is the gap we exploit.

### What we instrument (and why semantics are unchanged)
We do **not** patch α,β-CROWN's math. The adapter uses three public, read-only
`auto_LiRPA` entry points:
- `BoundedModule.compute_bounds(method=..., C=...)` — the exact bound the verifier
  computes; we only read the returned bounds (this is `certify()`).
- `compute_bounds(..., return_A=True, needed_A_dict=...)` — the output→input
  linear relaxation coefficients `lA, uA, lbias, ubias`.
- `node.lower` / `node.upper` after a pass — the per-layer pre-activation
  intervals the verifier already stores.

None of these alter weights, relaxations, split decisions, or the verdict, so the
certificate we read is the one the verifier produces unprobed. (Deeper state such
as per-node β dual variables *mid*-BaB would require patching the BaB loop; we
note where, but the float32 soundness finding does not need it.)

### The numerical gap we exploit
CROWN forms the margin's input coefficients by **backward association**
(`C^T W_out` first, i.e. subtracting the two large output rows, then times the
earlier layers); the forward pass forms the large logits and subtracts them
**last**. In float32 these associations disagree by ~`ulp(logit)`. We make that
matter by constructing a network whose two logits share a large common component
(~10⁷) and differ only by a tiny vector, so the **decision margin lives inside
the per-logit rounding noise**. One hidden layer with a large positive bias keeps
the ReLU active across the whole box, so the network is **exactly affine** there
and CROWN's relaxation is *exact in real arithmetic* — leaving float32 as the only
possible source of disagreement. The output bias is calibrated (quasi-safe /
quasi-adversarial search) so CROWN(float32) ≥ 0 (SAFE) while the float32 forward
margin at the box corner is < 0.

### Pre-ReLU analysis and calibration (`calibration.py`)
Two steps run before the attack, in `build_fp_adversarial_net`:
- **Exact pre-ReLU minimum.** For the hidden layer `z_i(x)=W_i·x+b_i` over the
  box, the exact minimum is the closed form `b_i + W_i·c − eps·‖W_i‖₁` (a linear
  form is minimized at a box corner). We compute it per neuron and require the
  **global** minimum `> 0` — a *proof* (not sampling) that every ReLU is active,
  so the net is exactly affine on the box.
- **Boundary-search calibration (loop).** Bisection localizes two boundaries in
  output-bias space: the *quasi-adversarial* boundary where the float32 forward
  margin at the corner crosses 0, and the *safe* boundary where the float32 CROWN
  bound crosses 0. Each is driven to a bracket `τ₁−τ₀ < 1e-7` (our numerical boundary-localization tolerance).
  The **straddle-window width** `fp_gap = adv_boundary − safe_boundary`
  is a *diagnostic* — explicitly **not** `τ₁−τ₀` (the earlier one-shot wrongly
  used `lb0−fwd0` for this; the loop computes it correctly because the float32
  forward margin is quantized, so `fwd(0)` is not the boundary). The calibrated
  bias is the window midpoint, which maximizes both margins so the violation is
  robust.

### Attack strategy (grounded, not blind)
Start at CROWN's minimizing corner `x* = center - eps·sign(∂margin/∂x)` — the exact
worst case for the affine net, and in general where the certified margin is
thinnest — then a small, seeded, single-input float32 hill-climb. Every accept
decision and the final verdict use a **single-input** forward pass (batched-GEMM
rounding differs by several ulp and would not reflect real deployment execution).

### Honest result
- **Constructed net**: genuine float32 soundness violation. CROWN certifies the
  margin ≥ 0 while a concrete in-box input mispredicts in float32 (the argmax
  flips). A float64 re-evaluation of the same input is ≥ 0, confirming it is a
  pure floating-point artifact — the floating-point soundness mechanism described by Jia & Rinard
  , reproduced on the VNN-COMP-winning verifier's engine.
- **Natural net** (control): CROWN certifies SAFE with a comfortable margin, and
  the attack **cannot** break it — the relaxation over-approximation dwarfs the
  float32 error. `atk_logs` reports the closest near-miss. This is the expected,
  valid negative, not a failure.

---

## Package layout
```
verifier_probe/
  region.py     InputRegion (center + eps + true_label)
  verifier.py   ABCrownVerifier.certify(model, X) -> Certificate  (+ internal-state access)
  attacker.py   Attacker(model, verifier, config).run(X) -> (adv_x, atk_logs)
  prober.py     Prober(verifier).dump(model, x) -> trace   (TRACE_SCHEMA documents every field)
  calibration.py  exact pre-ReLU bounds + output-bias boundary search
  models.py     constructed / natural / toy model builders
examples/       quickstart.py, run_experiment.py
tests/          test_verifier_probe.py
```

### `trace` schema
`verifier_probe.TRACE_SCHEMA` documents every field (meaning + shape): ordered
per-layer pre-activation `lower`/`upper` bounds, the output `lA/uA/lbias/ubias`
linear relaxation (one row per competitor-class margin), `certified_margin_lb`,
and `alpha_slopes` (populated under `method="CROWN-Optimized"`).

### Config keys
`seed, device, dtype ("float32"/"float64"), method ("CROWN"/"CROWN-Optimized"),
safe_tol, attack_budget, attack_batch, attack_seed, probe_eps`.
