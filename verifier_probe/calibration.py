from typing import Callable, Dict, Any, Tuple
import numpy as np


# --------------------------------------------------------------------------- #
def exact_preact_bounds(weight, bias, center, eps: float) -> Dict[str, Any]:
    """Exact per-neuron pre-activation interval over the L-inf box, in float64.

    weight : [width, n_in]   bias : [width]   center : [1, n_in] or [n_in]
    Returns per-neuron lower/upper, the global min and which neuron attains it,
    and all_active = (global_min > 0).
    """
    W = np.asarray(weight, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64).reshape(-1)
    c = np.asarray(center, dtype=np.float64).reshape(-1)
    wc = W @ c                                  # [width]
    radius = eps * np.abs(W).sum(axis=1)        # [width]  eps * ||W_i||_1
    lower = b + wc - radius
    upper = b + wc + radius
    gmin_i = int(np.argmin(lower))
    gmin = float(lower[gmin_i])
    return {
        "method": "closed-form corner (exact for an affine layer over an L-inf box)",
        "per_neuron_lower": lower.tolist(),
        "per_neuron_upper": upper.tolist(),
        "global_min_preact": gmin,
        "global_min_neuron": gmin_i,
        "all_active": bool(gmin > 0.0),
        "eps": eps,
    }


# --------------------------------------------------------------------------- #
def _bisect_root(f: Callable[[float], float], lo: float, hi: float,
                 tol: float, max_iter: int) -> Dict[str, Any]:
    """Localize a root of an INCREASING f to a bracket of width < tol.

    Requires f(lo) < 0 <= f(hi); expands the bracket a few times if not.
    Returns {tau0, tau1 (bracket with f(tau0)<0<=f(tau1)), width, iters,
    f_tau0, f_tau1, converged}.
    """
    flo, fhi = f(lo), f(hi)
    tries = 0
    while not (flo < 0.0 <= fhi) and tries < 40:
        span = hi - lo
        lo -= span; hi += span
        flo, fhi = f(lo), f(hi)
        tries += 1
    if not (flo < 0.0 <= fhi):
        return {"tau0": lo, "tau1": hi, "width": hi - lo, "iters": 0,
                "f_tau0": flo, "f_tau1": fhi, "converged": False,
                "note": "could not bracket a sign change"}
    it = 0
    while (hi - lo) > tol and it < max_iter:
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm < 0.0:
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
        it += 1
    return {"tau0": lo, "tau1": hi, "width": hi - lo, "iters": it,
            "f_tau0": flo, "f_tau1": fhi, "converged": (hi - lo) <= tol}


def calibrate_output_bias(crown_lb_of: Callable[[float], float],
                          fwd_of: Callable[[float], float],
                          init_guess: float = 0.0,
                          tau_tol: float = 1e-7,
                          bracket_halfwidth: float = 1.0,
                          max_iter: int = 200) -> Dict[str, Any]:
    """Boundary-search both decision boundaries and return the calibrated bias.

    crown_lb_of(ob) : float32 CROWN lower bound on the margin, as a function of
                      the output bias ob (increasing, slope ~ +1).
    fwd_of(ob)      : float32 forward margin at the CROWN corner (increasing).

    The roots are near ob = -crown_lb_of(0) and ob = -fwd_of(0) (slope ~ 1), so
    we seed tight brackets around those.
    """
    lb0 = crown_lb_of(0.0)
    fwd0 = fwd_of(0.0)

    # quasi-ADVERSARIAL boundary: fwd_of(ob) = 0 (below -> mispredict)
    adv = _bisect_root(fwd_of, -fwd0 - bracket_halfwidth, -fwd0 + bracket_halfwidth,
                       tau_tol, max_iter)
    # SAFE boundary: crown_lb_of(ob) = 0 (above -> verifier certifies safe)
    safe = _bisect_root(crown_lb_of, -lb0 - bracket_halfwidth, -lb0 + bracket_halfwidth,
                        tau_tol, max_iter)

    adv_boundary = 0.5 * (adv["tau0"] + adv["tau1"])
    safe_boundary = 0.5 * (safe["tau0"] + safe["tau1"])
    fp_gap = adv_boundary - safe_boundary          # straddle-window width (DIAGNOSTIC)

    if fp_gap > 0.0:
        calibrated_ob = 0.5 * (safe_boundary + adv_boundary)   # window midpoint
        status = "calibrated"
    else:
        # no float32 straddle: sit just inside the adversarial side as a near-miss
        calibrated_ob = adv_boundary - 0.5 * tau_tol
        status = "no-straddle (near-miss)"

    crown_at = crown_lb_of(calibrated_ob)
    fwd_at = fwd_of(calibrated_ob)
    return {
        "status": status,
        "tau_tol": tau_tol,
        "adversarial_boundary": {
            "ob_root": adv_boundary, "tau0": adv["tau0"], "tau1": adv["tau1"],
            "tau1_minus_tau0": adv["width"], "iters": adv["iters"],
            "converged_below_tol": adv["converged"],
        },
        "safe_boundary": {
            "ob_root": safe_boundary, "tau0": safe["tau0"], "tau1": safe["tau1"],
            "tau1_minus_tau0": safe["width"], "iters": safe["iters"],
            "converged_below_tol": safe["converged"],
        },
        "fp_gap_diagnostic": {
            "value": fp_gap,
            "meaning": "adv_boundary - safe_boundary: width of the float32 "
                       "straddle window in bias space; NOT tau1 - tau0",
        },
        "calibrated_out_bias": calibrated_ob,
        "crown_lb_at_calibrated": crown_at,
        "fwd_margin_at_calibrated": fwd_at,
        "straddle_verified": bool(crown_at >= 0.0 > fwd_at),
    }
