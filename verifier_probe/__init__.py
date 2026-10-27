import os, sys

def _bootstrap_auto_lirpa():

    for p in (os.environ.get("AUTO_LIRPA_PATH", ""),
              os.path.expanduser("~/ref/abc/auto_LiRPA"),
              "/home/claude/ref/abc/auto_LiRPA"):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            return p
    return None

_bootstrap_auto_lirpa()

from .region import InputRegion
from .verifier import ABCrownVerifier, Certificate
from .attacker import Attacker
from .prober import Prober, TRACE_SCHEMA
from .calibration import exact_preact_bounds, calibrate_output_bias
from . import models

__all__ = ["Attacker", "Prober", "ABCrownVerifier", "Certificate",
           "InputRegion", "TRACE_SCHEMA", "models",
           "exact_preact_bounds", "calibrate_output_bias"]
