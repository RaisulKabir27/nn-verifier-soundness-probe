
from dataclasses import dataclass
import torch


@dataclass
class InputRegion:
    """An L-inf robustness query.

    center     : [1, *input_shape] tensor, the nominal input.
    eps        : float, L-inf radius; the box is [center-eps, center+eps].
    true_label : int, the class that must remain the arg-max for the property
                 to hold. The margin is  logit[true_label] - max_{j != true}.
    """
    center: torch.Tensor
    eps: float
    true_label: int = 0

    @property
    def data_min(self) -> torch.Tensor:
        """Lower corner of the box (center - eps)."""
        return self.center - self.eps

    @property
    def data_max(self) -> torch.Tensor:
        """Upper corner of the box (center + eps)."""
        return self.center + self.eps

    def clamp(self, x: torch.Tensor) -> torch.Tensor:
        """Project x back into the box."""
        return torch.max(torch.min(x, self.data_max), self.data_min)
