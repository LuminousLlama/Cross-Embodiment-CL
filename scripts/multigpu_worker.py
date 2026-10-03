"""Per-rank worker for multi-GPU RSL-RL training of this project's tasks.

Launch through ``torch.distributed.run`` in place of Isaac Lab's ``multigpu.py`` worker. It fixes two
problems with running this task across GPUs:

1. Isaac Lab's per-rank worker never loads the ``isaaclab.tasks`` entry points that the single-GPU
   CLI loads, so the task is not registered on the ranks.
2. rsl_rl 5.4.1 updates ``EmpiricalNormalization`` from each rank's local batch only. With per-rank
   ADR the ranks see different observation distributions, so their normalizers drift apart while the
   gradients are averaged; a 3-GPU teacher stalled at ADR 40 and collapsed. Here every update uses
   batch statistics pooled over all ranks, which keeps the normalizers identical on every rank, as
   in a single-GPU run.
"""

import warp as wp

# Warp captures ``enable_backward`` when a module is created, so it must be set before importing anything
# that defines Warp kernels, as Isaac Lab's own worker does.
wp.config.enable_backward = False

import importlib.metadata  # noqa: E402
import importlib.util  # noqa: E402
import runpy  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from rsl_rl.modules import normalization  # noqa: E402

_local_update = normalization.EmpiricalNormalization.update


def _rank_synced_update(self, x: torch.Tensor) -> None:
    """Update the running statistics from the batch pooled over all ranks."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return _local_update(self, x)
    if not self.training:
        return
    if self.until is not None and self.count >= self.until:
        return
    values = x.double()
    packed = torch.cat([values.new_tensor([x.shape[0]]), values.sum(dim=0), (values * values).sum(dim=0)])
    dist.all_reduce(packed)
    dim = x.shape[1]
    count = packed[0]
    pooled_mean = packed[1 : 1 + dim] / count
    mean_x = pooled_mean.unsqueeze(0).to(x.dtype)
    var_x = (packed[1 + dim :] / count - pooled_mean**2).clamp_min(0.0).unsqueeze(0).to(x.dtype)
    count_x = int(count.item())
    self.count += count_x
    rate = count_x / self.count
    delta_mean = mean_x - self._mean
    self._mean += rate * delta_mean
    self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
    self._std = torch.sqrt(self._var)


normalization.EmpiricalNormalization.update = _rank_synced_update

for entry_point in importlib.metadata.entry_points(group="isaaclab.tasks"):
    entry_point.load()

runpy.run_path(importlib.util.find_spec("isaaclab_rl.entrypoints.multigpu").origin, run_name="__main__")
