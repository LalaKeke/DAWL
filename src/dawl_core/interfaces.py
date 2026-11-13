"""Explicit integration boundaries for a new DAWL training application.

The source-only distribution does not include dataset-specific K8/fixed costs,
the certified Theseus teacher, R5 candidate search, or full recovery machinery.
Implementations must be supplied explicitly; there are no toy fallbacks.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn


class IntegrationRequired(RuntimeError):
    """An omitted research component must be integrated before training."""


@dataclass
class SceneBatch:
    """Keep physical raw values and normalized neural features separate.

    External actor/static/map masks use True=valid. ControlNet's low-level
    actor-mask API instead uses True=padding; the planner converts it.
    Future labels belong in ``OuterTargets``, never in either mapping here.
    """

    raw: Mapping[str, Any]
    features: Mapping[str, Tensor]
    tokens: Sequence[str]
    metadata: Optional[Mapping[str, Any]] = None

    def validate_no_future_labels(self) -> None:
        def inspect(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    label = str(key).lower()
                    if label.startswith("gt_") or label in {
                        "target", "targets", "expert_trajectory", "official_fut_valid_mask"
                    }:
                        raise ValueError("Future labels must be separate: " + str(key))
                    inspect(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    inspect(item)

        inspect(self.raw)
        inspect(self.features)
        if not self.tokens or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("A nonempty batch of unique tokens is required")


@dataclass
class OuterTargets:
    positions: Tensor  # [B,6,2], current-LiDAR sensor-origin cumulative XY
    valid: Tensor      # [B,6], observed timepoints, not collision absence


@dataclass
class CostBatch:
    """Original contributions, including base coefficients exactly once.

    Unknown entries may be NaN and must have valid=False. The backend owns
    the real geometry, validity domains, static coefficient and exact NoSC
    certificate; this interface invents none of those.
    """

    A: Tensor               # [B,8]
    valid: Tensor           # [B,8]
    fixed: Tensor           # [B,18]
    fixed_valid: Tensor     # [B,18]
    sc_certificate: Any = None

    def validate(self) -> None:
        if self.A.ndim != 2 or self.A.shape[1] != 8 or self.A.shape[0] < 1:
            raise ValueError("Eight ordered per-token K8 contributions required")
        if self.valid.shape != self.A.shape or self.valid.dtype != torch.bool:
            raise ValueError("K8 valid mask must be bool [B,8]")
        if self.fixed.shape != (len(self.A), 18):
            raise ValueError("Eighteen named per-token fixed contributions required")
        if self.fixed_valid.shape != self.fixed.shape or self.fixed_valid.dtype != torch.bool:
            raise ValueError("Fixed valid mask must be bool [B,18]")
        if not torch.isfinite(self.A[self.valid]).all() or not torch.isfinite(self.fixed[self.fixed_valid]).all():
            raise FloatingPointError("Nonfinite observed cost")
        if bool((self.A[self.valid] < 0).any()):
            raise ValueError("Difficulty expects nonnegative observed K8 costs")


@dataclass
class TeacherResult:
    outer: Tensor              # [B], true finite-LM GT outer values
    gradient_ratios: Tensor    # [B,8], d outer / d ratios
    diagnostics: Mapping[str, Any]

    def validate(self, ratios: Tensor) -> None:
        if self.outer.shape != (len(ratios),) or self.gradient_ratios.shape != ratios.shape:
            raise ValueError("Teacher outer/VJP shapes must preserve independent samples")
        if not torch.isfinite(self.outer).all() or not torch.isfinite(self.gradient_ratios).all():
            raise FloatingPointError("Nonfinite teacher outer or hypergradient")
        if self.gradient_ratios.dtype != ratios.dtype or self.gradient_ratios.device != ratios.device:
            raise ValueError("Teacher must explicitly validate its VJP cast/transport before this boundary")


class CostBackend(ABC):
    """Supply original K8/18-fixed equations and validity on actual scenes."""

    @abstractmethod
    def evaluate(self, scene: SceneBatch, control_forward: Mapping[str, Any]) -> CostBatch:
        raise NotImplementedError


class ExpertTeacher(ABC):
    """Supply the real finite-3-step LM inner64 teacher; never a GT shortcut.

    Implementations detach the initial q and environment, solve with the
    original objective/acceptance rule, then apply GT only to the outer loss.
    They return the real per-sample VJP with finite/FP32-cast checks.
    """

    @abstractmethod
    def evaluate(
        self, scene: SceneBatch, normalized_controls: Tensor,
        ratios: Tensor, targets: OuterTargets,
    ) -> TeacherResult:
        raise NotImplementedError


class ControlUpdatePolicy(ABC):
    """Own real candidate validation and Control commit, e.g. current R5.

    Do not label an ordinary Adam step as R5. Its implementation must preserve
    all74-before20 clipping, frozen54 grad=None, native moments, finite search
    and rejection. A failed candidate stops the attempt; there is no skip.
    """

    @abstractmethod
    def prepare(self, prepared: Any, optimizer: Any) -> Any:
        """Validate a candidate in isolated copies, without a live step."""
        raise NotImplementedError

    @abstractmethod
    def commit_control(self, proposal: Any, optimizer: Any) -> None:
        """Perform one native step and any verified selected-parameter copy."""
        raise NotImplementedError


class CompleteStateTransaction(ABC):
    """Required external state/WAL/physical-event implementation.

    Includes both models/Adams/schedulers, history/EMA, sampler, modes and
    CPU/CUDA random streams. Abort preserves physical INTENT/RETURN evidence
    and consumed attempts; catastrophic loss is not called a full rollback.
    """

    @abstractmethod
    def begin(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def intent(self, optimizer_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def returned(self, optimizer_name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def validate_after_updates(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def commit(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def abort(self, error: BaseException) -> None:
        raise NotImplementedError


@dataclass
class TrainingComponents:
    """Explicit trusted plugin bundle used by scripts/train.py.

    The runner and the scientific integrations are intentionally not bundled.
    Validation checks interfaces, not scientific correctness or authorization.
    """

    costs: CostBackend
    control_policy: ControlUpdatePolicy
    transaction: CompleteStateTransaction
    runner: Callable[["TrainingComponents", Mapping[str, Any]], Mapping[str, Any]]
    expert_teacher: Optional[ExpertTeacher] = None

    def validate(self, supervision: str) -> None:
        if supervision not in ("theseus_gt", "difficulty_nosc"):
            raise ValueError("Use an explicit Weight supervision kind")
        for value, kind in ((self.costs, CostBackend), (self.control_policy, ControlUpdatePolicy),
                            (self.transaction, CompleteStateTransaction)):
            if not isinstance(value, kind):
                raise IntegrationRequired("Required integration missing: " + kind.__name__)
        if not callable(self.runner):
            raise IntegrationRequired("A dataset-aware training runner is required")
        if supervision == "theseus_gt" and not isinstance(self.expert_teacher, ExpertTeacher):
            raise IntegrationRequired("Real Theseus ExpertTeacher is required for the expert group")
