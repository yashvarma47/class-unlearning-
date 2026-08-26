"""Privacy leakage metrics -- objective ``f3``.

The question ``f3`` asks: **can an attacker still tell that ``D_f`` was in the
training set?** If unlearning worked, no -- the attacker should do no better
than a coin flip.

Attack model
------------
A score-based membership inference attack. Every sample gets a scalar score from
a single forward pass, chosen so that *members should score higher*:

``neg_loss``
    ``-CE(f(x), y)``. Training samples are memorised, so their loss is low and
    ``-loss`` is high. This is the classic Yeom et al. (2018) signal.
``confidence``
    ``max softmax``. Memorised samples are predicted confidently.

Members are ``D_f``; non-members are a held-out subset of the test set that is
disjoint from the subset used for the utility objective, so ``f2`` and ``f3``
never share a sample. The AUC of the score distribution is then how well the
attacker separates the two.

Objective
---------
``obj3 = 2 * |AUC - 0.5|``, in ``[0, 1]``:

* ``0`` -- the attack is at chance level. **Best.**
* ``1`` -- the attack is perfect (AUC 1.0) *or* perfectly inverted (AUC 0.0).

The absolute value matters: an AUC far *below* 0.5 is also a leak. It means the
forgotten samples are now anomalously badly fit -- the "Streisand effect", where
aggressive unlearning makes ``D_f`` identifiable by how unusually wrong the
model is on it. A naive ``AUC - 0.5`` objective would reward that.

Pluggability
------------
Metrics are looked up by name from :data:`PRIVACY_METRICS`, with
``privacy_metric: mia_auc`` in the config. RMIA (~128 shadow models) is far too
expensive per chromosome on a 4 GB GPU and is intended as a post-hoc check on
the final Pareto front; when it lands it registers here and nothing else
changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.metrics import roc_auc_score

#: Score functions available to the attack. Higher score = more member-like.
SCORE_TYPES = ("neg_loss", "confidence")


@dataclass
class PrivacyResult:
    """Outcome of a membership inference attack."""

    metric: str
    score_type: str
    auc: float
    leakage: float
    n_members: int
    n_nonmembers: int
    #: AUC for every score type computed, for reporting and comparison.
    auc_by_score_type: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "score_type": self.score_type,
            "auc": self.auc,
            "leakage": self.leakage,
            "n_members": self.n_members,
            "n_nonmembers": self.n_nonmembers,
            "auc_by_score_type": self.auc_by_score_type,
        }


def leakage_from_auc(auc: float) -> float:
    """``obj3 = 2 * |AUC - 0.5|``, mapping any AUC into ``[0, 1]``.

    Symmetric about 0.5 on purpose -- see the module docstring on the Streisand
    effect.
    """
    return float(2.0 * abs(float(auc) - 0.5))


def membership_scores(
    per_sample_loss: np.ndarray,
    per_sample_confidence: np.ndarray,
    score_type: str = "neg_loss",
) -> np.ndarray:
    """Turn per-sample forward-pass statistics into attack scores.

    Raises
    ------
    KeyError
        If ``score_type`` is not one of :data:`SCORE_TYPES`.
    """
    if score_type == "neg_loss":
        return -np.asarray(per_sample_loss, dtype=np.float64)
    if score_type == "confidence":
        return np.asarray(per_sample_confidence, dtype=np.float64)
    raise KeyError(f"unknown score_type '{score_type}'; expected one of {SCORE_TYPES}")


def compute_mia_auc(
    member_loss: np.ndarray,
    member_confidence: np.ndarray,
    nonmember_loss: np.ndarray,
    nonmember_confidence: np.ndarray,
    score_type: str = "neg_loss",
) -> PrivacyResult:
    """Run the cheap forward-pass MIA and return its AUC and leakage.

    Every available score type is evaluated and reported in
    ``auc_by_score_type``; ``score_type`` selects which one drives ``obj3``.
    Computing both costs nothing extra -- the forward pass already happened --
    and having the pair is useful when they disagree.

    Raises
    ------
    ValueError
        If either group is empty. An AUC over one class is undefined, and
        returning ``nan`` would poison the objective silently.

    Notes
    -----
    Non-finite scores (from a diverged operator) are dropped before the AUC is
    computed, and the surviving counts are reported. If a whole group is lost
    this raises, which is the honest outcome: the model is too damaged for the
    attack to mean anything.
    """
    if len(member_loss) == 0 or len(nonmember_loss) == 0:
        raise ValueError(
            f"MIA needs both groups: got {len(member_loss)} members and "
            f"{len(nonmember_loss)} non-members"
        )

    auc_by_score_type: dict[str, float] = {}
    for candidate in SCORE_TYPES:
        member = membership_scores(member_loss, member_confidence, candidate)
        nonmember = membership_scores(nonmember_loss, nonmember_confidence, candidate)

        member = member[np.isfinite(member)]
        nonmember = nonmember[np.isfinite(nonmember)]
        if len(member) == 0 or len(nonmember) == 0:
            raise ValueError(
                f"all '{candidate}' scores were non-finite; the model is too "
                f"damaged for a membership attack to be meaningful"
            )

        labels = np.concatenate([np.ones(len(member)), np.zeros(len(nonmember))])
        scores = np.concatenate([member, nonmember])
        auc_by_score_type[candidate] = float(roc_auc_score(labels, scores))

    auc = auc_by_score_type[score_type]
    return PrivacyResult(
        metric="mia_auc",
        score_type=score_type,
        auc=auc,
        leakage=leakage_from_auc(auc),
        n_members=int(len(member_loss)),
        n_nonmembers=int(len(nonmember_loss)),
        auc_by_score_type=auc_by_score_type,
    )


def _rmia_not_implemented(*args: Any, **kwargs: Any) -> PrivacyResult:
    raise NotImplementedError(
        "RMIA requires ~128 shadow models, each a full training run. That is "
        "infeasible per chromosome on a 4 GB GPU, so it is reserved as a "
        "post-hoc check on the final Pareto front. Use privacy_metric='mia_auc' "
        "inside the search."
    )


#: name -> implementation. Config key: ``evaluation.privacy_metric``.
PRIVACY_METRICS: dict[str, Callable[..., PrivacyResult]] = {
    "mia_auc": compute_mia_auc,
    "rmia_auc": _rmia_not_implemented,
}


def get_privacy_metric(name: str) -> Callable[..., PrivacyResult]:
    """Look up a privacy metric by name.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in PRIVACY_METRICS:
        raise KeyError(
            f"unknown privacy metric '{name}'; available: {sorted(PRIVACY_METRICS)}"
        )
    return PRIVACY_METRICS[name]
