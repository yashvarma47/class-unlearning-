"""The operator library and the chromosome that addresses it.

The properties pinned here are the ones a silent regression would make
unfalsifiable later: that the unsafe operators really are gone, that an operator
only touches its own layer group, and that the data-free controls stay data-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import medus_class.operators as ops  # noqa: E402
from medus_class.operators.selection import SELECTION_RULES  # noqa: E402
from medus_class.search import Chromosome, ChromosomeBounds, decode  # noqa: E402

GROUPS = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]

SAFE = {"MASK", "PRUNE", "RANDOM_PRUNE", "DAMP", "NOISE", "CLIP", "QUANTIZE", "RESET"}
BANNED = {"REINIT", "SIGN_FLIP"}


# --- the library ----------------------------------------------------------

def test_library_holds_exactly_the_safe_eight():
    present = set(ops.operator_names("editor")) | set(ops.operator_names("smoother"))
    assert present == SAFE


def test_unsafe_operators_are_absent_entirely():
    """Not disabled by config -- absent, so no config edit can restore them."""
    present = set(ops.operator_names("editor")) | set(ops.operator_names("smoother"))
    assert not (present & BANNED)

    implementations = set(ops.EDITOR_OPERATORS) | set(ops.SMOOTHER_OPERATORS)
    assert not (implementations & BANNED)


def test_every_operator_is_selectable():
    for family in ("editor", "smoother"):
        assert ops.operator_ids(family) == ops.selectable_operator_ids(family)


def test_operator_ids_are_contiguous_from_zero():
    for family in ("editor", "smoother"):
        assert ops.operator_ids(family) == list(range(ops.n_operators(family)))


def test_data_free_controls_keep_their_own_rules():
    """PRUNE and RANDOM_PRUNE are the baselines; they must not read D_f."""
    assert ops.EDITOR_OPERATORS["PRUNE"].selection_rule == "magnitude"
    assert ops.EDITOR_OPERATORS["RANDOM_PRUNE"].selection_rule == "random"
    assert ops.EDITOR_OPERATORS["PRUNE"].select_largest is False


def test_mask_uses_the_class_contrast_selector():
    assert ops.EDITOR_OPERATORS["MASK"].selection_rule == "class_contrast"


def test_selection_rules_are_the_three_expected():
    assert set(SELECTION_RULES) == {"class_contrast", "magnitude", "random"}


# --- intensity ------------------------------------------------------------

def test_level_zero_means_off_and_refuses_to_resolve():
    with pytest.raises(ValueError, match="OFF"):
        ops.resolve_hparams("editor", 0, ops.LEVEL_OFF)


@pytest.mark.parametrize("family", ["editor", "smoother"])
def test_every_operator_defines_five_levels(family):
    for operator_id in ops.operator_ids(family):
        for level in range(1, ops.MAX_LEVEL + 1):
            hparams = ops.resolve_hparams(family, operator_id, level)
            assert "ratio" in hparams


def test_intensity_is_monotone_in_ratio():
    """A higher level must never touch fewer connections."""
    for family in ("editor", "smoother"):
        for operator_id in ops.operator_ids(family):
            ratios = [ops.resolve_hparams(family, operator_id, lvl)["ratio"]
                      for lvl in range(1, ops.MAX_LEVEL + 1)]
            assert ratios == sorted(ratios), (
                f"{ops.operator_spec(family, operator_id)['name']} ratios "
                f"are not monotone: {ratios}"
            )


def test_level_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        ops.resolve_hparams("editor", 0, ops.MAX_LEVEL + 1)


def test_unknown_family_is_rejected():
    with pytest.raises(KeyError, match="unknown operator family"):
        ops.operator_ids("gradient")   # the predecessor's name, now invalid


# --- the chromosome -------------------------------------------------------

@pytest.fixture
def bounds():
    return ChromosomeBounds.from_registry(n_groups=6, implemented_only=True)


def test_bounds_match_the_library(bounds):
    assert bounds.n_editor_operators == 3
    assert bounds.n_smoother_operators == 5
    assert bounds.n_genes == 30          # 5 genes x 6 groups


def test_identity_chromosome_is_a_no_op(bounds):
    strategy = decode(Chromosome.identity(bounds), GROUPS)
    assert strategy.is_no_op
    assert strategy.actions == []


def test_random_chromosome_decodes_to_known_operators(bounds):
    rng = np.random.default_rng(0)
    for _ in range(20):
        strategy = decode(Chromosome.random(bounds, rng, p_active=0.7), GROUPS)
        for action in strategy.actions:
            assert action.family in ("editor", "smoother")
            assert action.operator_name.split("@")[0] in SAFE
            assert action.group_name in GROUPS
            assert 1 <= action.level <= ops.MAX_LEVEL


def test_max_level_caps_the_ladder():
    """The Plan A config restricts to the two gentlest rungs."""
    capped = ChromosomeBounds.from_registry(
        n_groups=6, implemented_only=True, max_level=2
    )
    rng = np.random.default_rng(1)
    for _ in range(20):
        strategy = decode(Chromosome.random(capped, rng, p_active=0.9), GROUPS)
        for action in strategy.actions:
            assert action.level <= 2


def test_vector_roundtrip(bounds):
    rng = np.random.default_rng(2)
    chromosome = Chromosome.random(bounds, rng, p_active=0.5)
    restored = Chromosome.from_vector(chromosome.to_vector(), bounds)
    assert restored == chromosome


# --- execution ------------------------------------------------------------

def test_operator_only_touches_its_own_group():
    """The rule the whole layer-wise chromosome depends on."""
    from medus_class.models import build_model, build_registry
    from medus_class.utils.config import load_config

    model_cfg = load_config("model/resnet18.yaml")["model"]
    model = build_model(model_cfg, num_classes=10)
    registry = build_registry(model, model_cfg)

    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    target = registry["layer3"]
    operator = ops.build_operator("editor", 0)          # MASK
    context = ops.OperatorContext(
        loaders=None, registry=registry, device="cpu", num_classes=10, seed=42,
        batch_cap=1, selection_rule="magnitude",        # data-free: needs no loaders
    )
    context.take_snapshot(model, target)
    operator.apply(model, context, target, {"ratio": 0.5})

    after = model.state_dict()
    changed = {name for name in before
               if not torch.equal(before[name], after[name])}

    assert changed, "the operator did nothing at all"
    for name in changed:
        assert name.startswith("layer3"), (
            f"operator on layer3 also modified {name}"
        )
