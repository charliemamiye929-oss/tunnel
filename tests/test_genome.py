import random

import pytest

from evotrader.genome import Genome, GenomeError, RiskParams, compile_genome, is_valid
from evotrader.population import (crossover, mutate, random_genome, seed_population)


def _genome(**over):
    base = {
        "name": "Dip Buyer",
        "thesis": "buy oversold pullbacks",
        "entry_rules": [{"when": "rsi14 < 35 and close > sma200", "weight": 0.3}],
        "exit_rules": ["rsi14 > 65"],
        "risk": {"max_position_pct": 0.35, "max_positions": 4},
    }
    base.update(over)
    return Genome.from_dict(base)


def test_roundtrip_through_dict():
    g = _genome()
    again = Genome.from_dict(g.to_dict())
    assert again.fingerprint() == g.fingerprint()
    assert again.entry_rules[0].when == g.entry_rules[0].when


def test_risk_params_are_clamped_to_sane_ranges():
    risk = RiskParams.from_dict({"max_position_pct": 99, "max_positions": -5,
                                 "stop_loss_pct": "nonsense", "cooldown_bars": 1e9})
    assert risk.max_position_pct == 1.0
    assert risk.max_positions == 1
    assert risk.stop_loss_pct == 0.10        # fell back to the default
    assert risk.cooldown_bars == 250


def test_rules_referencing_unknown_features_do_not_compile():
    with pytest.raises(GenomeError):
        compile_genome(_genome(entry_rules=[{"when": "alpha_signal > 1"}]))
    assert not is_valid(_genome(entry_rules=[]))
    assert not is_valid(_genome(exit_rules=[]))


def test_copy_gives_a_new_identity_and_leaves_the_parent_alone():
    parent = _genome()
    child = parent.copy(generation=3)
    child.entry_rules[0].when = "rsi14 < 10"
    assert child.id != parent.id
    assert parent.entry_rules[0].when == "rsi14 < 35 and close > sma200"
    assert child.generation == 3


def test_seed_population_is_valid_and_diverse():
    pop = seed_population(60, random.Random(0))
    assert len(pop) == 60
    assert all(is_valid(g) for g in pop)
    assert len({g.fingerprint() for g in pop}) > 40


def test_mutation_always_produces_a_valid_genome():
    rng = random.Random(7)
    parent = _genome()
    for _ in range(200):
        child = mutate(parent, rng, generation=1)
        assert is_valid(child)
        assert child.parents == [parent.id]


def test_crossover_mixes_both_parents_and_stays_valid():
    rng = random.Random(11)
    a, b = _genome(), _genome(name="Trend", entry_rules=[{"when": "close > sma50"}],
                              exit_rules=["close < sma50"])
    for _ in range(100):
        child = crossover(a, b, rng, generation=2)
        assert is_valid(child)
        assert set(child.parents) <= {a.id, b.id}


def test_random_genomes_compile():
    rng = random.Random(3)
    assert all(is_valid(random_genome(rng)) for _ in range(100))
